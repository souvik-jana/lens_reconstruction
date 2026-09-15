"""
Ultra-simple default-driven pipeline for GWEMFISH.

User-facing API (intended usage):
  1) ctx = setup_em_observation(cfg={})
  2) ctx = setup_gw_observation(ctx, cfg={})
  3) samples, truths = run_inference(ctx, mode='EM+GW', method='deriv-approx', cfg={})
     ``deriv-approx`` uses the Taylor (banana) model only. ``hmc`` / ``hmc-informed`` use the full
     ``ProbModel`` likelihood; optional ``cfg['inference']['H0']`` overrides the Fisher Hessian for
     informed NUTS. See ``run_inference`` docstring.
  4) plot_posterior(samples, truths, cfg={})
  5) source_samples = to_source_plane_samples(samples, ctx, cfg={})
  6) plot_source_posterior(source_samples, truths, cfg={})

This module intentionally keeps a high-level configuration surface:
  - Almost everything can be overridden via a single `cfg` dict.
  - `cfg['priors']` supports callables, fixed values, or numpyro distribution objects.
  - Set `cfg['use_parameter_layout']=True` for flat names `lens0_*`, `source0_*`, `light0_*`
    and defaults from ``gwemfish.profile_prior_rules`` (see ``parameter_layout`` / ``flex_prob_model``).
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from numpyro.handlers import trace, substitute, seed


def _deep_merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively deep-merge dictionaries (override wins)."""
    if not override:
        return base
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def deep_merge_cfg(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins on conflicts).

    Same policy as ``setup_em_observation``, ``setup_gw_observation``, and
    ``run_inference``: nested dict sections are merged; leaves replace outright.
    If ``override`` is ``None``, empty, or not a mapping (e.g. missing JSON field),
    returns a fresh deep copy of ``base``.
    """
    if not isinstance(override, dict) or not override:
        return copy.deepcopy(base)
    return _deep_merge_dict(base, override)


def restore_em_model_factories(cfg, *, defaults=None):
    """Reattach EM ``source_model_class`` / ``lens_light_model_class`` callables after JSON load.

    ``to_serializable`` stores callables as ``repr`` strings; merging a saved bundle
    therefore breaks ``setup_em_observation``. This overlays the live factories from
    ``defaults`` (normally ``make_default_cfg()``) when the merged value is not callable.
    """
    ref = defaults if isinstance(defaults, dict) else make_default_cfg()
    em = cfg.get("em") if isinstance(cfg, dict) else None
    ref_em = ref.get("em") if isinstance(ref.get("em"), dict) else {}
    if not isinstance(em, dict):
        return
    for key in ("source_model_class", "lens_light_model_class"):
        if not callable(em.get(key)) and callable(ref_em.get(key)):
            em[key] = ref_em[key]


def _save_dict_npz(path: Optional[str], data: Dict[str, Any]) -> None:
    """Save a flat dictionary to .npz if path is provided."""
    if not path:
        return
    import numpy as np

    np.savez(path, **{k: np.asarray(v) for k, v in data.items()})


def _resolve_output_path(path: Optional[str], output_dir: Optional[str]) -> Optional[str]:
    """Resolve save path under output_dir for relative paths."""
    if not path:
        return None
    import os

    if os.path.isabs(path) or not output_dir:
        return path
    return os.path.join(output_dir, path)


def _output_json_path(out_cfg: Dict[str, Any]) -> Optional[str]:
    """Pipeline JSON path: ``json_path`` (preferred) or legacy ``save_pipeline_json_path``."""
    jp = out_cfg.get("json_path")
    if jp is not None:
        return jp
    return out_cfg.get("save_pipeline_json_path")


def _output_json_tag(out_cfg: Dict[str, Any]) -> Optional[str]:
    """Optional tag for pipeline JSON: ``json_tag`` or legacy ``save_pipeline_json_tag``."""
    jt = out_cfg.get("json_tag")
    if jt is not None:
        return jt
    return out_cfg.get("save_pipeline_json_tag")


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create parent directory for a file path if needed."""
    if not path:
        return
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_tag_to_path(path: Optional[str], tag: Optional[str]) -> Optional[str]:
    """Append a tag before file extension, e.g. a.npz -> a_tag.npz."""
    if not path or not tag:
        return path
    import os

    root, ext = os.path.splitext(path)
    safe_tag = str(tag).strip().replace("-", "_").replace(" ", "_")
    return f"{root}_{safe_tag}{ext}"


def _prefer_existing_tagged_json(path: Optional[str]) -> Optional[str]:
    """Prefer an existing tagged sibling JSON if base path does not exist.

    Example:
      base: outputs/pipeline_outputs.json
      existing sibling: outputs/pipeline_outputs_hmc_informed.json
      -> returns sibling path so later stages update same file incrementally.
    """
    if not path:
        return None
    import glob
    import os

    if os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    candidates = sorted(glob.glob(f"{root}_*{ext}"))
    if not candidates:
        return path
    return max(candidates, key=os.path.getmtime)


def _to_serializable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if callable(obj):
        return repr(obj)
    try:
        return np.asarray(obj).tolist()
    except Exception:
        return repr(obj)


def to_serializable(obj: Any) -> Any:
    """Best-effort convert ``obj`` to JSON-serializable data (pipeline JSON payloads).

    This is the public alias of ``_to_serializable``; behavior is identical.
    """
    return _to_serializable(obj)


def _save_pipeline_json(
    path: Optional[str],
    *,
    ctx: Optional[Dict[str, Any]] = None,
    samples_image_plane: Optional[Dict[str, Any]] = None,
    truths_image_plane: Optional[Dict[str, Any]] = None,
    source_plane_samples: Optional[Dict[str, Any]] = None,
) -> None:
    """Save key setup/injection/sample artifacts as JSON if path is provided."""
    if not path:
        return
    import json

    fisher = (ctx or {}).get("fisher") or {}
    likelihood = (ctx or {}).get("likelihood") or {}
    # g0/H0 at the truth point are the record of whether the expansion was built
    # somewhere sensible; without them a saved run cannot be re-checked later.
    fisher_block = None
    if fisher:
        fisher_block = {
            "keys": likelihood.get("keys_to_include"),
            "u0": likelihood.get("u0"),
            "logp0": fisher.get("logp0"),
            "g0": fisher.get("g0"),
            "g0_scaled": fisher.get("g0_scaled"),
            "H0": fisher.get("H0"),
        }

    payload = {
        "injection_parameters": (ctx or {}).get("truth_params"),
        "setup_parameters": {
            "cfg": (ctx or {}).get("cfg"),
            "kwargs_lens": (ctx or {}).get("kwargs_lens"),
            "lens_model_list": (ctx or {}).get("lens_model_list"),
            "n_images": (ctx or {}).get("n_images"),
        },
        "samples_image_plane": samples_image_plane,
        "truths_image_plane": truths_image_plane,
        "samples_source_plane": source_plane_samples,
        "fisher": fisher_block,
        "diagnostics": (ctx or {}).get("diagnostics"),
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _fisher_covariance(FM, keys=None):
    """Invert the Fisher matrix in a whitened basis, then check it is usable.

    ``u0`` is in raw physical units, so the parameters span many orders of magnitude
    (dL ~ 3e4 Mpc, T_star ~ 3e7 s, e2 ~ 0.6). A direct ``inv`` on that matrix can have
    a condition number around 1e20 -- past what float64 can invert -- and returns a
    covariance with small *negative* eigenvalues, which makes
    ``multivariate_normal`` produce NaN samples with no error raised.

    Rescaling each row/column by 1/sqrt(|FM_ii|) removes the unit disparity, which is
    almost all of the ill-conditioning: on catalog system 555 it takes the condition
    number from 8e19 to 8e12, comfortably invertible. The scaling cancels exactly, so
    this is the same covariance, just computed without the round-off.
    """
    import numpy as _np
    import jax.numpy as jnp

    FM_np = _np.asarray(FM, dtype=float)
    diag = _np.abs(_np.diag(FM_np))
    scale = _np.where(diag > 0, 1.0 / _np.sqrt(_np.where(diag > 0, diag, 1.0)), 1.0)

    FM_scaled = FM_np * scale[:, None] * scale[None, :]
    try:
        cov_scaled = _np.linalg.inv(FM_scaled)
    except _np.linalg.LinAlgError:
        cov_scaled = _np.linalg.pinv(FM_scaled)
    cov = cov_scaled * scale[:, None] * scale[None, :]
    cov = 0.5 * (cov + cov.T)

    eig = _np.linalg.eigvalsh(cov)
    if eig.min() <= 0:
        worst = keys[int(_np.argmin(_np.abs(_np.linalg.eigvalsh(FM_np))))] if keys else "?"
        warnings.warn(
            f"Fisher covariance is not positive definite (min eigenvalue "
            f"{eig.min():.3e}); the Fisher matrix is singular to working precision "
            f"along a near-degenerate direction (weakest parameter: {worst}). This "
            "usually means the observation does not constrain every free parameter -- "
            "e.g. a 3-image system gives 2 time delays + 3 dL_eff = 5 observables, so "
            "5 free parameters leave nothing over. Fix one parameter via cfg['priors'] "
            "or add EM data. Clipping the non-positive eigenvalues so sampling can "
            "proceed, but treat the widths along those directions as unbounded.",
            UserWarning, stacklevel=2,
        )
        vals, vecs = _np.linalg.eigh(cov)
        floor = max(eig.max(), 1.0) * 1e-12
        cov = (vecs * _np.maximum(vals, floor)) @ vecs.T
        cov = 0.5 * (cov + cov.T)
    return jnp.asarray(cov)


def make_default_cfg() -> Dict[str, Any]:
    """Return a default configuration dict for the pipeline."""
    # Lazy imports so this module stays lightweight.
    from .config import (
        DEFAULT_PIXEL_GRID_KWARGS,
        DEFAULT_PSF_KWARGS,
        DEFAULT_NOISE_KWARGS_SIMU,
        DEFAULT_NOISE_KWARGS_INFERENCE,
        DEFAULT_LENS_MODEL_LIST,
        DEFAULT_KWARGS_LENS,
        DEFAULT_ZL,
        DEFAULT_ZS,
        DEFAULT_SOURCE_POS_EM,
        DEFAULT_SOURCE_POS_GW,
        DEFAULT_KWARGS_SOURCE,
        DEFAULT_KWARGS_LENS_LIGHT,
        DEFAULT_KWARGS_NUMERICS,
        DEFAULT_SOURCE_LIGHT_MODEL,
        DEFAULT_LENS_LIGHT_MODEL,
    )
    from .config import SOLVER_PARAMS

    # Deep-copied so a caller mutating cfg["gw"]["solver_params"]["helens"][...] cannot
    # reach back into the module-level defaults and affect every later cfg.
    _gw_solver_params = copy.deepcopy(SOLVER_PARAMS)
    return {
        "jax": {
            "ncpus": None,  # None => use all available
            "enable_x64": True,
            "platform": "cpu",
            "verbose": True,
        },
        "em": {
            "enabled": True,
            "pixel_grid_kwargs": DEFAULT_PIXEL_GRID_KWARGS,
            "psf_kwargs": DEFAULT_PSF_KWARGS,
            "noise_simu_kwargs": DEFAULT_NOISE_KWARGS_SIMU,
            "noise_inf_kwargs": DEFAULT_NOISE_KWARGS_INFERENCE,
            "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
            "exposure_time": DEFAULT_NOISE_KWARGS_INFERENCE["exposure_time"],
            "source_pos": DEFAULT_SOURCE_POS_EM,
            "kwargs_source": DEFAULT_KWARGS_SOURCE,
            "kwargs_lens_light": DEFAULT_KWARGS_LENS_LIGHT,
            # `DEFAULT_*_LIGHT_MODEL` are *callables* returning herculens model instances.
            "source_model_class": DEFAULT_SOURCE_LIGHT_MODEL,
            "lens_light_model_class": DEFAULT_LENS_LIGHT_MODEL,
            "seed": 87651,
        },
        "gw": {
            "enabled": True,
            "n_images": 4,
            "source_pos": DEFAULT_SOURCE_POS_GW,
            "cosmology": {"H0": 67.3, "Om0": 0.316},
            # Merges jaxtronomy image-position defaults (``solver``, grid kwargs) with Helens keys.
            "solver_params": _gw_solver_params,
            # GW likelihood multipliers (see ``ProbModel``): sigma_td * gw_obs['time_delays'],
            # sigma_dL_eff * gw_obs['dL_eff'], epsilon * ones_like(betx_x_diff).
            "error_scales": {
                "sigma_td": 0.05,
                "sigma_dL_eff": 0.2,
                "epsilon": 0.005,
            },
            # For GW-only we constrain each image_x/i and image_y/i by a uniform box around truth.
            "image_box_half_width": 0.6,
            # Legacy MST location (kept for backwards compatibility).
            "use_mst": False,
            "k_mst": 0.0,
        },
        # Global Mass Sheet Transform (MST) controls (preferred over ``cfg['gw']['use_mst']/['k_mst']``).
        # Applies to all modes (EM-only, GW-only, EM+GW) where relevant.
        "mst": {
            "enabled": False,
            "k_mst": 0.0,
        },
        "lens": {
            "lens_model_list": DEFAULT_LENS_MODEL_LIST,
            "kwargs_lens": DEFAULT_KWARGS_LENS,
            "zl": DEFAULT_ZL,
            "zs": DEFAULT_ZS,
        },
        # Inferred parameter priors overrides (optional).
        # Values can be:
        # - callable: numpyro-style zero-arg callable returning numpyro.sample(...)
        # - fixed scalar/array: will be wrapped into a constant callable
        # - numpyro Distribution instance: will be wrapped into a callable sampling it.
        "priors": {},
        "inference": {
            "num_warmup": 6000,
            "num_samples": 12000,
            "num_chains": 2,
            "max_tree_depth": 10,
            "dense_mass": True,
            # Hessian-informed NUTS controls (mass matrix from Fisher ``H0`` unless ``H0`` is set).
            "hmc_informed_scale": 1.0,
            "hmc_informed_perturb_scale": 0.1,
            # Optional: ``True`` enables informed NUTS for ``hmc`` (full model) or ``deriv-approx``
            # (banana model). Ignored for ``hmc-informed`` (always informed) and ``fisher``.
            "informed": None,
            # Optional (n, n) Hessian override for informed NUTS; None uses Fisher ``H0`` from ``compute_fisher``.
            "H0": None,
            # Gaussian (Hessian) sampling uses `n_fisher_samples`.
            "n_fisher_samples": 5000,
            "fisher_order": 2,
            # After evaluating g/H at the given (truth) values, Newton–Raphson
            # jumps until MAP (|g|<grad_tol), then re-expand Fisher there.
            # max_jumps=None => keep jumping until convergence (safety cap inside).
            "newton_maxp": {
                "enabled": True,
                "max_jumps": None,
                "grad_tol": 5e-2,
                "verbose": True,
                "line_search": True,
                # Floor positive-at-start params; pin *sigma* at given (noiseless-safe).
                "apply_default_floors": True,
                "floor_rel": 1e-3,
                "floor_abs": 1e-12,
            },
            "rng_key": 123,
            # How the one-shot get_sample() PRNGKey is seeded.
            "prior_sample_rng_key": 123,
            # Pre-inference checks at the truth point: image positions, observables,
            # source-box vs caustic, gradient. "raise" | "warn" | "off".
            # "off" is unsafe for the *-source methods: their Fisher expansion is
            # built at truth, so a broken solve there corrupts everything downstream
            # with nothing to signal it.
            "diagnostics": "warn",
            # Per-check thresholds; anything omitted keeps gwemfish's default.
            # See gwemfish.diagnostics.DEFAULT_THRESHOLDS for the values and how they
            # were calibrated. Keys: position_tol (arcsec), observable_rtol,
            # condition_limit, gradient_sigma.
            "diagnostics_thresholds": {},
        },
        "plot": {
            "plot_mode": "groupwise",  # groupwise | combined | subset
            "color": "#2c3e50",
            "truth_color": "red",
            "show_titles": True,
            "title_kwargs": {"fontsize": 10},
            "title_fmt": ".3f",
            "quantiles": [0.05, 0.5, 0.975],
            # Passed to ``corner.corner(..., hist_kwargs=...)`` (matplotlib 1D histograms).
            "hist_kwargs": {"density": True},
            "params_to_plot": None,
            "figsize": None,
            "save_path": None,
            # Optional suffix tag appended before extension on save.
            # e.g. "corner_{group_name}.png" + "hmc" -> "corner_{group_name}_hmc.png"
            "save_tag": None,
            # plot_system_observation_pal (opt-in; see cfg_reference PAL mirror section)
            "pal_plot_dataset": True,
            "pal_plot_tracer": True,
            "pal_dataset": "both",
        },
        "source_plane": {
            # Unused by ``to_source_plane_samples`` (image count follows ``_resolve_gw_n_images``).
            "n_images": 4,
            "n_subsample": None,
            "seed": 42,
            "filter_std": None,
            "use_filtered": False,
        },
        "output": {
            "output_dir": "outputs",
            # Optional output data paths (.npz).
            "save_samples_path": None,
            "save_truths_path": None,
            "save_source_samples_path": None,
            "save_system_plot_path": None,
            "save_psf_plot_path": None,
            "save_pal_dataset_plot_path": None,
            "save_pal_tracer_plot_path": None,
            "json_path": None,
            # ``plot_system_observation``: ``"gw"`` (default), ``"em"``, ``"both"``, or ``"none"``.
            "system_plot_image_overlay": "gw",
        },
        # Use ``lens0_*``, ``source0_*``, ``light0_*`` names and ``profile_prior_rules`` defaults.
        "use_parameter_layout": False,
    }


def _normalize_priors_overrides(
    priors_override: Dict[str, Any],
    *,
    jnp: Any,
    numpyro: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Convert `cfg['priors']` into a numpyro-style priors registry (zero-arg callables)."""
    normalized: Dict[str, Any] = {}
    for name, value in (priors_override or {}).items():
        # Distributions first: in numpyro they are callable too, so must precede callable().
        if isinstance(value, dist.Distribution):
            normalized[name] = (lambda n=name, d=value: numpyro.sample(n, d))
            continue
        # Numpyro-style zero-arg callables (custom priors).
        if callable(value):
            normalized[name] = value
            continue

        # Fixed scalar/array => constant callable (no sampling site).
        normalized[name] = (lambda v=value: jnp.asarray(v))

    return normalized


def _fixed_literal_prior_keys(priors_user: Optional[Dict[str, Any]]) -> set[str]:
    """Names in ``cfg['priors']`` given as fixed scalars/arrays (not callables / Distributions)."""
    if not priors_user:
        return set()
    import numpy as np
    import numpyro.distributions as dist

    out: set[str] = set()
    for k, v in priors_user.items():
        if isinstance(v, dist.Distribution):
            continue
        if callable(v):
            continue
        try:
            arr = np.asarray(v)
            if arr.dtype == object:
                continue
            if arr.size >= 1:
                out.add(k)
        except (TypeError, ValueError):
            continue
    return out


def _float_from_user_prior_literal(v: Any) -> float:
    import numpy as np

    return float(np.asarray(v).reshape(-1)[0])


def _merge_flex_priors_with_gw_image_pos(
    priors_reg: Dict[str, Any],
    priors_internal: Dict[str, Any],
    priors_image_pos: Dict[str, Any],
    priors_user_norm: Dict[str, Any],
) -> Dict[str, Any]:
    """Do not overwrite user image priors with default tight GW uniforms (flex layout paths)."""
    out = {**priors_reg, **priors_internal}
    for k, v in priors_image_pos.items():
        if k not in priors_user_norm:
            out[k] = v
    return out


def _resolve_mst_settings(
    cfg_full: Dict[str, Any],
    *,
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float]:
    """Resolve MST settings from top-level cfg with backward-compatible fallbacks."""
    mst_cfg = (cfg_full.get("mst") or {}) if isinstance(cfg_full, dict) else {}
    gw_cfg = (cfg_full.get("gw") or {}) if isinstance(cfg_full, dict) else {}
    ctx = ctx or {}

    # Preferred: top-level ``mst`` block.
    mst_enabled = bool(mst_cfg.get("enabled", False))
    mst_k = float(mst_cfg.get("k_mst", 0.0))

    # Backward-compatible legacy GW keys.
    gw_enabled = bool(gw_cfg.get("use_mst", False))
    gw_k = float(gw_cfg.get("k_mst", 0.0))

    # If top-level mst is explicitly enabled, it wins. Otherwise honor legacy gw flags.
    use_mst = mst_enabled or gw_enabled or bool(ctx.get("use_mst", False))

    if mst_enabled:
        k_mst = mst_k
    elif gw_enabled:
        k_mst = gw_k
    else:
        k_mst = mst_k if "k_mst" in mst_cfg else float(ctx.get("k_mst", 0.0))

    return use_mst, k_mst


# Legacy flat ``truth_params`` need these keys even when ``lens_model_list`` has no EPL/SIE
# (e.g. SIS-only, NFW, flex layout). Layout mode still overwrites via ``truth_vector_from_kwargs``.
_LEGACY_TRUTH_MASS_KW_FALLBACK: Dict[str, Any] = {
    "theta_E": 1.0,
    "e1": 0.0,
    "e2": 0.0,
    "gamma": 2.0,
    "center_x": 0.0,
    "center_y": 0.0,
}


def _legacy_epl_like_kw(kwargs_lens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick main mass dict for legacy ``lens_*`` truth keys; never raise if EPL is absent."""

    def _is_shear_kw(kw: Dict[str, Any]) -> bool:
        return "gamma1" in kw and "theta_E" not in kw

    for kw in kwargs_lens:
        if _is_shear_kw(kw):
            continue
        if "theta_E" in kw and "gamma" in kw:
            return kw
    for kw in kwargs_lens:
        if _is_shear_kw(kw):
            continue
        if "theta_E" in kw:
            return kw
    return dict(_LEGACY_TRUTH_MASS_KW_FALLBACK)


def _legacy_shear_kw(kwargs_lens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the SHEAR-like component, or zero shear if the mass list has no such profile."""
    for kw in kwargs_lens:
        if "gamma1" in kw:
            return kw
    return {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0}


# Explicit defaults per mass profile so sparse user ``kwargs_lens`` match what the
# jaxtronomy/lenstronomy solver and herculens use implicitly. User entries override.
# ``setup_lens`` returns kwargs unchanged; merging here makes ``ctx['kwargs_lens']`` the
# single source of truth for legacy ``truth_params`` (no duplicate magic numbers).
_DEFAULT_KWARGS_BY_LENS_MODEL: Dict[str, Dict[str, float]] = {
    "EPL": {"e1": 0.0, "e2": 0.0, "gamma": 2.0, "center_x": 0.0, "center_y": 0.0},
    "SIE": {"e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
    "SIS": {"center_x": 0.0, "center_y": 0.0},
    "SHEAR": {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
    "NIE": {"e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
}


def _kwargs_lens_with_explicit_defaults(
    lens_model_list: List[str], kwargs_lens: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if len(kwargs_lens) != len(lens_model_list):
        raise ValueError(
            f"len(kwargs_lens)={len(kwargs_lens)} != len(lens_model_list)={len(lens_model_list)}"
        )
    out: List[Dict[str, Any]] = []
    for name, kw in zip(lens_model_list, kwargs_lens):
        base = _DEFAULT_KWARGS_BY_LENS_MODEL.get(name, {})
        out.append({**base, **copy.deepcopy(kw)})
    return out


def _resolve_psf_kwargs(em_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Tie ``psf_kwargs`` to the image grid and return the kwargs to build the PSF with.

    ``psf_kwargs["pixel_size"]`` is the arcsec/pixel of the *kernel array*, and nothing
    resamples it onto the image grid: for GAUSSIAN herculens rebuilds the convolution
    from ``pixel_grid.pixel_width`` and only *exposes* the array, so a ``pixel_size``
    that disagrees with ``pix_scl`` silently hands every consumer of
    ``PSF.kernel_point_source`` (the PAL mirror, ``plot_psf``) a kernel sampled on a
    different grid than the image. The two must therefore be equal.

    A genuinely sub-pixel PSF goes through PIXEL instead: supply the kernel sampled at
    ``pix_scl / p`` with ``kernel_supersampling_factor=p`` (integer), which herculens
    degrades onto the image grid.
    """
    psf_kwargs = dict(em_cfg["psf_kwargs"])
    psf_type = str(psf_kwargs.get("psf_type", "GAUSSIAN"))
    pix_scl = float(em_cfg["pixel_grid_kwargs"]["pix_scl"])

    if psf_type == "GAUSSIAN":
        pixel_size = psf_kwargs.get("pixel_size")
        if pixel_size is None:
            psf_kwargs["pixel_size"] = pix_scl
        elif not math.isclose(float(pixel_size), pix_scl, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(
                f"psf_kwargs['pixel_size']={pixel_size} != pixel_grid_kwargs['pix_scl']"
                f"={pix_scl}. A GAUSSIAN kernel is rendered on its own pixel_size grid and "
                "is never resampled onto the image grid, so the two must match. For a "
                "sub-pixel PSF use psf_type='PIXEL' with the kernel sampled at "
                "pix_scl / p and kernel_supersampling_factor=p (integer)."
            )
        # A leftover PIXEL kernel is ignored by herculens; drop it so the built PSF and
        # the recorded cfg agree.
        psf_kwargs.pop("kernel_point_source", None)
        psf_kwargs.pop("kernel_supersampling_factor", None)

    elif psf_type == "PIXEL":
        # pixel_size is never read for PIXEL -- the kernel carries its own sampling.
        psf_kwargs.pop("pixel_size", None)
        factor = psf_kwargs.get("kernel_supersampling_factor", 1)
        if int(factor) != factor or int(factor) < 1:
            raise ValueError(
                f"psf_kwargs['kernel_supersampling_factor']={factor} must be a positive "
                "integer: herculens degrades the kernel by whole pixels, so a kernel "
                "sampled at pix_scl / p only exists for integer p."
            )
        psf_kwargs["kernel_supersampling_factor"] = int(factor)

    return psf_kwargs


def _check_psf_supersampling(em_cfg: Dict[str, Any]) -> None:
    """Warn when a supersampled PIXEL kernel will not be used as supplied.

    ``kernel_supersampling_factor`` (p) only declares the sampling of the kernel
    array. The fine kernel reaches the convolution only when the numerics also
    supersample with the *same* factor and ``supersampling_convolution=True``;
    otherwise herculens either degrades it to the image grid and discards the
    detail, or (p != n) silently replaces it with one interpolated from the
    degraded kernel. Neither case raises, so warn here.
    """
    # Only PIXEL uses the supplied kernel: herculens builds its own for GAUSSIAN and
    # ignores both the kernel and its factor, so a leftover factor is not a mismatch.
    if str(em_cfg["psf_kwargs"].get("psf_type", "GAUSSIAN")) != "PIXEL":
        return

    kernel_ss = int(em_cfg["psf_kwargs"].get("kernel_supersampling_factor", 1))
    if kernel_ss <= 1:
        return

    numerics = em_cfg["kwargs_numerics"]
    numerics_ss = int(numerics.get("supersampling_factor", 1))
    supersampled_conv = bool(numerics.get("supersampling_convolution", False))

    if numerics_ss == 1 or not supersampled_conv:
        warnings.warn(
            f"psf_kwargs['kernel_supersampling_factor']={kernel_ss} but "
            f"kwargs_numerics has supersampling_factor={numerics_ss}, "
            f"supersampling_convolution={supersampled_conv}. The convolution runs on "
            "the image grid with the degraded kernel, so the supersampled detail is "
            "unused. Set supersampling_factor to "
            f"{kernel_ss} and supersampling_convolution=True to use it.",
            stacklevel=2,
        )
    elif numerics_ss != kernel_ss:
        warnings.warn(
            f"psf_kwargs['kernel_supersampling_factor']={kernel_ss} but "
            f"kwargs_numerics['supersampling_factor']={numerics_ss}. herculens will "
            f"discard the supplied kernel and interpolate a replacement at factor "
            f"{numerics_ss}. Set both to the same value.",
            stacklevel=2,
        )


def recommend_supersampling(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Advise on EM supersampling settings. Pure: never edits cfg, never applies anything.

    Sub-pixel structure is the reason to supersample. Two independent sources of it:

      * an undersampled PSF -- sigma_px = fwhm / 2.3548 / pix_scl below ~1 means the
        kernel is narrower than a pixel, so the convolution itself must move onto the
        subgrid (``supersampling_convolution=True``, which needs the kernel supplied at
        the same factor).
      * a source smaller than a pixel -- R_sersic / pix_scl below ~1 means evaluating
        the profile at pixel centres misrepresents the pixel integral. Supersampling the
        *evaluation* fixes this on its own; the convolution can stay on the image grid.

    Thresholds are heuristics anchored on two measured systems, not a systematic study;
    ``verification`` in the returned dict says how to confirm for a given cfg.

    Returns a dict with the diagnostics, the recommendation, the reason, and a ready
    ``cfg_snippet``. The caller decides whether to use it -- the default everywhere
    stays supersampling_factor=1.
    """
    cfg_full = _deep_merge_dict(make_default_cfg(), cfg or {})
    em_cfg = cfg_full["em"]

    pix_scl = float(em_cfg["pixel_grid_kwargs"]["pix_scl"])
    psf_kwargs = em_cfg["psf_kwargs"]
    psf_type = str(psf_kwargs.get("psf_type", "GAUSSIAN"))

    fwhm = psf_kwargs.get("fwhm")
    psf_sigma_px = None
    if fwhm is not None:
        psf_sigma_px = float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / pix_scl

    source_r_px = None
    kwargs_source = em_cfg.get("kwargs_source") or []
    if kwargs_source and "R_sersic" in kwargs_source[0]:
        source_r_px = float(kwargs_source[0]["R_sersic"]) / pix_scl

    psf_undersampled = psf_sigma_px is not None and psf_sigma_px < 1.0
    source_subpixel = source_r_px is not None and source_r_px < 1.0

    reasons = []
    if psf_undersampled:
        reasons.append(
            f"PSF sigma = {psf_sigma_px:.2f} px (< 1): the kernel is narrower than a "
            "pixel, so the convolution belongs on the subgrid"
        )
    if source_subpixel:
        reasons.append(
            f"source R_sersic = {source_r_px:.2f} px (< 1): pixel-centre evaluation "
            "misrepresents the pixel integral"
        )

    if psf_undersampled:
        factor = 2
        convolution = True
    elif source_subpixel:
        factor = 2
        convolution = False
        reasons.append(
            "PSF is adequately sampled, so supersampled convolution is not needed -- "
            "evaluation-only supersampling is cheaper and keeps the PAL mirror tight"
        )
    else:
        factor = 1
        convolution = False
        reasons.append(
            "PSF and source are both resolved by the pixel grid; supersampling would "
            "cost time without changing the model"
        )

    if factor > 1:
        numerics_snippet = {
            "supersampling_factor": factor,
            "supersampling_convolution": convolution,
        }
    else:
        numerics_snippet = {"supersampling_factor": 1}

    notes = []
    if factor > 1:
        notes.append(
            f"supersampling_factor={factor} is a STARTING POINT, not a converged "
            "answer. Run check_supersampling_convergence() and raise the factor until "
            "the change falls below your tolerance."
        )
    if factor > 1 and convolution:
        if psf_type == "PIXEL":
            notes.append(
                f"Supply the kernel sampled at pix_scl / {factor} with "
                f"kernel_supersampling_factor={factor}; a mismatched factor makes "
                "herculens discard it and interpolate a replacement."
            )
        notes.append(
            "The PAL mirror carries a ~2-3% of peak systematic under supersampled "
            "convolution (PAL convolves at image scale). Use "
            "supersampling_convolution=False for tight PAL cross-checks."
        )
    if factor > 1:
        notes.append(f"Cost scales as roughly {factor}**2 profile evaluations.")

    # Measured: at sigma ~0.21 px the clean model still moves ~8% of peak between
    # factor 3 and 4, so no modest factor converges. The pixel scale, not the
    # numerics, is the limitation there.
    if psf_sigma_px is not None and psf_sigma_px < 0.5:
        notes.append(
            f"PSF sigma = {psf_sigma_px:.2f} px is far below Nyquist: supersampling "
            "converges slowly and the model stays factor-dependent at any affordable "
            "setting. Treat results as pixel-scale limited, or use a finer grid."
        )

    return {
        "pix_scl": pix_scl,
        "psf_type": psf_type,
        "psf_sigma_px": psf_sigma_px,
        "source_R_sersic_px": source_r_px,
        "psf_undersampled": psf_undersampled,
        "source_subpixel": source_subpixel,
        "recommended_supersampling_factor": factor,
        "recommended_supersampling_convolution": convolution,
        "recommended_kernel_supersampling_factor": factor if convolution else 1,
        "reason": "; ".join(reasons),
        "notes": notes,
        "cfg_snippet": {"em": {"kwargs_numerics": numerics_snippet}},
        "verification": (
            "Confirm rather than trust the thresholds: "
            "check_supersampling_convergence(cfg) compares clean models across factors. "
            "Adopt the smallest factor whose difference from the next one is below your "
            "tolerance; if none is, the pixel scale is the limitation, not the numerics."
        ),
    }


def check_supersampling_convergence(
    cfg: Optional[Dict[str, Any]] = None,
    factors: Tuple[int, ...] = (1, 2, 3, 4),
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Measure how the clean EM model changes with supersampling_factor.

    The thresholds in ``recommend_supersampling`` are heuristics; this is the
    measurement. Each factor is compared against the next one up, and the converged
    factor is the smallest whose successor changes the model by less than
    ``tolerance`` (relative to the peak). ``None`` means the sequence never settled
    -- the grid is too coarse for the PSF, and no affordable factor will fix that.

    Costs one EM setup per factor. Never edits the cfg it is given.
    """
    import numpy as np

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg or {})
    psf_kwargs = cfg_full["em"]["psf_kwargs"]
    # Same rule as _check_psf_supersampling: the factor only binds a PIXEL kernel.
    kernel_ss = (
        int(psf_kwargs.get("kernel_supersampling_factor", 1))
        if str(psf_kwargs.get("psf_type", "GAUSSIAN")) == "PIXEL"
        else 1
    )

    models = {}
    for factor in factors:
        cfg_factor = copy.deepcopy(cfg_full)
        cfg_factor["em"]["kwargs_numerics"] = {
            "supersampling_factor": factor,
            "supersampling_convolution": factor > 1,
        }
        # A PIXEL kernel is tied to its own sampling, so a scan over numerics factors
        # would otherwise trip the p != n path and silently swap the kernel.
        if kernel_ss > 1 and factor != kernel_ss:
            raise ValueError(
                "check_supersampling_convergence cannot scan factors against a kernel "
                f"fixed at kernel_supersampling_factor={kernel_ss}. Scan with a "
                "GAUSSIAN psf_kwargs, or rebuild the kernel per factor."
            )
        ctx = setup_em_observation(cfg=cfg_factor)
        models[factor] = np.asarray(
            ctx["lens_image"].model(
                kwargs_lens=ctx["kwargs_lens"],
                kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
                kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
            ),
            float,
        )

    peak = float(max(np.max(np.abs(m)) for m in models.values()))
    steps = {}
    for lower, upper in zip(factors[:-1], factors[1:]):
        steps[lower] = float(np.max(np.abs(models[lower] - models[upper]))) / peak

    converged = None
    for factor in factors[:-1]:
        if steps[factor] < tolerance:
            converged = factor
            break

    return {
        "factors": list(factors),
        "tolerance": tolerance,
        "rel_change_to_next_factor": steps,
        "converged_factor": converged,
        "pixel_scale_limited": converged is None,
    }


def setup_em_observation(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simulate or set up all EM components needed for inference.

    Note:
        This function does not configure JAX. Call `gwemfish.setup_jax(...)`
        explicitly at script/notebook start.
    """

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)

    # Lazy imports after setup_jax to avoid importing jax early.
    from .lens_setup import setup_lens
    from .data_sim import (
        setup_pixel_grid,
        setup_psf,
        setup_noise,
        simulate_em,
    )

    from .config import SOLVER_PARAMS

    if not cfg_full["em"]["enabled"]:
        return {}

    lens_cfg = cfg_full["lens"]
    em_cfg = cfg_full["em"]

    lens_model_list = lens_cfg["lens_model_list"]
    kwargs_lens_in = _kwargs_lens_with_explicit_defaults(
        lens_model_list, lens_cfg["kwargs_lens"]
    )
    cfg_full["lens"]["kwargs_lens"] = kwargs_lens_in
    zl = lens_cfg["zl"]
    zs = lens_cfg["zs"]
    # Herculens EM simulation uses `kwargs_source[0]['center_x/y']` for the source light center.
    # Use the same values to define lens equation "source_pos" for any derived image-position truth.
    src_center_x = em_cfg["kwargs_source"][0].get("center_x", em_cfg["source_pos"][0])
    src_center_y = em_cfg["kwargs_source"][0].get("center_y", em_cfg["source_pos"][1])
    source_pos_em = (src_center_x, src_center_y)
    
    # Mass model + image positions for EM (not GW-specific): ``simulate_em`` needs
    # ``lens_mass_model``; EM lens-equation image positions go into ``truth_params``
    # as ``x_image_true_em`` / ``y_image_true_em`` (distinct from GW ``image_x*``).
    # ``setup_lens`` returns the same ``kwargs_lens`` reference it was given.
    kwargs_lens, x_image_true_em, y_image_true_em, lens_mass_model = setup_lens(
        lens_model_list=lens_model_list,
        kwargs_lens=kwargs_lens_in,
        zl=zl,
        zs=zs,
        source_pos=source_pos_em,
        solver_params=cfg_full["gw"].get("solver_params", SOLVER_PARAMS),
    )

    pixel_grid = setup_pixel_grid(**em_cfg["pixel_grid_kwargs"])
    # Reconcile the kernel grid with the image grid before building the PSF, and record
    # what was actually used in ctx["cfg"].
    em_cfg["psf_kwargs"] = _resolve_psf_kwargs(em_cfg)
    psf = setup_psf(**em_cfg["psf_kwargs"])
    _check_psf_supersampling(em_cfg)
    noise_simu = setup_noise(**em_cfg["noise_simu_kwargs"])
    noise_inf = setup_noise(**em_cfg["noise_inf_kwargs"])

    source_model = em_cfg["source_model_class"]()
    lens_light_model = em_cfg["lens_light_model_class"]()

    # Use MassModelMassSheet for EM simulation when MST is enabled.
    use_mst, k_mst = _resolve_mst_settings(cfg_full)
    if use_mst and k_mst != 0.0:
        from .mass_sheet_model import MassModelMassSheet
        lens_mass_model = MassModelMassSheet(
            lens_model_list, k_mst=float(k_mst)
        )

    em_obs, lens_image = simulate_em(
        kwargs_lens=cfg_full["lens"]["kwargs_lens"],
        kwargs_source=em_cfg["kwargs_source"],
        kwargs_lens_light=em_cfg["kwargs_lens_light"],
        lens_mass_model=lens_mass_model,
        source_model_class=source_model,
        lens_light_model_class=lens_light_model,
        pixel_grid=pixel_grid,
        psf=psf,
        noise_class=noise_simu,
        kwargs_numerics=em_cfg["kwargs_numerics"],
        exposure_time=em_cfg["exposure_time"],
        seed=em_cfg["seed"],
    )

    src = em_cfg["kwargs_source"][0]
    light = em_cfg["kwargs_lens_light"][0]

    # Legacy ``lens_theta_E`` / ``lens_e1`` / … names assume an EPL+shear decomposition and are
    # wrong for e.g. SIS-only. With ``use_parameter_layout``, mass truth is only ``lens0_*`` /
    # … from ``truth_vector_from_kwargs`` (plus optional legacy source/light keys below).
    common_truth = {
        "source_amp": float(src["amp"]),
        "source_R_sersic": float(src["R_sersic"]),
        "source_n": float(src["n_sersic"]),
        "source_e1": float(src["e1"]),
        "source_e2": float(src["e2"]),
        "source_center_x": float(src.get("center_x", em_cfg["source_pos"][0])),
        "source_center_y": float(src.get("center_y", em_cfg["source_pos"][1])),
        "light_amp": float(light["amp"]),
        "light_R_sersic": float(light["R_sersic"]),
        "light_n": float(light["n_sersic"]),
        "light_e1": float(light["e1"]),
        "light_e2": float(light["e2"]),
        "light_center_x": float(light.get("center_x", 0.0)),
        "light_center_y": float(light.get("center_y", 0.0)),
        "noise_sigma_bkg": float(em_cfg["noise_simu_kwargs"]["background_rms"]),
        "x_image_true_em": x_image_true_em,
        "y_image_true_em": y_image_true_em,
    }

    if cfg_full.get("use_parameter_layout"):
        from .parameter_layout import build_parameter_layout, truth_vector_from_kwargs

        truth_params = dict(common_truth)
        ent, _ = build_parameter_layout(
            lens_image,
            kwargs_lens=kwargs_lens,
            kwargs_source=em_cfg["kwargs_source"],
            kwargs_lens_light=em_cfg["kwargs_lens_light"],
        )
        truth_params.update(
            truth_vector_from_kwargs(
                ent,
                kwargs_lens=kwargs_lens,
                kwargs_source=em_cfg["kwargs_source"],
                kwargs_lens_light=em_cfg["kwargs_lens_light"],
                extra={"noise_sigma_bkg": truth_params["noise_sigma_bkg"]},
            )
        )
        if use_mst:
            truth_params["k_mst"] = float(k_mst)
    else:
        epl = _legacy_epl_like_kw(kwargs_lens)
        shear = _legacy_shear_kw(kwargs_lens)
        truth_params = {
            "lens_theta_E": float(epl["theta_E"]),
            "lens_e1": float(epl["e1"]),
            "lens_e2": float(epl["e2"]),
            "lens_gamma": float(epl["gamma"]) if "gamma" in epl else 2.0,
            "lens_gamma1": float(shear["gamma1"]),
            "lens_gamma2": float(shear["gamma2"]),
            "lens_center_x": float(epl["center_x"]),
            "lens_center_y": float(epl["center_y"]),
            **common_truth,
        }
        if use_mst:
            truth_params["k_mst"] = float(k_mst)

    return {
        "cfg": cfg_full,
        "kwargs_lens": kwargs_lens,
        "lens_model_list": lens_model_list,
        "lens_mass_model": lens_mass_model,
        "pixel_grid": pixel_grid,
        "lens_image": lens_image,
        "noise_inf": noise_inf,
        "em_obs": em_obs,
        "truth_params": truth_params,
        "use_mst": use_mst,
        "k_mst": float(k_mst) if use_mst else 0.0,
    }


def setup_gw_observation(ctx: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simulate all GW components needed for inference (gw observables + lens_gw).

    Note:
        This function does not configure JAX. Call `gwemfish.setup_jax(...)`
        explicitly at script/notebook start.
    """

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)

    from .lens_setup import setup_lens
    from .data_sim import simulate_gw, simulate_gw_mst
    from .jaxcosmo import JAXCosmology

    if not cfg_full["gw"]["enabled"]:
        return ctx

    # Ensure lens geometry exists in ctx (for GW-only use cases).
    if "kwargs_lens" not in ctx or "lens_mass_model" not in ctx or "lens_model_list" not in ctx:
        lens_cfg = cfg_full["lens"]
        kwargs_lens_gw = _kwargs_lens_with_explicit_defaults(
            lens_cfg["lens_model_list"], lens_cfg["kwargs_lens"]
        )
        cfg_full["lens"]["kwargs_lens"] = kwargs_lens_gw
        kwargs_lens, _, _, lens_mass_model = setup_lens(
            lens_model_list=lens_cfg["lens_model_list"],
            kwargs_lens=kwargs_lens_gw,
            zl=lens_cfg["zl"],
            zs=lens_cfg["zs"],
            source_pos=cfg_full["gw"]["source_pos"],
            solver_params=cfg_full["gw"].get("solver_params"),
        )
        ctx = dict(ctx)  # shallow copy
        ctx.update(
            {
                "kwargs_lens": kwargs_lens,
                "lens_model_list": lens_cfg["lens_model_list"],
                "lens_mass_model": lens_mass_model,
            }
        )

    lens_cfg = cfg_full["lens"]
    gw_cfg = cfg_full["gw"]

    cosmology = JAXCosmology(**gw_cfg["cosmology"])

    use_mst, k_mst = _resolve_mst_settings(cfg_full, ctx=ctx)
    if use_mst:
        x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw_mst(
            source_pos=gw_cfg["source_pos"],
            kwargs_lens=ctx["kwargs_lens"],
            lens_mass_model=ctx["lens_mass_model"],
            cosmology=cosmology,
            zl=lens_cfg["zl"],
            zs=lens_cfg["zs"],
            k_mst=k_mst,
            lens_model_list=ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
            solver_params=gw_cfg.get("solver_params"),
        )
    else:
        x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(
            source_pos=gw_cfg["source_pos"],
            kwargs_lens=ctx["kwargs_lens"],
            lens_mass_model=ctx["lens_mass_model"],
            cosmology=cosmology,
            zl=lens_cfg["zl"],
            zs=lens_cfg["zs"],
            lens_model_list=ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
            solver_params=gw_cfg.get("solver_params"),
        )

    dL_true = cosmology.luminosity_distance(lens_cfg["zs"])
    T_star_true = data_GW["Tstar_in_seconds"]

    # Update truth params with GW observables used by the prob models.
    truth_params = dict(ctx.get("truth_params", {}))

    # Legacy flat mass keys (EPL+shear); skipped when using parameter layout (mass truth is
    # ``lens0_*`` / … from ``truth_vector_from_kwargs`` below).
    if "kwargs_lens" in ctx and not cfg_full.get("use_parameter_layout"):
        kwargs_lens = ctx["kwargs_lens"]
        epl = _legacy_epl_like_kw(kwargs_lens)
        shear = _legacy_shear_kw(kwargs_lens)
        truth_params.update(
            {
                "lens_theta_E": float(epl["theta_E"]),
                "lens_e1": float(epl["e1"]),
                "lens_e2": float(epl["e2"]),
                "lens_gamma": float(epl["gamma"]) if "gamma" in epl else 2.0,
                "lens_center_x": float(epl["center_x"]),
                "lens_center_y": float(epl["center_y"]),
                "lens_gamma1": float(shear["gamma1"]),
                "lens_gamma2": float(shear["gamma2"]),
            }
        )
    truth_params.update(
        {
            "T_star": float(T_star_true),
            "dL": float(dL_true),
        }
    )

    for i in range(len(x_img_gw)):
        truth_params[f"image_x{i+1}"] = float(x_img_gw[i])
        truth_params[f"image_y{i+1}"] = float(y_img_gw[i])

    if use_mst:
        truth_params["k_mst"] = k_mst

    if cfg_full.get("use_parameter_layout"):
        from .parameter_layout import build_mass_parameter_entries, build_parameter_layout, truth_vector_from_kwargs
        from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

        if ctx.get("lens_image") is not None:
            em = cfg_full.get("em") or {}
            kwargs_source = em.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
            kwargs_lens_light = em.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
            ent, _ = build_parameter_layout(
                ctx["lens_image"],
                kwargs_lens=ctx["kwargs_lens"],
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
            )
            truth_params.update(
                truth_vector_from_kwargs(
                    ent,
                    kwargs_lens=ctx["kwargs_lens"],
                    kwargs_source=kwargs_source,
                    kwargs_lens_light=kwargs_lens_light,
                )
            )
        else:
            ent = build_mass_parameter_entries(
                ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"]
            )
            truth_params.update(
                truth_vector_from_kwargs(
                    ent,
                    kwargs_lens=ctx["kwargs_lens"],
                    kwargs_source=[],
                    kwargs_lens_light=[],
                )
            )

    ctx = dict(ctx)
    ctx.update(
        {
            "cfg": cfg_full,
            "gw_obs": gw_obs,
            "data_GW": data_GW,
            "lens_gw": lens_gw,
            "x_img_gw": x_img_gw,
            "y_img_gw": y_img_gw,
            "truth_params": truth_params,
            "n_images": gw_cfg["n_images"],
            "use_mst": use_mst,
            "k_mst": k_mst if use_mst else 0.0,
        }
    )
    return ctx


def prune_gw_images(ctx: Dict[str, Any], n_keep: int) -> Dict[str, Any]:
    """Keep the first ``n_keep`` GW images and prune dependent fields consistently.

    Useful when a faint extra image is a numerical artifact and should be ignored
    for downstream inference.

    Updates (when present):
      - ``ctx['x_img_gw']``, ``ctx['y_img_gw']``, ``ctx['n_images']``
      - ``ctx['cfg']['gw']['n_images']``
      - ``ctx['data_GW']`` image-wise arrays and time-delay arrays
      - ``ctx['gw_obs']['dL_eff']`` and ``ctx['gw_obs']['time_delays']``
      - ``ctx['truth_params']`` image_x*/image_y* keys
    """
    import jax.numpy as jnp

    if n_keep <= 0:
        raise ValueError(f"`n_keep` must be >= 1, got {n_keep}.")

    if "x_img_gw" not in ctx or "y_img_gw" not in ctx:
        raise ValueError("Context must contain both `x_img_gw` and `y_img_gw`.")

    x_img = jnp.asarray(ctx["x_img_gw"])
    y_img = jnp.asarray(ctx["y_img_gw"])
    if int(x_img.size) != int(y_img.size):
        raise ValueError(
            f"Inconsistent GW images: len(x_img_gw)={int(x_img.size)} "
            f"!= len(y_img_gw)={int(y_img.size)}."
        )
    if n_keep > int(x_img.size):
        raise ValueError(
            f"`n_keep`={n_keep} exceeds available GW images ({int(x_img.size)})."
        )

    out = dict(ctx)
    out["x_img_gw"] = x_img[:n_keep]
    out["y_img_gw"] = y_img[:n_keep]
    out["n_images"] = int(n_keep)

    cfg_cur = out.get("cfg")
    if isinstance(cfg_cur, dict):
        cfg_new = dict(cfg_cur)
        gw_cfg = dict(cfg_new.get("gw", {}))
        gw_cfg["n_images"] = int(n_keep)
        cfg_new["gw"] = gw_cfg
        out["cfg"] = cfg_new

    data_gw = out.get("data_GW")
    if isinstance(data_gw, dict):
        dg = dict(data_gw)
        image_level_keys = (
            "beta_x",
            "beta_y",
            "psi",
            "mu",
            "phi_in_arcsecsq",
            "tarrivals_in_seconds",
            "tarrivals_in_days",
            "tarrivals_days",
        )
        for key in image_level_keys:
            if key in dg:
                dg[key] = jnp.asarray(dg[key])[:n_keep]

        if "tarrivals_in_seconds" in dg:
            dg["time_delays_in_seconds"] = jnp.diff(dg["tarrivals_in_seconds"])
        elif "time_delays_in_seconds" in dg:
            dg["time_delays_in_seconds"] = jnp.asarray(dg["time_delays_in_seconds"])[
                : max(n_keep - 1, 0)
            ]

        if "tarrivals_in_days" in dg:
            dg["time_delays_in_days"] = jnp.diff(dg["tarrivals_in_days"])
        elif "tarrivals_days" in dg:
            dg["time_delays_in_days"] = jnp.diff(dg["tarrivals_days"])
        elif "time_delays_in_days" in dg:
            dg["time_delays_in_days"] = jnp.asarray(dg["time_delays_in_days"])[
                : max(n_keep - 1, 0)
            ]

        out["data_GW"] = dg

    gw_obs = out.get("gw_obs")
    if isinstance(gw_obs, dict):
        go = dict(gw_obs)
        if "dL_eff" in go:
            go["dL_eff"] = jnp.asarray(go["dL_eff"])[:n_keep]
        if "time_delays" in go:
            go["time_delays"] = jnp.asarray(go["time_delays"])[: max(n_keep - 1, 0)]
        out["gw_obs"] = go

    truth_params = dict(out.get("truth_params", {}))
    for key in list(truth_params.keys()):
        if (key.startswith("image_x") or key.startswith("image_y")) and key[7:].isdigit():
            truth_params.pop(key, None)
    for i in range(n_keep):
        truth_params[f"image_x{i+1}"] = float(out["x_img_gw"][i])
        truth_params[f"image_y{i+1}"] = float(out["y_img_gw"][i])
    out["truth_params"] = truth_params

    return out


def _resolve_gw_n_images(ctx: Dict[str, Any], cfg_full: Dict[str, Any]) -> int:
    """Canonical GW image count (same rules as ``run_inference`` / ``setup_gw_observation``)."""
    import jax.numpy as jnp

    truth_params = ctx.get("truth_params", {}) or {}
    n_images_cfg = int(cfg_full["gw"]["n_images"])
    x_img_gw_ctx = ctx.get("x_img_gw")
    n_from_x = int(jnp.asarray(x_img_gw_ctx).size) if x_img_gw_ctx is not None else 0
    n_from_truth = sum(1 for k in truth_params if k.startswith("image_x") and k[7:].isdigit())
    gw_obs_ctx = ctx.get("gw_obs") or {}
    dle = gw_obs_ctx.get("dL_eff")
    n_from_dle = int(jnp.asarray(dle).size) if dle is not None else 0

    if n_from_x > 0:
        n_images = n_from_x
        if n_from_truth > 0 and n_from_truth != n_images:
            raise ValueError(
                f"Inconsistent context: len(ctx['x_img_gw'])={n_images} but truth_params has "
                f"{n_from_truth} image_x* entries. Re-run `setup_gw_observation` or fix ctx."
            )
        if n_from_dle > 0 and n_from_dle != n_images:
            raise ValueError(
                f"Inconsistent context: len(ctx['x_img_gw'])={n_images} but gw_obs['dL_eff'] "
                f"has length {n_from_dle}. Re-run `setup_gw_observation` or fix ctx."
            )
        if n_images_cfg != n_images:
            import warnings

            warnings.warn(
                f"cfg['gw']['n_images']={n_images_cfg} does not match len(ctx['x_img_gw'])="
                f"{n_images}. Using the latter so the prob model matches the simulated images.",
                UserWarning,
                stacklevel=3,
            )
    elif n_from_truth > 0:
        n_images = n_from_truth
    elif n_from_dle > 0:
        n_images = n_from_dle
    else:
        n_images = n_images_cfg
    return n_images


def _build_inference_probmodel(ctx, mode, cfg_full):
    """Build full NumPyro probmodel (same as hmc / deriv-approx path)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .prob_model import ProbModel, ProbModel_GW_only, ProbModel_EM_only

    priors_user = cfg_full.get("priors", {})
    priors_user_norm = _normalize_priors_overrides(
        priors_user, jnp=jnp, numpyro=numpyro, dist=dist,
    )
    truth_params = ctx.get("truth_params", {}) or {}
    n_images = _resolve_gw_n_images(ctx, cfg_full)
    half_width = float(cfg_full["gw"]["image_box_half_width"])
    gw_error_scales = cfg_full["gw"].get("error_scales", {})
    ctx_cfg = ctx.get("cfg", {}) if isinstance(ctx, dict) else {}
    user_set_gw_error_scales = bool(
        (isinstance(ctx_cfg, dict) and isinstance(ctx_cfg.get("gw"), dict)
         and "error_scales" in ctx_cfg.get("gw", {}))
        or (isinstance(cfg_full.get("gw"), dict) and "error_scales" in cfg_full.get("gw", {}))
    )

    priors_internal = {}
    use_mst, _ = _resolve_mst_settings(cfg_full, ctx=ctx)
    if use_mst and mode in ("EM+GW", "GW-only", "EM-only") and "k_mst" not in priors_user_norm:
        priors_internal["k_mst"] = lambda: numpyro.sample(
            "k_mst", dist.Uniform(-0.99999, 0.99999)
        )

    priors_image_pos = {}
    if mode == "GW-only":
        for i in range(n_images):
            xk = f"image_x{i+1}"
            yk = f"image_y{i+1}"
            if xk not in truth_params or yk not in truth_params:
                raise ValueError(f"Missing truth image positions '{xk}'/'{yk}' in ctx['truth_params']")
            xt = float(truth_params[xk])
            yt = float(truth_params[yk])
            priors_image_pos[xk] = (
                lambda name=xk, low=xt - half_width, high=xt + half_width: numpyro.sample(
                    name, dist.Uniform(low, high)
                )
            )
            priors_image_pos[yk] = (
                lambda name=yk, low=yt - half_width, high=yt + half_width: numpyro.sample(
                    name, dist.Uniform(low, high)
                )
            )

    priors_combined = {**priors_internal, **priors_image_pos, **priors_user_norm}
    use_layout = bool(cfg_full.get("use_parameter_layout", False))
    entries = None
    registry = None
    image_position_priors_override = None
    priors_flex = None

    if use_layout:
        from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE
        from .flex_prob_model import FlexProbModelEMGW, FlexProbModelEMOnly, FlexProbModelGWOnly
        from .parameter_layout import (
            build_mass_parameter_entries,
            build_parameter_layout,
            build_priors_registry,
        )

        if mode == "EM+GW":
            em_local = cfg_full.get("em") or {}
            kwargs_source = em_local.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
            kwargs_lens_light = em_local.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
            entries, _ = build_parameter_layout(
                ctx["lens_image"],
                kwargs_lens=ctx["kwargs_lens"],
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
            )
            registry = build_priors_registry(
                entries,
                lens_image=ctx["lens_image"],
                user_priors=priors_user_norm,
            )
            priors_flex = _merge_flex_priors_with_gw_image_pos(
                registry, priors_internal, priors_image_pos, priors_user_norm
            )
            x_true = [truth_params[f"image_x{i+1}"] for i in range(n_images)]
            y_true = [truth_params[f"image_y{i+1}"] for i in range(n_images)]
            image_position_priors_override = {
                "x_image_true": jnp.asarray(x_true),
                "y_image_true": jnp.asarray(y_true),
                "delx": jnp.asarray([2.0 * half_width] * n_images),
                "dely": jnp.asarray([2.0 * half_width] * n_images),
            }
            probmodel = FlexProbModelEMGW(
                entries,
                priors_flex,
                n_images=n_images,
                gw_observations=ctx["gw_obs"],
                em_observations=ctx["em_obs"],
                lens_image=ctx["lens_image"],
                lens_gw=ctx["lens_gw"],
                noise=ctx["noise_inf"],
                image_position_priors=image_position_priors_override,
                gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
                use_mst=use_mst,
            )
        elif mode == "GW-only":
            entries = build_mass_parameter_entries(
                ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"]
            )
            registry = build_priors_registry(
                entries,
                mass_model=ctx["lens_mass_model"],
                user_priors=priors_user_norm,
            )
            priors_flex = _merge_flex_priors_with_gw_image_pos(
                registry, priors_internal, priors_image_pos, priors_user_norm
            )
            probmodel = FlexProbModelGWOnly(
                entries,
                priors_flex,
                n_images=n_images,
                gw_observations=ctx["gw_obs"],
                lens_gw=ctx["lens_gw"],
                gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
                use_mst=use_mst,
            )
        else:
            em_local = cfg_full.get("em") or {}
            kwargs_source = em_local.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
            kwargs_lens_light = em_local.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
            entries, _ = build_parameter_layout(
                ctx["lens_image"],
                kwargs_lens=ctx["kwargs_lens"],
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
            )
            registry = build_priors_registry(
                entries,
                lens_image=ctx["lens_image"],
                user_priors=priors_user_norm,
            )
            priors_flex = {**registry, **priors_internal, **priors_user_norm}
            probmodel = FlexProbModelEMOnly(
                entries,
                priors_flex,
                em_observations=ctx["em_obs"],
                lens_image=ctx["lens_image"],
                noise=ctx["noise_inf"],
                use_mst=use_mst,
            )
    elif mode == "EM+GW":
        x_true = [truth_params[f"image_x{i+1}"] for i in range(n_images)]
        y_true = [truth_params[f"image_y{i+1}"] for i in range(n_images)]
        image_position_priors_override = {
            "x_image_true": jnp.asarray(x_true),
            "y_image_true": jnp.asarray(y_true),
            "delx": jnp.asarray([2.0 * half_width] * n_images),
            "dely": jnp.asarray([2.0 * half_width] * n_images),
        }
        probmodel = ProbModel(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            em_observations=ctx["em_obs"],
            lens_image=ctx["lens_image"],
            lens_gw=ctx["lens_gw"],
            noise=ctx["noise_inf"],
            priors=priors_combined,
            image_position_priors=image_position_priors_override,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
            use_mst=use_mst,
        )
    elif mode == "GW-only":
        probmodel = ProbModel_GW_only(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            lens_gw=ctx["lens_gw"],
            priors=priors_combined,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
            use_mst=use_mst,
        )
    else:
        probmodel = ProbModel_EM_only(
            em_observations=ctx["em_obs"],
            lens_image=ctx["lens_image"],
            noise=ctx["noise_inf"],
            priors=priors_combined,
        )

    return {
        "probmodel": probmodel,
        "n_images": n_images,
        "truth_params": truth_params,
        "half_width": half_width,
        "use_layout": use_layout,
        "entries": entries,
        "registry": registry,
        "likelihood_seed": int(cfg_full["inference"]["rng_key"]),
        "priors_flex": priors_flex,
        "priors_combined": priors_combined,
        "image_position_priors_override": image_position_priors_override,
    }


def _build_inference_probmodel_source_plane(ctx, mode, cfg_full):
    """Build the source-plane NumPyro probmodel (ProbModelSourcePlane /
    ProbModelSourcePlane_GW_only) with a *differentiable* solver, for
    method='deriv-approx-source'. Mirrors _build_inference_probmodel's
    truth-centered-box pattern for image positions, but for y0gw/y1gw.

    Only supports mode in ('EM+GW', 'GW-only') -- source vs image plane is a
    distinction about how GW image positions are sampled, so it doesn't apply
    to 'EM-only' (run_inference raises before calling this for that mode).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .prob_model import ProbModelSourcePlane, ProbModelSourcePlane_GW_only
    from .lens_setup import build_lens_solver

    priors_user = cfg_full.get("priors", {})
    priors_user_norm = _normalize_priors_overrides(
        priors_user, jnp=jnp, numpyro=numpyro, dist=dist,
    )
    truth_params = dict(ctx.get("truth_params", {}) or {})
    n_images = _resolve_gw_n_images(ctx, cfg_full)
    gw_error_scales = cfg_full["gw"].get("error_scales", {})
    user_set_gw_error_scales = "error_scales" in cfg_full.get("gw", {})

    # ctx["truth_params"] never carries 'y0gw'/'y1gw' (unlike image_x*/image_y*,
    # which setup_gw_observation does populate) -- the GW source position only
    # lives in cfg["gw"]["source_pos"]. Backfill it here so downstream code
    # (run_inference's Fisher expansion-point lookup, truths_dict, etc.) can find
    # 'y0gw'/'y1gw' in truth_params like every other sampled parameter.
    if "y0gw" not in truth_params or "y1gw" not in truth_params:
        src_pos = cfg_full["gw"].get("source_pos")
        if src_pos is not None:
            truth_params.setdefault("y0gw", float(src_pos[0]))
            truth_params.setdefault("y1gw", float(src_pos[1]))

    # Half-width of the truth-centered y0gw/y1gw prior box, analogous to
    # cfg["gw"]["image_box_half_width"] on the image-plane side. Override via
    # cfg["gw"]["source_box_half_width"].
    src_half_width = float(cfg_full["gw"].get("source_box_half_width", 0.05))
    y0_truth = float(truth_params.get("y0gw", 0.0))
    y1_truth = float(truth_params.get("y1gw", 0.0))

    priors_source_pos = {
        "y0gw": (lambda low=y0_truth - src_half_width, high=y0_truth + src_half_width:
                 numpyro.sample("y0gw", dist.Uniform(low, high))),
        "y1gw": (lambda low=y1_truth - src_half_width, high=y1_truth + src_half_width:
                 numpyro.sample("y1gw", dist.Uniform(low, high))),
    }
    priors_combined = {**priors_source_pos, **priors_user_norm}

    use_mst, _ = _resolve_mst_settings(cfg_full, ctx=ctx)
    if use_mst and "k_mst" not in priors_user_norm:
        priors_combined["k_mst"] = lambda: numpyro.sample(
            "k_mst", dist.Uniform(-0.99999, 0.99999)
        )

    pixel_grid = ctx.get("pixel_grid")
    if pixel_grid is None:
        from .data_sim import setup_pixel_grid
        pg_kwargs = cfg_full.get("em", {}).get("pixel_grid_kwargs", {})
        pixel_grid = setup_pixel_grid(**pg_kwargs)
    lens_cfg = cfg_full["lens"]
    kwargs_lens_ctx = ctx.get("kwargs_lens") or lens_cfg["kwargs_lens"]
    lens_center = (float(kwargs_lens_ctx[0].get("center_x", 0.0)),
                   float(kwargs_lens_ctx[0].get("center_y", 0.0)))
    solver, solver_params, solver_resolved = build_lens_solver(
        ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
        lens_cfg["zl"], lens_cfg["zs"], ctx["lens_gw"],
        solver_params=cfg_full["gw"].get("solver_params"),
        n_images=n_images, pixel_grid=pixel_grid, polish=True,
        lens_center=lens_center,
    )
    ctx["solver_resolved"] = solver_resolved

    use_layout = bool(cfg_full.get("use_parameter_layout", False))
    entries = None
    registry = None
    priors_flex = None

    if use_layout:
        from .flex_prob_model import FlexProbModelSourcePlaneEMGW, FlexProbModelSourcePlaneGWOnly
        from .parameter_layout import (
            build_mass_parameter_entries,
            build_parameter_layout,
            build_priors_registry,
        )

        if mode == "GW-only":
            entries = build_mass_parameter_entries(
                ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"]
            )
            registry = build_priors_registry(
                entries, mass_model=ctx["lens_mass_model"], user_priors=priors_user_norm,
            )
            priors_flex = {**registry, **priors_combined}
            probmodel = FlexProbModelSourcePlaneGWOnly(
                entries,
                priors_flex,
                n_images=n_images,
                gw_observations=ctx["gw_obs"],
                lens_gw=ctx["lens_gw"],
                solver=solver,
                solver_params=solver_params,
                gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
                use_mst=use_mst,
            )
        else:  # mode == "EM+GW"
            from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

            em_local = cfg_full.get("em") or {}
            kwargs_source = em_local.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
            kwargs_lens_light = em_local.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
            entries, _ = build_parameter_layout(
                ctx["lens_image"],
                kwargs_lens=ctx["kwargs_lens"],
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
            )
            registry = build_priors_registry(
                entries, lens_image=ctx["lens_image"], user_priors=priors_user_norm,
            )
            priors_flex = {**registry, **priors_combined}
            probmodel = FlexProbModelSourcePlaneEMGW(
                entries,
                priors_flex,
                n_images=n_images,
                gw_observations=ctx["gw_obs"],
                em_observations=ctx["em_obs"],
                lens_image=ctx["lens_image"],
                lens_gw=ctx["lens_gw"],
                noise=ctx["noise_inf"],
                solver=solver,
                solver_params=solver_params,
                gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
                use_mst=use_mst,
            )
    elif mode == "GW-only":
        probmodel = ProbModelSourcePlane_GW_only(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            lens_gw=ctx["lens_gw"],
            solver=solver,
            solver_params=solver_params,
            priors=priors_combined,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
            use_mst=use_mst,
        )
    else:  # mode == "EM+GW"
        probmodel = ProbModelSourcePlane(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            em_observations=ctx["em_obs"],
            lens_image=ctx["lens_image"],
            lens_gw=ctx["lens_gw"],
            noise=ctx["noise_inf"],
            solver=solver,
            solver_params=solver_params,
            priors=priors_combined,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
            use_mst=use_mst,
        )

    return {
        "probmodel": probmodel,
        "n_images": n_images,
        "truth_params": truth_params,
        "half_width": src_half_width,
        "use_layout": use_layout,
        "entries": entries,
        "registry": registry,
        "likelihood_seed": int(cfg_full["inference"]["rng_key"]),
        "priors_flex": priors_flex,
        "priors_combined": priors_combined,
        "image_position_priors_override": None,
    }


def _run_nautilus_source_inference(ctx, mode, cfg_full):
    """Dispatch Nautilus source-plane nested sampling."""
    from .nautilus_common import build_em_only_nautilus_problem
    from .nautilus_source_inference import (
        build_em_gw_source_plane_problem,
        build_gw_source_plane_problem,
    )

    if mode == "GW-only":
        prior, loglike, param_names = build_gw_source_plane_problem(ctx, cfg_full)
    elif mode == "EM+GW":
        prior, loglike, param_names = build_em_gw_source_plane_problem(ctx, cfg_full)
    elif mode == "EM-only":
        prior, loglike, param_names = build_em_only_nautilus_problem(ctx, cfg_full)
    else:
        raise ValueError(
            "nautilus-source supports mode 'GW-only', 'EM+GW', and 'EM-only' only"
        )

    return _finish_nautilus_run(ctx, cfg_full, prior, loglike, param_names, "nautilus_source")


def _run_nautilus_image_inference(ctx, mode, cfg_full):
    """Dispatch Nautilus image-plane nested sampling (GW image_x/y sampling)."""
    from .nautilus_image_inference import build_image_plane_problem

    prior, loglike, param_names = build_image_plane_problem(ctx, mode, cfg_full)
    return _finish_nautilus_run(ctx, cfg_full, prior, loglike, param_names, "nautilus_image")


def _finish_nautilus_run(ctx, cfg_full, prior, loglike, param_names, method_tag):
    from .nautilus_common import run_nautilus

    _skip = {"solver_backend", "solver_validation_tol"}
    n_cfg = cfg_full.get("nautilus", {})
    run_kwarg_keys = {"n_eff", "n_like_max", "discard_exploration", "timeout"}
    run_kwargs = {k: v for k, v in n_cfg.items() if k in run_kwarg_keys}
    nautilus_top = {k: v for k, v in n_cfg.items()
                    if k not in _skip and k not in run_kwarg_keys}
    samples = run_nautilus(prior, loglike, run_kwargs=run_kwargs, **nautilus_top)

    truth_params = ctx.get("truth_params", {}) or {}
    truths_dict = {k: float(truth_params[k]) for k in param_names if k in truth_params}

    out_cfg = cfg_full.get("output", {})
    out_dir = out_cfg.get("output_dir")
    save_samples_path = _resolve_output_path(out_cfg.get("save_samples_path"), out_dir)
    save_truths_path = _resolve_output_path(out_cfg.get("save_truths_path"), out_dir)
    save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
    save_samples_path = _append_tag_to_path(save_samples_path, method_tag)
    save_truths_path = _append_tag_to_path(save_truths_path, method_tag)
    save_json_path = _append_tag_to_path(save_json_path, method_tag)
    _ensure_parent_dir(save_samples_path)
    _ensure_parent_dir(save_truths_path)
    _save_dict_npz(save_samples_path, samples)
    _save_dict_npz(save_truths_path, truths_dict)
    _save_pipeline_json(save_json_path, ctx=ctx,
                        samples_image_plane=samples,
                        truths_image_plane=truths_dict)
    return samples, truths_dict


def run_inference(
    ctx: Dict[str, Any],
    *,
    mode: str = "EM+GW",
    method: str = "deriv-approx",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run inference and return (samples_dict, truths_dict).

    **``deriv-approx``** — MCMC on the Fisher / banana model (``ProbModelFisher*``). Default: plain NUTS.
    Set ``cfg['inference']['informed']=True`` for Hessian-informed NUTS on that approximate model.

    **``hmc``** — Plain NUTS on the full ``ProbModel`` / ``ProbModel_GW_only`` / ``ProbModel_EM_only``.
    For Hessian-informed NUTS on the same full model, use ``method='hmc-informed'`` (or ``hmc`` with
    ``cfg['inference']['informed']=True``).

    **``hmc-informed``** — Same full model as ``hmc``, but always ``run_mcmc_informed`` (Fisher mass
    matrix). Plain NUTS on the full model is ``method='hmc'`` only; ``informed=False`` is not valid here.

    **``hmc-source`` / ``hmc-informed-source``** — Source-plane counterparts of ``hmc`` /
    ``hmc-informed``: full likelihood on ``ProbModelSourcePlane`` / ``ProbModelSourcePlane_GW_only``
    (samples ``y0gw``/``y1gw`` directly, solves the lens equation inside the model via the
    differentiable solver), instead of image positions. Same informed/plain semantics as their
    image-plane counterparts. Not defined for ``mode='EM-only'`` (mirrors ``deriv-approx-source``).

    **``nautilus-source``** — Nested sampling with source-plane GW (``y0gw``/``y1gw`` + lens solver).
    **``nautilus-image``** — Nested sampling with image-plane GW (``image_x*``/``image_y*``); EM-only
    uses the same pixel likelihood as ``nautilus-source``. Sampler settings: ``cfg['nautilus']``
    (incl. ``prior_check``, default True: checkpointed runs write a prior-fingerprint sidecar
    ``<filepath>.priors.json`` and refuse ``resume=True`` if the current priors differ from the
    ones the checkpoint was built with).

    **``fisher-source``** — Source-plane counterpart of ``fisher``: same Fisher-Gaussian-sample
    early return (``N(u0, inv(-H0))``, no NUTS), but ``H0``/``u0``/``keys_to_include`` come from
    the source-plane probmodel (``y0gw``/``y1gw``) instead of image positions. Not defined for
    ``mode='EM-only'`` (mirrors ``deriv-approx-source``).

    ``compute_fisher`` always uses the full model to build ``H0`` at the expansion point.

    **Mass sheet (MST)** — Prefer top-level ``cfg['mst'] = {'enabled': bool, 'k_mst': float}``.
    Legacy ``cfg['gw']['use_mst']`` / ``cfg['gw']['k_mst']`` are still accepted as fallbacks.
    For inference, ``run_inference`` adds a default ``k_mst`` uniform prior unless
    ``cfg['priors']['k_mst']`` is set. GW forward uses ``LensImageGW.compute_mst`` so ``k_mst``
    is JAX-traceable.
    """
    if mode not in ("EM+GW", "GW-only", "EM-only"):
        raise ValueError("mode must be one of: 'EM+GW', 'GW-only', 'EM-only'")
    method_norm = method.strip().lower()
    if method_norm == "nautilus":
        raise ValueError(
            "method='nautilus' was removed; use 'nautilus-source' (source-plane GW) "
            "or 'nautilus-image' (image-plane GW sampling)."
        )
    if method_norm not in (
        "deriv-approx", "deriv-approx-source", "fisher", "fisher-source", "hmc", "hmc-informed",
        "hmc-source", "hmc-informed-source",
        "nautilus-source", "nautilus-image",
    ):
        raise ValueError(
            "method must be one of: 'deriv-approx', 'deriv-approx-source', 'fisher', "
            "'fisher-source', 'HMC', 'HMC-informed', 'hmc-source', 'hmc-informed-source', "
            "'nautilus-source', 'nautilus-image'"
        )
    source_plane_methods = ("deriv-approx-source", "hmc-source", "hmc-informed-source", "fisher-source")
    if method_norm in source_plane_methods and mode == "EM-only":
        raise ValueError(
            f"method={method_norm!r} is not defined for mode='EM-only' "
            "(no GW image positions to parametrize in source vs image plane)."
        )

    # Lazy imports inside this function.
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .prob_model import (
        ProbModel,
        ProbModelFisher,
        ProbModelFisher_GW_only,
        ProbModel_GW_only,
        ProbModel_EM_only,
    )
    from .inference import run_mcmc, run_mcmc_informed
    from .fisher import (
        compute_fisher,
        default_param_floors,
        format_map_params,
        newton_raphson_maxp,
    )
    from .diagnostics import diagnose_system

    # Prefer the simulation/setup configuration already attached to `ctx`.
    # This keeps EM/GW setup choices and derived truth generation consistent
    # with inference-time settings (image boxes, priors normalization, etc.).
    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)

    if method_norm == "nautilus-source":
        return _run_nautilus_source_inference(ctx, mode, cfg_full)
    if method_norm == "nautilus-image":
        return _run_nautilus_image_inference(ctx, mode, cfg_full)

    setup_seed = cfg_full["inference"]["rng_key"]
    fisher_seed = int(cfg_full["inference"]["rng_key"])
    likelihood_seed = int(cfg_full["inference"]["rng_key"])

    priors_user = cfg_full.get("priors", {})
    if method_norm in source_plane_methods:
        built = _build_inference_probmodel_source_plane(ctx, mode, cfg_full)
    else:
        built = _build_inference_probmodel(ctx, mode, cfg_full)
    probmodel = built["probmodel"]
    likelihood_seed = built["likelihood_seed"]
    truth_params = built["truth_params"]
    priors_flex = built["priors_flex"]
    priors_combined = built["priors_combined"]
    image_position_priors_override = built["image_position_priors_override"]

    # Keys to include and expansion point. Fixed literals in ``cfg['priors']`` are held at those
    # values in ``input_params`` but omitted from ``keys_to_include`` so Fisher / deriv-approx does
    # not differentiate w.r.t. them (matches EM+GW behavior where user priors override image boxes).
    fixed_literal_keys = _fixed_literal_prior_keys(priors_user)
    prior_sample = probmodel.get_sample(prng_key=jax.random.PRNGKey(int(cfg_full["inference"]["prior_sample_rng_key"])))
    keys_all = list(prior_sample.keys())
    keys_to_include = [k for k in keys_all if k not in fixed_literal_keys]

    input_params: Dict[str, Any] = {}
    missing = []
    for k in keys_all:
        if k in fixed_literal_keys:
            input_params[k] = _float_from_user_prior_literal(priors_user[k])
        elif k in truth_params:
            input_params[k] = truth_params[k]
        else:
            missing.append(k)
    if missing:
        raise ValueError(
            "Cannot build Fisher expansion point `input_params` for keys: "
            + ", ".join(missing)
            + ". Provide these via `cfg['priors']` (and re-run setup), or ensure simulation truth matches."
        )
    def check_contributions(model, rng_key=None):
        def get_likelihood_contributions(params, _rng_key=rng_key):
            """Extract likelihood and prior contributions from all sites in the model trace."""
            key = jax.random.PRNGKey(0) if _rng_key is None else _rng_key
            
            # Get trace with parameters substituted
            seeded_model = seed(model, key)
            substituted_model = substitute(seeded_model, data=params)
            tr = trace(substituted_model).get_trace()
            
            log_prior = 0.0
            log_likelihood = 0.0
            prior_components = {}
            likelihood_components = {}
            
            print("=" * 80)
            print("LIKELIHOOD CONTRIBUTIONS FROM ALL SITES")
            print("=" * 80)
            print(f"{'Site Name':<30} {'Type':<15} {'Log Prob':<20} {'Value'}")
            print("-" * 80)
            
            for site_name, site in tr.items():
                if site['type'] == 'sample':
                    value = site['value']
                    log_prob = jnp.sum(site['fn'].log_prob(value))
                    
                    if site.get('is_observed', False):
                        log_likelihood += log_prob
                        likelihood_components[site_name] = float(log_prob)
                        site_type = "LIKELIHOOD"
                    else:
                        log_prior += log_prob
                        prior_components[site_name] = float(log_prob)
                        site_type = "PRIOR"
                    
                    # Format value for display
                    if hasattr(value, 'shape') and value.size > 1:
                        value_str = f"shape={value.shape}"
                    else:
                        value_str = str(float(value)) if jnp.isscalar(value) else str(value)
                    
                    print(f"{site_name:<30} {site_type:<15} {float(log_prob):<20.6f} {value_str}")
            
            print("-" * 80)
            print(f"{'TOTAL PRIOR':<30} {'':<15} {float(log_prior):<20.6f}")
            print(f"{'TOTAL LIKELIHOOD':<30} {'':<15} {float(log_likelihood):<20.6f}")
            print(f"{'TOTAL POSTERIOR':<30} {'':<15} {float(log_prior + log_likelihood):<20.6f}")
            print("=" * 80)
            
            return {
                'log_prior': float(log_prior),
                'log_likelihood': float(log_likelihood),
                'log_posterior': float(log_prior + log_likelihood),
                'prior_components': prior_components,
                'likelihood_components': likelihood_components,
                'trace': tr
            }
        return get_likelihood_contributions


     ### This will return the likelihood function which user can use for checks
    # Seed the model
    def give_user_likelihood_function(probmodel, input_params, keys_to_include, likelihood_seed):
        seeded_model = seed(probmodel.model, jax.random.PRNGKey(likelihood_seed))
        # Create logdensity function
        def logdensity_fn(args):
            # print("args:", args)
            log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
            return log_density
        
        # Create vectorized logdensity function
        def logdensity_fn_vec(u):
            input_ = input_params.copy()
            for i, key in enumerate(keys_to_include):
                input_[key] = u[i]
            return logdensity_fn(input_)
        
        return logdensity_fn_vec, logdensity_fn
    
    logdensity_fn_vec, logdensity_fn = give_user_likelihood_function(probmodel, input_params, keys_to_include, likelihood_seed)
    contributions = check_contributions(probmodel.model, jax.random.PRNGKey(likelihood_seed))
    ctx["likelihood"] = {}
    ctx["likelihood"]["probmodel"] = probmodel
    ctx["likelihood"]["likelihood_function_vec"] = logdensity_fn_vec
    ctx["likelihood"]["likelihood_function"] = logdensity_fn
    ctx["likelihood"]["input_params"] = input_params
    ctx["likelihood"]["keys_to_include"] = keys_to_include
    ctx["likelihood"]["check_contributions"] = contributions

    u0_given = jnp.asarray([input_params[k] for k in keys_to_include], dtype=jnp.float64)
    ctx["likelihood"]["u0_given"] = u0_given
    # Truths / fixed labels stay at the given values; Fisher expands at u0 (possibly MAP).
    truths_dict = {k: float(input_params[k]) for k in keys_all}

    # Newton–Raphson maxP polish: jump until |g|<grad_tol (or max_jumps), then
    # expand Fisher / sample Gaussian at the landed MAP.
    newton_cfg = dict(cfg_full.get("inference", {}).get("newton_maxp") or {})
    newton_enabled = bool(newton_cfg.get("enabled", True))
    u0 = u0_given
    input_params_fisher = input_params
    if newton_enabled:
        _mj = newton_cfg.get("max_jumps", None)
        max_jumps = None if _mj is None else int(_mj)
        floors = None
        if bool(newton_cfg.get("apply_default_floors", True)):
            floors = default_param_floors(
                keys_to_include,
                u0_given,
                rel=float(newton_cfg.get("floor_rel", 1e-3)),
                abs_floor=float(newton_cfg.get("floor_abs", 1e-12)),
            )
        u0, newton_info = newton_raphson_maxp(
            logdensity_fn_vec,
            u0_given,
            max_jumps=max_jumps,
            grad_tol=float(newton_cfg.get("grad_tol", 5e-2)),
            verbose=bool(newton_cfg.get("verbose", True)),
            line_search=bool(newton_cfg.get("line_search", True)),
            floors=floors,
        )
        input_params_fisher = dict(input_params)
        map_params = {}
        for i, key in enumerate(keys_to_include):
            map_params[key] = float(u0[i])
            input_params_fisher[key] = u0[i]
        ctx["likelihood"]["newton_maxp"] = {
            "n_jumps": newton_info["n_jumps"],
            "grad_norm": newton_info["grad_norm"],
            "grad_norm_scaled": newton_info["grad_norm_scaled"],
            "converged": newton_info["converged"],
            "logp": newton_info["logp"],
            "u_map": newton_info["u_map"],
            "u0_given": newton_info["u0_given"],
            "map_params": map_params,
        }
        print(
            format_map_params(
                keys_to_include,
                newton_info["u_map"],
                newton_info["u0_given"],
                grad_norm=newton_info["grad_norm_scaled"],
                n_jumps=newton_info["n_jumps"],
                logp=newton_info["logp"],
            )
        )
    ctx["likelihood"]["u0"] = u0
    # --- DEBUG ---
    print("use_mst in probmodel:", probmodel.use_mst)
    print("k_mst in keys_to_include:", "k_mst" in keys_to_include)
    print("k_mst truth:", ctx["truth_params"].get("k_mst"))
    print("keys_to_include:", keys_to_include)
    print("input_params (given):", {k: input_params[k] for k in keys_to_include})
    if newton_enabled:
        print("expansion u0 (after newton_maxp):",
              {k: float(u0[i]) for i, k in enumerate(keys_to_include)})
# --- END DEBUG ---
    approx_logp, logp0, g0, H0, F0, Q0 = compute_fisher(
        model=probmodel.model,
        input_params=input_params_fisher,
        keys_to_include=keys_to_include,
        u0=u0,
        rng_key=jax.random.PRNGKey(fisher_seed),
        order=int(cfg_full["inference"]["fisher_order"]),
    )
    ctx['fisher'] = {}
    ctx['fisher']['approx_logp'] = approx_logp
    ctx['fisher']['logp0'] = logp0
    ctx['fisher']['g0'] = g0
    ctx['fisher']['H0'] = H0
    ctx['fisher']['F0'] = F0
    ctx['fisher']['Q0'] = Q0
    # Diagnostic: gradient and Hessian diagonal at u0 (Fisher expansion point).
    for i, key in enumerate(keys_to_include):
        gi = g0[i]
        if jnp.isnan(gi):
            print(f"Gradient w.r.t {key} is nan")
        elif jnp.isinf(gi):
            print(f"Gradient w.r.t {key} is inf")
        elif gi == 0:
            print(f"Gradient w.r.t {key} is zero")
    h_diag = jnp.diag(H0)
    for i, key in enumerate(keys_to_include):
        hi = h_diag[i]
        if jnp.isnan(hi):
            print(f"Hessian diagonal H[{key},{key}] is nan")
        elif jnp.isinf(hi):
            print(f"Hessian diagonal H[{key},{key}] is inf")
        elif hi == 0:
            print(f"Hessian diagonal H[{key},{key}] is zero")

    # Scaled gradient: g0 / sqrt(|H_ii|), i.e. how many 1-sigma steps each parameter
    # sits from the peak. The raw gradient is not comparable across parameters
    # because u0 is unscaled (y1gw ~ 1e-6 alongside theta_E ~ 2).
    g0_scaled = g0 / jnp.sqrt(jnp.abs(h_diag))
    ctx['fisher']['g0_scaled'] = g0_scaled
    print("Scaled gradient at expansion point (g0 / sqrt|diag H0|):")
    for key, gs in zip(keys_to_include, g0_scaled):
        print(f"  {key:24s} {float(gs): .4e}")

    diag_level = cfg_full["inference"].get("diagnostics", "warn")
    diagnostics_report = diagnose_system(
        ctx, cfg_full, method_norm, mode,
        solver=getattr(probmodel, "solver", None),
        solver_params=getattr(probmodel, "solver_params", None),
        g0=g0, H0=H0, keys=keys_to_include, level=diag_level,
    )
    ctx["diagnostics"] = diagnostics_report

    # Fisher-only: sample from Gaussian N(u0, cov)
    priors_for_fisher = priors_flex if priors_flex is not None else priors_combined

    if method_norm in ("fisher", "fisher-source"):
        FM = -H0
        cov = _fisher_covariance(FM, keys_to_include)

        key = jax.random.PRNGKey(int(setup_seed))
        n_fisher_samples = int(cfg_full["inference"]["n_fisher_samples"])
        samples_cov_array = jax.random.multivariate_normal(key, u0, cov, shape=(n_fisher_samples,))
        samples = {keys_to_include[i]: samples_cov_array[:, i] for i in range(len(keys_to_include))}
        out_cfg = cfg_full.get("output", {})
        out_dir = out_cfg.get("output_dir")
        save_samples_path = _resolve_output_path(out_cfg.get("save_samples_path"), out_dir)
        save_truths_path = _resolve_output_path(out_cfg.get("save_truths_path"), out_dir)
        save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
        save_samples_path = _append_tag_to_path(save_samples_path, method_norm)
        save_truths_path = _append_tag_to_path(save_truths_path, method_norm)
        save_json_path = _append_tag_to_path(save_json_path, method_norm)
        _ensure_parent_dir(save_samples_path)
        _ensure_parent_dir(save_truths_path)
        _save_dict_npz(save_samples_path, samples)
        _save_dict_npz(save_truths_path, truths_dict)
        _save_pipeline_json(
            save_json_path,
            ctx=ctx,
            samples_image_plane=samples,
            truths_image_plane=truths_dict,
        )
        return samples, truths_dict

    # Banana / Taylor model (``ProbModelFisher*``): used only for ``deriv-approx``.
    if mode == "GW-only":
        fisher_prob_model = ProbModelFisher_GW_only(
            keys_to_include=keys_to_include,
            approx_logp=approx_logp,
            priors=priors_for_fisher,
        )
    else:
        fisher_prob_model = ProbModelFisher(
            keys_to_include=keys_to_include,
            approx_logp=approx_logp,
            priors=priors_for_fisher,
            image_position_priors=image_position_priors_override,
        )

    H0_for_mcmc = H0
    h0_override = cfg_full["inference"].get("H0")
    if h0_override is not None:
        H0_for_mcmc = jnp.asarray(h0_override, dtype=jnp.float64)

    if method_norm in ("deriv-approx", "deriv-approx-source"):
        mcmc_model = fisher_prob_model.model
    else:
        # ``hmc`` / ``hmc-informed``: full likelihood (not the banana model).
        mcmc_model = probmodel.model

    inf_cfg = cfg_full["inference"]
    informed_override = inf_cfg.get("informed")

    if method_norm in ("hmc-informed", "hmc-informed-source"):
        if informed_override is False:
            raise ValueError(
                f"method={method_norm!r} always uses Hessian-informed NUTS. "
                "Use method='hmc'/'hmc-source' for plain NUTS on the full model."
            )
        use_informed_mcmc = True
    elif method_norm in ("hmc", "hmc-source"):
        use_informed_mcmc = informed_override is True
    elif method_norm in ("deriv-approx", "deriv-approx-source"):
        use_informed_mcmc = informed_override is True
    else:
        use_informed_mcmc = False

    if use_informed_mcmc:
        print("Running informed HMC")
        samples, summary, extra_fields, mcmc = run_mcmc_informed(
            mcmc_model,
            u0=u0,
            keys_to_include=keys_to_include,
            H0=H0_for_mcmc,
            num_warmup=int(cfg_full["inference"]["num_warmup"]),
            num_samples=int(cfg_full["inference"]["num_samples"]),
            max_tree_depth=int(cfg_full["inference"]["max_tree_depth"]),
            num_chains=int(cfg_full["inference"]["num_chains"]),
            scale=float(cfg_full["inference"].get("hmc_informed_scale", 1.0)),
            perturb_scale=float(cfg_full["inference"].get("hmc_informed_perturb_scale", 0.1)),
            regularize=bool(cfg_full["inference"].get("regularize", False)),
            rng_key=jax.random.PRNGKey(int(setup_seed)),
        )
    else:
        print("Running uninformed HMC")
        samples, summary, extra_fields, mcmc = run_mcmc(
            mcmc_model,
            num_warmup=int(cfg_full["inference"]["num_warmup"]),
            num_samples=int(cfg_full["inference"]["num_samples"]),
            max_tree_depth=int(cfg_full["inference"]["max_tree_depth"]),
            dense_mass=bool(cfg_full["inference"]["dense_mass"]),
            num_chains=int(cfg_full["inference"]["num_chains"]),
            rng_key=jax.random.PRNGKey(int(setup_seed)),
        )
    _ = (summary, extra_fields, mcmc)
    out_cfg = cfg_full.get("output", {})
    out_dir = out_cfg.get("output_dir")
    save_samples_path = _resolve_output_path(out_cfg.get("save_samples_path"), out_dir)
    save_truths_path = _resolve_output_path(out_cfg.get("save_truths_path"), out_dir)
    save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
    save_samples_path = _append_tag_to_path(save_samples_path, method_norm)
    save_truths_path = _append_tag_to_path(save_truths_path, method_norm)
    save_json_path = _append_tag_to_path(save_json_path, method_norm)
    _ensure_parent_dir(save_samples_path)
    _ensure_parent_dir(save_truths_path)
    _save_dict_npz(save_samples_path, samples)
    _save_dict_npz(save_truths_path, truths_dict)
    _save_pipeline_json(
        save_json_path,
        ctx=ctx,
        samples_image_plane=samples,
        truths_image_plane=truths_dict,
    )
    return samples, truths_dict


def plot_posterior(
    samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot posterior using the grouped corner utilities."""
    return _plot_samples_common(samples=samples, truths=truths, cfg=cfg)


def _plot_samples_common(
    *,
    samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Internal common plotting backend used by image/source-plane helpers."""
    import numpy as np

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)
    plot_cfg = cfg_full["plot"]

    from .corner_plot_utils import (
        create_default_param_groups,
        plot_grouped_corner,
        plot_custom_params,
    )

    plot_mode = plot_cfg.get("plot_mode", "groupwise")
    color = plot_cfg.get("color", "#2c3e50")
    truth_color = plot_cfg.get("truth_color", "red")

    out_cfg = cfg_full.get("output", {})
    save_path = _resolve_output_path(plot_cfg.get("save_path"), out_cfg.get("output_dir"))
    save_path = _append_tag_to_path(save_path, plot_cfg.get("save_tag"))
    _ensure_parent_dir(save_path)

    if truths is None:
        truths = {}

    if plot_mode == "groupwise":
        param_groups = create_default_param_groups(samples)
        truths_dict = {
            group: {p: float(truths[p]) for p in params if p in truths}
            for group, params in param_groups.items()
        }
        grouped_kw: Dict[str, Any] = {}
        if plot_cfg.get("hist_kwargs") is not None:
            grouped_kw["hist_kwargs"] = plot_cfg["hist_kwargs"]
        return plot_grouped_corner(
            samples_dict=samples,
            param_groups=param_groups,
            truths_dict=truths_dict,
            color=color,
            truth_color=truth_color,
            show_titles=bool(plot_cfg.get("show_titles", True)),
            title_kwargs=plot_cfg.get("title_kwargs", {"fontsize": 10}),
            title_fmt=plot_cfg.get("title_fmt", ".3f"),
            quantiles=plot_cfg.get("quantiles", [0.05, 0.5, 0.975]),
            save_path=save_path,
            **grouped_kw,
        )

    if plot_mode in ("combined", "subset"):
        params_to_plot = plot_cfg.get("params_to_plot")
        if params_to_plot is None:
            params_to_plot = list(samples.keys())
        custom_kw: Dict[str, Any] = {}
        if plot_cfg.get("hist_kwargs") is not None:
            custom_kw["hist_kwargs"] = plot_cfg["hist_kwargs"]
        return plot_custom_params(
            samples=samples,
            params_to_plot=params_to_plot,
            truths=truths,
            color=color,
            truth_color=truth_color,
            show_titles=bool(plot_cfg.get("show_titles", True)),
            title_kwargs=plot_cfg.get("title_kwargs", {"fontsize": 10}),
            title_fmt=plot_cfg.get("title_fmt", ".3f"),
            quantiles=plot_cfg.get("quantiles", [0.05, 0.5, 0.975]),
            save_path=save_path,
            **custom_kw,
        )

    raise ValueError("plot_mode must be one of: 'groupwise', 'combined', 'subset'")


def to_source_plane_samples(
    samples: Dict[str, Any],
    ctx: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    build_kwargs_lens_vectorised_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Convert image-plane posterior samples to source-plane posteriors.

    This wraps `image_samples_to_source_samples` and returns a dict containing
    `y0gw`, `y1gw`, and a `source_plane_samples` dictionary ready for plotting.

    The number of lensed images is taken from the GW context (``len(ctx['x_img_gw'])`` when
    present, else truth / ``gw_obs['dL_eff']`` / ``cfg['gw']['n_images']``) — the same rule
    as ``run_inference``, not from ``cfg['source_plane']['n_images']``.

    Missing ``image_x*`` / ``image_y*`` entries (e.g. fixed in ``cfg['priors']`` so they were
    not MCMC-sampled) are broadcast to all samples from fixed ``cfg['priors']``, ``truth_params``,
    or ``ctx['x_img_gw']`` / ``ctx['y_img_gw']``, like lens parameters. Those filled keys are
    stripped from the returned ``source_plane_samples`` dicts so they do not look like posterior
    draws.

    Config options (under `cfg['source_plane']`, all optional):
      - n_subsample: int or None
      - seed: int
      - filter_std: float or None
      - use_filtered: bool

    Optional `cfg['output']` keys:
      - json_path: str — pipeline JSON path (relative to ``output_dir`` unless absolute).
        Legacy alias: ``save_pipeline_json_path``.
      - json_tag: str — appended to ``json_path`` (same rules as ``plot['save_tag']`` for corner plots),
        e.g. ``hmc-informed`` → ``..._hmc_informed.json``. Legacy alias: ``save_pipeline_json_tag``.

    Args:
        build_kwargs_lens_vectorised_fn:
            Optional custom lens-kwargs builder passed through to
            `image_samples_to_source_samples(...)`. Keep as None for default
            EPL+SHEAR handling.
    """
    # Same merge order as ``run_inference``: start from ``ctx['cfg']`` (setup
    # choices like ``gw['n_images']``), then apply the call-time ``cfg``.
    # Using only ``make_default_cfg()`` here would drop e.g. ``n_images: 2``
    # when the user passes only ``cfg={'output': ...}``.
    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    sp_cfg = cfg_full.get("source_plane", {})
    priors_cfg = cfg_full.get("priors", {})

    from .image_to_source import image_samples_to_source_samples

    if "lens_gw" not in ctx:
        raise ValueError(
            "Missing `lens_gw` in context. Run `setup_gw_observation(...)` before "
            "calling `to_source_plane_samples(...)`."
        )

    n_images = _resolve_gw_n_images(ctx, cfg_full)

    use_layout = bool(cfg_full.get("use_parameter_layout", False))
    build_fn = build_kwargs_lens_vectorised_fn

    if use_layout and ctx.get("lens_image") is not None:
        from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE
        from .parameter_layout import build_parameter_layout, build_vectorised_lens_kwargs_fn, flat_keys

        em_local = cfg_full.get("em") or {}
        kwargs_source = em_local.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
        kwargs_lens_light = em_local.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
        entries, _ = build_parameter_layout(
            ctx["lens_image"],
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        required_lens_keys = [fk for fk in flat_keys(entries) if fk.startswith("lens")]
        if build_fn is None:
            build_fn = build_vectorised_lens_kwargs_fn(
                entries, len(ctx["lens_mass_model"].func_list)
            )
    elif use_layout:
        from .parameter_layout import (
            build_mass_parameter_entries,
            build_vectorised_lens_kwargs_fn,
            flat_keys,
        )

        entries = build_mass_parameter_entries(
            ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"]
        )
        required_lens_keys = list(flat_keys(entries))
        if build_fn is None:
            build_fn = build_vectorised_lens_kwargs_fn(
                entries, len(ctx["lens_mass_model"].func_list)
            )
    else:
        required_lens_keys = [
            "lens_theta_E",
            "lens_e1",
            "lens_e2",
            "lens_gamma",
            "lens_gamma1",
            "lens_gamma2",
            "lens_center_x",
            "lens_center_y",
        ]

    if not samples:
        raise ValueError("Empty samples dict passed to `to_source_plane_samples(...)`.")
    first_key = next(iter(samples))
    n_samps = len(samples[first_key])
    sample_dict_for_source = dict(samples)
    truth_params = ctx.get("truth_params", {})

    import jax.numpy as jnp
    import numpyro.distributions as dist

    for k in required_lens_keys:
        if k in sample_dict_for_source:
            continue

        value = None
        if k in priors_cfg:
            pv = priors_cfg[k]
            if not callable(pv) and not isinstance(pv, dist.Distribution):
                value = pv
        if value is None and k in truth_params:
            value = truth_params[k]

        if value is None:
            raise ValueError(
                f"Missing required key '{k}' for source-plane conversion. "
                "Provide it in sampled parameters, fixed cfg['priors'], or truth_params."
            )
        sample_dict_for_source[k] = jnp.full((n_samps,), float(value))

    x_img_ctx = ctx.get("x_img_gw")
    y_img_ctx = ctx.get("y_img_gw")
    for i in range(1, n_images + 1):
        xk, yk = f"image_x{i}", f"image_y{i}"
        for k in (xk, yk):
            if k in sample_dict_for_source:
                continue
            value = None
            if k in priors_cfg:
                pv = priors_cfg[k]
                if not callable(pv) and not isinstance(pv, dist.Distribution):
                    value = pv
            if value is None and k in truth_params:
                value = truth_params[k]
            if value is None and x_img_ctx is not None and k == xk and i - 1 < len(x_img_ctx):
                value = float(x_img_ctx[i - 1])
            if value is None and y_img_ctx is not None and k == yk and i - 1 < len(y_img_ctx):
                value = float(y_img_ctx[i - 1])
            if value is None:
                raise ValueError(
                    f"Missing '{k}' for source-plane ray shooting (image {i} of {n_images}). "
                    "Include it in MCMC samples, set a fixed value in cfg['priors'], put it in "
                    "truth_params, or ensure ctx['x_img_gw']/['y_img_gw'] covers this image."
                )
            sample_dict_for_source[k] = jnp.full((n_samps,), float(value))

    # Keys auto-filled from fixed values (not sampled) — drop from returned source-plane dicts.
    auto_filled_fixed_keys = {k for k in required_lens_keys if k not in samples}
    for i in range(1, n_images + 1):
        for k in (f"image_x{i}", f"image_y{i}"):
            if k not in samples:
                auto_filled_fixed_keys.add(k)

    result = image_samples_to_source_samples(
        sample_dict=sample_dict_for_source,
        lens_model_obj=ctx["lens_gw"],
        n_images=n_images,
        build_kwargs_lens_vectorised_fn=build_fn,
        n_subsample=sp_cfg.get("n_subsample"),
        seed=int(sp_cfg.get("seed", 42)),
        filter_std=sp_cfg.get("filter_std"),
    )

    # Apply MST scaling to source positions if k_mst was sampled
    if "k_mst" in sample_dict_for_source:
        k_mst_samples = jnp.asarray(sample_dict_for_source["k_mst"])
        result["source_plane_samples"]["y0gw"] = (
            result["source_plane_samples"]["y0gw"] * (1.0 - k_mst_samples)
        )
        result["source_plane_samples"]["y1gw"] = (
            result["source_plane_samples"]["y1gw"] * (1.0 - k_mst_samples)
        )
        result["y0gw_all"] = result["y0gw_all"] * (1.0 - k_mst_samples[:, None])
        result["y1gw_all"] = result["y1gw_all"] * (1.0 - k_mst_samples[:, None])
        result["y0gw_mean"] = result["y0gw_mean"] * (1.0 - k_mst_samples)
        result["y1gw_mean"] = result["y1gw_mean"] * (1.0 - k_mst_samples)
        if "source_plane_samples_filtered" in result:
            mask = result["consistent_mask"]
            k_filtered = k_mst_samples[mask]
            result["source_plane_samples_filtered"]["y0gw"] *= (1.0 - k_filtered)
            result["source_plane_samples_filtered"]["y1gw"] *= (1.0 - k_filtered)


    # Remove auto-filled fixed lens keys from returned source-plane sample dicts.
    # They are needed internally for ray-shooting, but should not appear as posterior samples.
    for k in auto_filled_fixed_keys:
        result.get("source_plane_samples", {}).pop(k, None)
        if "source_plane_samples_filtered" in result:
            result.get("source_plane_samples_filtered", {}).pop(k, None)

    # Convenience alias for plotting: optionally return filtered source-plane dict.
    use_filtered = bool(sp_cfg.get("use_filtered", False))
    if use_filtered and "source_plane_samples_filtered" in result:
        result["source_plane_samples_plot"] = result["source_plane_samples_filtered"]
    else:
        result["source_plane_samples_plot"] = result["source_plane_samples"]

    out_cfg = cfg_full.get("output", {})
    out_dir = out_cfg.get("output_dir")
    save_source_samples_path = _resolve_output_path(out_cfg.get("save_source_samples_path"), out_dir)
    save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
    json_tag = _output_json_tag(out_cfg)
    if json_tag:
        save_json_path = _append_tag_to_path(save_json_path, str(json_tag).strip().lower())
    save_json_path = _prefer_existing_tagged_json(save_json_path)
    _ensure_parent_dir(save_source_samples_path)
    _save_dict_npz(save_source_samples_path, result["source_plane_samples_plot"])
    _save_pipeline_json(
        save_json_path,
        ctx=ctx,
        samples_image_plane=sample_dict_for_source,
        source_plane_samples=result["source_plane_samples_plot"],
    )

    return result


def plot_source_posterior(
    source_samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot source-plane posterior samples with the same plot options as plot_posterior.

    Args:
        source_samples:
            Either:
              - output dict from `to_source_plane_samples(...)`, or
              - a plain samples dict already in source-plane parameter space.
        truths:
            Optional truth dict (e.g. includes `y0gw`, `y1gw`).
        cfg:
            Same `cfg['plot']` options used by `plot_posterior`.
    """
    if "source_plane_samples_plot" in source_samples:
        samples_for_plot = source_samples["source_plane_samples_plot"]
    elif "source_plane_samples" in source_samples:
        samples_for_plot = source_samples["source_plane_samples"]
    else:
        samples_for_plot = source_samples

    return _plot_samples_common(samples=samples_for_plot, truths=truths, cfg=cfg)


def compute_noise_snr_maps(ctx):
    """Per-pixel noise sigma and S/N maps for the EM observation.

    Model-based convention (matches PyAutoLens' noise map in expectation, and the
    verified gwemfish<->PAL comparison): ``sigma = sqrt(bg_rms^2 + max(model, 0)/t_exp)``
    with ``model`` the PSF-convolved clean image including lens light, and
    ``snr = data / sigma`` on the noisy observation.

    Returns:
        (noise_map, snr_map) as numpy arrays in gwemfish layout (row 0 = bottom).
    """
    import numpy as np

    if "lens_image" not in ctx or "em_obs" not in ctx:
        raise ValueError("Missing EM context. Run `setup_em_observation(...)` first.")

    em_cfg = ctx["cfg"]["em"]
    bg_rms = float(em_cfg["noise_simu_kwargs"]["background_rms"])
    t_exp = float(em_cfg["exposure_time"])
    model = np.asarray(
        ctx["lens_image"].model(
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=em_cfg["kwargs_source"],
            kwargs_lens_light=em_cfg["kwargs_lens_light"],
        )
    )
    noise_map = np.sqrt(bg_rms**2 + np.clip(model, 0.0, None) / t_exp)
    data = np.asarray(ctx["em_obs"]["data"])
    return noise_map, data / noise_map


def plot_system_observation(
    ctx: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot clean image, noisy observation, and S/N map with optional image overlays.

    Overlay mode: ``cfg['output']['system_plot_image_overlay']`` — ``"gw"`` (default,
    uses ``truth_params['image_x*']`` / ``image_y*`` or ``ctx['x_img_gw']``),
    ``"em"`` (lens equation at EM source: ``x_image_true_em`` / ``y_image_true_em``),
    ``"both"``, or ``"none"``.

    If `cfg['output']['save_system_plot_path']` is set, saves the figure there.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)

    if "lens_image" not in ctx or "em_obs" not in ctx:
        raise ValueError("Missing EM context. Run `setup_em_observation(...)` first.")

    kwargs_source = cfg_full["em"]["kwargs_source"]
    kwargs_lens_light = cfg_full["em"]["kwargs_lens_light"]
    kwargs_lens = ctx["kwargs_lens"]

    image_clean = ctx["lens_image"].model(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )
    data = ctx["em_obs"]["data"]
    noise_map = np.sqrt(
        float(cfg_full["em"]["noise_simu_kwargs"]["background_rms"]) ** 2
        + np.clip(np.asarray(image_clean), 0.0, None) / float(cfg_full["em"]["exposure_time"])
    )
    snr_map = np.asarray(data) / noise_map

    # Use sky coordinates if pixel grid is available; fallback to index axes.
    xx = yy = None
    if "pixel_grid" in ctx and hasattr(ctx["pixel_grid"], "pixel_coordinates"):
        try:
            xx, yy = ctx["pixel_grid"].pixel_coordinates
        except Exception:
            xx = yy = None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    if xx is not None and yy is not None:
        im1 = ax1.pcolormesh(xx, yy, np.asarray(image_clean), shading="auto")
        im2 = ax2.pcolormesh(xx, yy, np.asarray(data), shading="auto")
        im3 = ax3.pcolormesh(xx, yy, snr_map, shading="auto")
    else:
        im1 = ax1.imshow(np.asarray(image_clean), origin="lower")
        im2 = ax2.imshow(np.asarray(data), origin="lower")
        im3 = ax3.imshow(snr_map, origin="lower")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    ax1.set_title("Clean lensing image")
    ax2.set_title("Noisy observation")
    ax3.set_title("S/N map")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("RA [arcsec]")
        ax.set_ylabel("Dec [arcsec]")

    out_plot = cfg_full.get("output", {})
    overlay = str(out_plot.get("system_plot_image_overlay", "gw")).strip().lower()
    if overlay not in ("gw", "em", "both", "none"):
        overlay = "gw"

    tp = ctx.get("truth_params", {})
    gw_cfg = cfg_full.get("gw") or {}
    n_gw_cap = int(gw_cfg.get("n_images", 32))

    def _gw_xy() -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for i in range(n_gw_cap):
            kx, ky = f"image_x{i + 1}", f"image_y{i + 1}"
            if kx not in tp or ky not in tp:
                break
            xs.append(float(tp[kx]))
            ys.append(float(tp[ky]))
        if xs:
            return np.asarray(xs), np.asarray(ys)
        if "x_img_gw" in ctx and "y_img_gw" in ctx:
            xg = np.asarray(ctx["x_img_gw"]).ravel()
            yg = np.asarray(ctx["y_img_gw"]).ravel()
            if xg.size and xg.size == yg.size:
                return xg, yg
        return np.asarray([]), np.asarray([])

    def _em_xy() -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(tp.get("x_image_true_em", tp.get("x_image_true", [])))
        y = np.asarray(tp.get("y_image_true_em", tp.get("y_image_true", [])))
        return x, y

    any_legend = False
    if overlay in ("gw", "both"):
        x_gw, y_gw = _gw_xy()
        if x_gw.size > 0 and y_gw.size > 0 and x_gw.size == y_gw.size:
            ax1.scatter(x_gw, y_gw, color="k", marker="x", s=60, label="GW images")
            ax2.scatter(x_gw, y_gw, color="k", marker="x", s=60, label="GW images")
            any_legend = True
    if overlay in ("em", "both"):
        x_em, y_em = _em_xy()
        if x_em.size > 0 and y_em.size > 0 and x_em.size == y_em.size:
            ax1.scatter(
                x_em, y_em, color="tab:orange", marker="+", s=80, label="EM (lens eq.)"
            )
            ax2.scatter(
                x_em, y_em, color="tab:orange", marker="+", s=80, label="EM (lens eq.)"
            )
            any_legend = True
    if any_legend:
        ax1.legend()
        ax2.legend()

    fig.tight_layout()
    out_cfg = cfg_full.get("output", {})
    save_path = _resolve_output_path(out_cfg.get("save_system_plot_path"), out_cfg.get("output_dir"))
    _ensure_parent_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_psf(ctx, *, cfg=None):
    """Plot the PSF kernel used by the EM forward model (linear and log10 panels).

    Works for any ``psf_type`` (GAUSSIAN, PIXEL): the kernel is read from
    ``ctx['lens_image'].PSF.kernel_point_source``.

    For PIXEL that array *is* the convolution kernel. For GAUSSIAN it is a rendering
    of the PSF on the ``pixel_size`` grid -- herculens convolves via
    ``GaussianConvolution(sigma, pixel_grid.pixel_width)`` rather than with this array.
    ``setup_em_observation`` pins ``pixel_size`` to ``pix_scl``, so the two agree.

    If ``cfg['output']['save_psf_plot_path']`` is set, saves the figure there
    (resolved against ``output_dir`` like the other plot functions).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)

    if "lens_image" not in ctx:
        raise ValueError("Missing EM context. Run `setup_em_observation(...)` first.")

    kernel = np.asarray(ctx["lens_image"].PSF.kernel_point_source, float)
    pix_scl = float(cfg_full["em"]["pixel_grid_kwargs"]["pix_scl"])
    half = kernel.shape[0] * pix_scl / 2.0
    ext = [-half, half, -half, half]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    im1 = ax1.imshow(kernel, origin="lower", extent=ext)
    log_kernel = np.log10(np.where(kernel > 0, kernel, np.nan))
    im2 = ax2.imshow(log_kernel, origin="lower", extent=ext)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    psf_type = str(cfg_full["em"]["psf_kwargs"].get("psf_type", "GAUSSIAN"))
    ax1.set_title(f"PSF kernel ({psf_type})")
    ax2.set_title("PSF kernel (log10)")
    for ax in (ax1, ax2):
        ax.set_xlabel("RA [arcsec]")
        ax.set_ylabel("Dec [arcsec]")

    fig.tight_layout()
    out_cfg = cfg_full.get("output", {})
    save_path = _resolve_output_path(out_cfg.get("save_psf_plot_path"), out_cfg.get("output_dir"))
    _ensure_parent_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_lens_system_with_source_localization(
    source_samples: Dict[str, Any],
    kwargs_lens: Any,
    lens_model_list: Any,
    z_lens: float,
    z_source: float,
    truths_source: Optional[Dict[str, float]] = None,
    *,
    level: float = 0.90,
    figsize: Tuple[float, float] = (10, 5),
    cmap_string: str = "RdPu",
    numPix: int = 600,
    deltaPix: float = 0.01,
    point_source: bool = True,
    with_caustics: bool = True,
    fast_caustic: bool = True,
    coord_inverse: bool = False,
    contour_color: str = "cyan",
    contour_linestyle: str = "-",
    contour_linewidth: float = 2.0,
    scatter_alpha: float = 0.08,
    scatter_size: float = 2.0,
    truth_color: str = "k",
    save_path: Optional[str] = None,
) -> Any:
    """Plot lenstronomy lens map and overlay one source-localization contour.

    This is a generic helper for tutorials/notebooks: it overlays a highest
    posterior density contour (default 90%) from one source posterior sample set
    (`y0gw`, `y1gw`) on top of a lenstronomy caustic plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from lenstronomy.Plots import lens_plot
    from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel

    if "y0gw" not in source_samples or "y1gw" not in source_samples:
        raise ValueError("source_samples must include 'y0gw' and 'y1gw'.")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1), e.g. 0.90.")

    x = np.asarray(source_samples["y0gw"]).ravel()
    y = np.asarray(source_samples["y1gw"]).ravel()
    if x.size < 10 or y.size < 10:
        raise ValueError("Need enough y0gw/y1gw samples to estimate contour.")

    lens_model_plot = LenstronomyLensModel(
        lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
    )

    fig, ax = plt.subplots(figsize=figsize)
    lens_plot.lens_model_plot(
        ax,
        lensModel=lens_model_plot,
        kwargs_lens=kwargs_lens,
        sourcePos_x=float(np.mean(x)),
        sourcePos_y=float(np.mean(y)),
        point_source=point_source,
        with_caustics=with_caustics,
        fast_caustic=fast_caustic,
        coord_inverse=coord_inverse,
        numPix=numPix,
        deltaPix=deltaPix,
        cmap_string=cmap_string,
    )

    for obj in list(ax.get_images()) + list(ax.collections):
        if hasattr(obj, "set_cmap"):
            obj.set_cmap(cmap_string)

    vals = np.vstack([x, y])
    kde = gaussian_kde(vals)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    padx = 0.15 * (xmax - xmin + 1e-12)
    pady = 0.15 * (ymax - ymin + 1e-12)

    xi = np.linspace(xmin - padx, xmax + padx, 220)
    yi = np.linspace(ymin - pady, ymax + pady, 220)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    z_level = z_sorted[np.searchsorted(cdf, level)]

    ax.scatter(x, y, s=scatter_size, alpha=scatter_alpha, color=contour_color)
    ax.contour(
        X,
        Y,
        Z,
        levels=[z_level],
        colors=[contour_color],
        linewidths=contour_linewidth,
        linestyles=[contour_linestyle],
    )
    ax.scatter([np.mean(x)], [np.mean(y)], s=40, color=contour_color, marker="o", label="Posterior mean")

    if truths_source is not None and "y0gw" in truths_source and "y1gw" in truths_source:
        ax.scatter(
            [float(truths_source["y0gw"])],
            [float(truths_source["y1gw"])],
            s=70,
            color=truth_color,
            marker="x",
            label="True source",
        )

    ax.set_xlabel("y0gw [arcsec]")
    ax.set_ylabel("y1gw [arcsec]")
    ax.set_title(f"Lens system with {int(level * 100)}% source-localization contour")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_lens_system_with_source_local_setup(
    source_samples: Dict[str, Any],
    ctx: Dict[str, Any],
    truths_source: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper using lens config directly from pipeline context.

    Reads ``kwargs_lens``, ``lens_model_list``, ``zl``, and ``zs`` from ``ctx`` and
    forwards all plotting options to ``plot_lens_system_with_source_localization``.
    """
    if "kwargs_lens" not in ctx:
        raise ValueError("ctx must include 'kwargs_lens'. Run setup_em_observation(...) first.")
    if "cfg" not in ctx or "lens" not in ctx["cfg"]:
        raise ValueError("ctx must include cfg['lens'] with lens_model_list/zl/zs.")

    lens_cfg = ctx["cfg"]["lens"]
    lens_model_list = lens_cfg.get("lens_model_list")
    z_lens = lens_cfg.get("zl")
    z_source = lens_cfg.get("zs")
    if lens_model_list is None or z_lens is None or z_source is None:
        raise ValueError("ctx['cfg']['lens'] must contain 'lens_model_list', 'zl', and 'zs'.")

    truths_source_use = truths_source
    if truths_source_use is None:
        gw_cfg = ((ctx.get("cfg") or {}).get("gw") or {})
        src = gw_cfg.get("source_pos")
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            truths_source_use = {"y0gw": float(src[0]), "y1gw": float(src[1])}

    return plot_lens_system_with_source_localization(
        source_samples=source_samples,
        kwargs_lens=ctx["kwargs_lens"],
        lens_model_list=lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
        truths_source=truths_source_use,
        **kwargs,
    )


def plot_source_plane_caustic_with_localization(
    source_samples: Dict[str, Any],
    kwargs_lens: Any,
    lens_model_list: Any,
    z_lens: float,
    z_source: float,
    truths_source: Optional[Dict[str, float]] = None,
    *,
    level: float = 0.90,
    figsize: Tuple[float, float] = (6, 6),
    contour_color: str = "tab:blue",
    caustic_color: str = "k",
    caustic_linewidth: float = 1.5,
    scatter_alpha: float = 0.08,
    scatter_size: float = 2.0,
    truth_color: str = "red",
    show_scatter: bool = True,
    show_posterior_mean: bool = True,
    show_truth: bool = True,
    save_path: Optional[str] = None,
    compute_window: float = 5.0,
    grid_scale: float = 0.01,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Any:
    """Plot source-plane caustic(s) and one localization contour only.

    This helper intentionally does NOT draw critical curves or image-plane maps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    if "y0gw" not in source_samples or "y1gw" not in source_samples:
        raise ValueError("source_samples must include 'y0gw' and 'y1gw'.")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1), e.g. 0.90.")

    x = np.asarray(source_samples["y0gw"]).ravel()
    y = np.asarray(source_samples["y1gw"]).ravel()
    if x.size < 10 or y.size < 10:
        raise ValueError("Need enough y0gw/y1gw samples to estimate contour.")

    lens_model = LenstronomyLensModel(
        lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
    )
    ext = LensModelExtensions(lensModel=lens_model)
    (_ra_crit, _dec_crit, ra_caustic, dec_caustic) = ext.critical_curve_caustics(
        kwargs_lens=kwargs_lens,
        compute_window=compute_window,
        grid_scale=grid_scale,
        center_x=center_x,
        center_y=center_y,
    )

    vals = np.vstack([x, y])
    kde = gaussian_kde(vals)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    padx = 0.15 * (xmax - xmin + 1e-12)
    pady = 0.15 * (ymax - ymin + 1e-12)
    xi = np.linspace(xmin - padx, xmax + padx, 220)
    yi = np.linspace(ymin - pady, ymax + pady, 220)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    z_level = z_sorted[np.searchsorted(cdf, level)]

    fig, ax = plt.subplots(figsize=figsize)
    for xc, yc in zip(ra_caustic, dec_caustic):
        ax.plot(np.asarray(xc), np.asarray(yc), color=caustic_color, lw=caustic_linewidth)

    if show_scatter:
        ax.scatter(x, y, s=scatter_size, alpha=scatter_alpha, color=contour_color)
    ax.contour(X, Y, Z, levels=[z_level], colors=[contour_color], linewidths=2.0, linestyles=["-"])
    if show_posterior_mean:
        ax.scatter([np.mean(x)], [np.mean(y)], s=40, color=contour_color, marker="o", label="Posterior mean")

    if show_truth and truths_source is not None and "y0gw" in truths_source and "y1gw" in truths_source:
        ax.scatter(
            [float(truths_source["y0gw"])],
            [float(truths_source["y1gw"])],
            s=70,
            color=truth_color,
            marker="x",
            label="True source",
        )

    ax.set_xlabel("y0gw [arcsec]")
    ax.set_ylabel("y1gw [arcsec]")
    ax.set_title(f"Source-plane caustic with {int(level * 100)}% localization contour")
    ax.set_aspect("equal", adjustable="box")
    if show_posterior_mean or (show_truth and truths_source is not None):
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_source_plane_caustic_with_localization_from_setup(
    source_samples: Dict[str, Any],
    ctx: Dict[str, Any],
    truths_source: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper for source-plane-caustic plot using setup context."""
    if "kwargs_lens" not in ctx:
        raise ValueError("ctx must include 'kwargs_lens'. Run setup_em_observation(...) first.")
    if "cfg" not in ctx or "lens" not in ctx["cfg"]:
        raise ValueError("ctx must include cfg['lens'] with lens_model_list/zl/zs.")

    lens_cfg = ctx["cfg"]["lens"]
    lens_model_list = lens_cfg.get("lens_model_list")
    z_lens = lens_cfg.get("zl")
    z_source = lens_cfg.get("zs")
    if lens_model_list is None or z_lens is None or z_source is None:
        raise ValueError("ctx['cfg']['lens'] must contain 'lens_model_list', 'zl', and 'zs'.")

    truths_source_use = truths_source
    if truths_source_use is None:
        gw_cfg = ((ctx.get("cfg") or {}).get("gw") or {})
        src = gw_cfg.get("source_pos")
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            truths_source_use = {"y0gw": float(src[0]), "y1gw": float(src[1])}

    return plot_source_plane_caustic_with_localization(
        source_samples=source_samples,
        kwargs_lens=ctx["kwargs_lens"],
        lens_model_list=lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
        truths_source=truths_source_use,
        **kwargs,
    )

