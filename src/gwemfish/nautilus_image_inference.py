"""
Nautilus image-plane inference: sample GW image_x/y directly; full HMC-equivalent likelihood.

EM-only delegates to shared pixel likelihood in nautilus_common (same as nautilus-source).
"""

import warnings

import jax
import scipy.stats as sps

from .ellipticity_reparam import (
    mass_qphi_prefixes_from_entries,
    qphi_derived_keys,
    swap_ellipticity_for_qphi,
    validate_parametrization,
)
from .nautilus_common import (
    build_em_only_nautilus_problem,
    build_nautilus_prior,
    layout_defaults_from_registry,
    parse_cfg_priors,
    probmodel_log_likelihood,
    _extract_dist_from_callable,
    _numpyro_to_spec,
)
from .priors import DEFAULT_PRIORS_GW_SOURCE_PLANE


def image_position_default_dists(n_images, truth_params, half_width):
    default_dists = {}
    for i in range(n_images):
        xk = f"image_x{i + 1}"
        yk = f"image_y{i + 1}"
        if xk not in truth_params or yk not in truth_params:
            raise ValueError(f"Missing truth image positions '{xk}'/'{yk}' in ctx['truth_params']")
        xt = float(truth_params[xk])
        yt = float(truth_params[yk])
        default_dists[xk] = sps.uniform(xt - half_width, 2.0 * half_width)
        default_dists[yk] = sps.uniform(yt - half_width, 2.0 * half_width)
    return default_dists


def dist_from_prior_callable(callable_prior):
    d = _extract_dist_from_callable(callable_prior)
    if d is None:
        return None, None
    kind, val = _numpyro_to_spec(d)
    if kind == "fixed":
        return "fixed", val
    if kind == "dist":
        return "dist", val
    return None, None


def build_scipy_priors_for_probmodel(probmodel, built, cfg_full):
    from .simple_pipeline import (
        _fixed_literal_prior_keys,
        _float_from_user_prior_literal,
    )

    priors_user = cfg_full.get("priors", {})
    fixed_literal_keys = _fixed_literal_prior_keys(priors_user)
    prior_sample = probmodel.get_sample(
        prng_key=jax.random.PRNGKey(int(cfg_full["inference"]["prior_sample_rng_key"]))
    )
    # get_sample() (herculens NumpyroModel) filters only on is_observed, so in q_phi
    # mode it also returns the numpyro.deterministic lens_e1/lens_e2 sites -- exclude
    # those or nautilus would sample them as if independent of lens_q/lens_phi.
    parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
    if built.get("entries") is not None:
        qphi_prefixes = mass_qphi_prefixes_from_entries(built["entries"], parametrization)
    else:
        qphi_prefixes = frozenset({"lens"}) if parametrization == "q_phi" else frozenset()
    derived_keys = qphi_derived_keys(qphi_prefixes)
    keys_all = [k for k in prior_sample.keys() if k not in derived_keys]
    keys_to_sample = [k for k in keys_all if k not in fixed_literal_keys]

    default_dists = {}
    registry_fixed = {}

    if built["use_layout"] and built["entries"] is not None and built["registry"] is not None:
        entries_for_defaults = [
            e for e in built["entries"]
            if e.flat_key not in fixed_literal_keys
        ]
        reg_dists, reg_fixed = layout_defaults_from_registry(
            entries_for_defaults, built["registry"]
        )
        default_dists.update(reg_dists)
        registry_fixed.update(reg_fixed)

    # Excluding the derived keys from keys_to_sample is not enough: the nautilus
    # prior is built from default_dists, not from keys_to_sample, and the layout
    # registry above put '{prefix}_e1'/'{prefix}_e2' straight into it. Without
    # this swap nautilus samples e1/e2 *and* q as independent parameters -- the
    # lens model then reads the sampled e1/e2 and ignores q entirely, so q comes
    # back as its prior. Same call the other three nautilus builders make
    # (nautilus_common.py, nautilus_source_inference.py x2).
    if qphi_prefixes:
        swap_ellipticity_for_qphi(default_dists, qphi_prefixes)

    n_images = built["n_images"]
    truth_params = built["truth_params"]
    half_width = built["half_width"]

    if mode_needs_image_boxes(built, keys_to_sample):
        default_dists.update(
            image_position_default_dists(n_images, truth_params, half_width)
        )

    for key in keys_to_sample:
        if key in default_dists or key in registry_fixed:
            continue
        if key in probmodel.priors:
            kind, val = dist_from_prior_callable(probmodel.priors[key])
            if kind == "fixed":
                registry_fixed[key] = val
            elif kind == "dist":
                default_dists[key] = val
            else:
                warnings.warn(f"Could not build scipy prior for '{key}'; skipping.")
        elif key.startswith("image_x") or key.startswith("image_y"):
            img_dists = image_position_default_dists(n_images, truth_params, half_width)
            if key in img_dists:
                default_dists[key] = img_dists[key]

    bounds = DEFAULT_PRIORS_GW_SOURCE_PLANE
    scipy_overrides, cfg_fixed = parse_cfg_priors(
        priors_user, default_dists, bounds
    )
    fixed_params = {**registry_fixed, **cfg_fixed}
    for k in fixed_literal_keys:
        fixed_params[k] = _float_from_user_prior_literal(priors_user[k])

    for key in keys_to_sample:
        if key in fixed_params:
            continue
        if key not in default_dists and key not in scipy_overrides:
            raise ValueError(
                f"No scipy prior for sampled key '{key}'. "
                "Set cfg['priors'] or ensure truth_params / layout registry cover it."
            )

    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)
    if fixed_params:
        print(f"  Fixed params (not sampled): {sorted(fixed_params.keys())}")
    return prior, fixed_params, keys_to_sample


def mode_needs_image_boxes(built, keys_to_sample):
    for k in keys_to_sample:
        if k.startswith("image_x") or k.startswith("image_y"):
            return True
    return False


def build_image_plane_problem(ctx, mode, cfg):
    """Returns (nautilus.Prior, log_likelihood, param_names)."""
    from .simple_pipeline import _build_inference_probmodel, _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)

    if mode == "EM-only":
        return build_em_only_nautilus_problem(ctx, cfg)

    if mode not in ("GW-only", "EM+GW"):
        raise ValueError(
            "nautilus-image supports mode 'GW-only', 'EM+GW', and 'EM-only' only"
        )

    built = _build_inference_probmodel(ctx, mode, cfg_full)
    probmodel = built["probmodel"]
    likelihood_seed = built["likelihood_seed"]

    prior, fixed_params, _ = build_scipy_priors_for_probmodel(
        probmodel, built, cfg_full
    )

    def log_likelihood(params):
        full = {**fixed_params, **params}
        return probmodel_log_likelihood(probmodel, full, rng_key=likelihood_seed)

    print("Warming up image-plane log_likelihood (triggers JAX compilation)...")
    try:
        truth_params = ctx.get("truth_params", {}) or {}
        warm = {k: float(truth_params[k]) for k in prior.keys if k in truth_params}
        warm.update({k: fixed_params[k] for k in fixed_params})
        lv = log_likelihood(warm)
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as exc:
        warnings.warn(f"Warm-up call failed: {exc}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names
