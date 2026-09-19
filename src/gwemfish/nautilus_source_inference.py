"""
Nautilus nested-sampler interface for GW source-plane inference.

Source-plane GW parametrization: sample y0gw/y1gw, solve lens equation for images.
EM-only uses shared pixel likelihood from nautilus_common.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import scipy.stats as sps

from .lens_setup import build_lens_solver, solve_and_select
from .data_sim import compute_gw_from_images
from .ellipticity_reparam import (
    expand_qphi_params,
    mass_qphi_prefixes_from_entries,
    swap_ellipticity_for_qphi,
    validate_parametrization,
    warn_if_ellipticity_prior_keys_unused,
)
from .priors import DEFAULT_PRIORS_GW_SOURCE_PLANE
from .nautilus_common import (
    EM_EXTRA_DEFAULT_DISTS,
    build_em_only_nautilus_problem,
    build_nautilus_prior,
    layout_defaults_from_registry,
    parse_cfg_priors,
    run_nautilus,
)


def _build_kwargs_lens(params):
    return [
        {
            "theta_E":  float(params["lens_theta_E"]),
            "e1":       float(params["lens_e1"]),
            "e2":       float(params["lens_e2"]),
            "gamma":    float(params["lens_gamma"]),
            "center_x": float(params.get("lens_center_x", 0.0)),
            "center_y": float(params.get("lens_center_y", 0.0)),
        },
        {
            "gamma1": float(params["lens_gamma1"]),
            "gamma2": float(params["lens_gamma2"]),
            "ra_0":   0.0,
            "dec_0":  0.0,
        },
    ]


def _normal_logpdf(x, mu, sigma):
    return -0.5 * jnp.sum(((x - mu) / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma ** 2))


def _gw_loglike_from_images(x_pos, y_pos, kwargs_lens, lens_gw,
                             T_star, dL, gw_obs, error_scales):
    sigma_td_frac = error_scales.get("sigma_td", 0.3)
    sigma_dL_frac = error_scales.get("sigma_dL_eff", 0.3)
    td_floor = error_scales.get("sigma_td_floor", 1.0)

    _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
        jnp.array(x_pos), jnp.array(y_pos), kwargs_lens, lens_gw, T_star, dL
    )

    obs_td = jnp.array(gw_obs["time_delays"])
    obs_dL_eff = jnp.array(gw_obs["dL_eff"])

    sigma_td = jnp.maximum(td_floor, sigma_td_frac * obs_td)
    sigma_dL_eff = sigma_dL_frac * obs_dL_eff

    return float(
        _normal_logpdf(model_td, obs_td, sigma_td)
        + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff)
    )


def _tnorm(lo, hi, loc=0.0, scale=0.3):
    a, b = (lo - loc) / scale, (hi - loc) / scale
    return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)


_GW_DEFAULT_DISTS = {
    "lens_theta_E":  lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_e1":       lambda b: _tnorm(*b),
    "lens_e2":       lambda b: _tnorm(*b),
    "lens_gamma":    lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_gamma1":   lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_gamma2":   lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_center_x": lambda b: sps.norm(loc=0.0, scale=0.1),
    "lens_center_y": lambda b: sps.norm(loc=0.0, scale=0.1),
    "T_star":        lambda b: sps.loguniform(b[0], b[1]),
    "dL":            lambda b: sps.uniform(b[0], b[1] - b[0]),
    "y0gw":          lambda b: sps.uniform(b[0], b[1] - b[0]),
    "y1gw":          lambda b: sps.uniform(b[0], b[1] - b[0]),
}


def _gw_extra_defaults(bounds, keys):
    return {k: _GW_DEFAULT_DISTS[k](bounds[k]) for k in keys}


def _solve_images(solver, solver_params, y0, y1, kwargs_lens,
                  lens_center_x, lens_center_y, n_images, lens_gw=None):
    """Solve at one sampled point, rejecting configurations with the wrong image count.

    The count comes from ``select_images``' ``n_distinct``, not from ``len(x_pos)``:
    the returned array is always ``n_images`` long by construction, so a length test
    is a compile-time constant and can never fire. That is why the old check missed
    padded/duplicated helens solutions entirely.
    """
    x_pos, y_pos, _, flags = solve_and_select(
        solver, solver_params, jnp.array([y0, y1]), kwargs_lens, lens_gw,
        n_images, lens_center_x, lens_center_y,
    )
    if int(flags["n_distinct"]) != n_images:
        return None, None
    return list(x_pos), list(y_pos)


def validate_helens_solver(solver, solver_params, kwargs_lens_truth,
                            y_truth, x_images_truth, y_images_truth,
                            lens_center_x=0.0, lens_center_y=0.0,
                            tol=0.05, lens_gw=None):
    """Check the solver reproduces the simulated images at the true parameters.

    Counts *distinct* images rather than the length of the returned array: the
    array is always ``n_images`` long by construction, so the old length test was a
    compile-time constant that could never fail, and padded or duplicated solutions
    sailed through it.
    """
    n_images = len(x_images_truth)
    x_sol, y_sol, _, flags = solve_and_select(
        solver, solver_params, jnp.array(y_truth), kwargs_lens_truth, lens_gw,
        n_images, lens_center_x, lens_center_y,
    )

    n_distinct = int(flags["n_distinct"])
    if n_distinct != n_images:
        raise RuntimeError(
            f"Solver found {n_distinct} distinct images at the true parameters, "
            f"expected {n_images} (padding={int(flags['n_padding'])}, "
            f"duplicates={int(flags['n_duplicate'])}, "
            f"central={bool(flags['has_central'])}). "
            "Raise solver_params['helens']['nsubdivisions'] or ['nsolutions'], or "
            "switch cfg['gw']['solver_params']['backend'] to 'jaxtronomy'."
        )

    x_sorted = np.sort(np.array(x_sol))
    x_truth_s = np.sort(np.array(x_images_truth))
    y_sorted = np.sort(np.array(y_sol))
    y_truth_s = np.sort(np.array(y_images_truth))
    max_err = float(np.max(np.abs(
        np.concatenate([x_sorted - x_truth_s, y_sorted - y_truth_s])
    )))

    # Name the finder actually in use: this path is no longer helens-only.
    finder = getattr(solver, "finder", None)
    backend = {"HelensImageFinder": "helens",
               "JaxtronomyImageFinder": "jaxtronomy"}.get(type(finder).__name__, "solver")

    if max_err > tol:
        warnings.warn(
            f"{backend} solver: max image position residual = {max_err:.4f} arcsec "
            f"(tol={tol}). " + (
                "Lower solver_params['helens']['pixel_scale_factor'] or raise "
                "['nsubdivisions']." if backend == "helens" else
                "Raise solver_params['jaxtronomy']['Nmeas'].")
        )
    else:
        print(f"{backend} solver validation passed: "
              f"max residual = {max_err:.6f} arcsec")

    return max_err


def build_nautilus_solver(ctx, cfg_full, n_images):
    """Build the solver for a nautilus-source run, honouring cfg solver settings.

    Same construction path as the gradient-based source-plane methods, so a
    nautilus-source run and an hmc-source run on one cfg use identically-configured
    solvers rather than silently differing in accuracy.

    Nested sampling needs no derivatives, so the Newton polish is optional here --
    and it is the only method where it is. ``cfg["nautilus"]["polish"]``:

      "auto" (default)  polish only when it changes the answer: skipped for the
                        jaxtronomy finders, whose positions are already exact or
                        converged, and applied for helens, whose triangle search is
                        only good to ~0.05 arcsec.
      True / False      force it on or off. False reproduces the historical raw-helens
                        path exactly, at the same cost.
    """
    nautilus_cfg = cfg_full.get("nautilus", {})
    lens_cfg = cfg_full["lens"]
    solver_params_cfg = dict(cfg_full.get("gw", {}).get("solver_params") or {})

    # Deprecated alias: cfg["nautilus"]["solver_backend"] predates the shared
    # solver_params["backend"] and means the same thing.
    legacy_backend = nautilus_cfg.get("solver_backend")
    if legacy_backend is not None and "backend" not in solver_params_cfg:
        warnings.warn(
            "cfg['nautilus']['solver_backend'] is deprecated; use "
            "cfg['gw']['solver_params']['backend'], which applies to every method.",
            DeprecationWarning, stacklevel=2,
        )
        solver_params_cfg["backend"] = legacy_backend

    pixel_grid = ctx.get("pixel_grid")
    if pixel_grid is None:
        from .data_sim import setup_pixel_grid
        pg_kwargs = cfg_full.get("em", {}).get("pixel_grid_kwargs", {})
        pixel_grid = setup_pixel_grid(**pg_kwargs)
        print("  pixel_grid not in ctx (EM disabled) — created from cfg pixel_grid_kwargs.")

    kwargs_lens_ctx = ctx.get("kwargs_lens") or lens_cfg["kwargs_lens"]
    lens_center = (float(kwargs_lens_ctx[0].get("center_x", 0.0)),
                   float(kwargs_lens_ctx[0].get("center_y", 0.0)))

    polish_cfg = nautilus_cfg.get("polish", "auto")
    if polish_cfg == "auto":
        from .image_finders import resolve_backend
        resolved = resolve_backend(
            solver_params_cfg.get("backend", "auto"),
            ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
        )
        polish = resolved == "helens"
    else:
        polish = bool(polish_cfg)

    solver, solver_params, resolved = build_lens_solver(
        ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
        lens_cfg["zl"], lens_cfg["zs"], ctx["lens_gw"],
        solver_params=solver_params_cfg, n_images=n_images,
        pixel_grid=pixel_grid, polish=polish, lens_center=lens_center,
    )
    print(f"  solver: backend={resolved['backend']} polish={polish} "
          f"nsolutions={resolved['nsolutions']}")
    return solver, solver_params, resolved


def build_gw_source_plane_problem(ctx, cfg):
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    gw_cfg = cfg_full.get("gw", {})
    nautilus_cfg = cfg_full.get("nautilus", {})

    lens_gw = ctx["lens_gw"]
    gw_obs = ctx["gw_obs"]
    n_images = len([k for k in ctx.get("truth_params", {}) if k.startswith("image_x")])
    error_scales = gw_cfg.get("error_scales", {})
    truth_params = ctx.get("truth_params", {})

    bounds = {**DEFAULT_PRIORS_GW_SOURCE_PLANE, **gw_cfg.get("source_plane_bounds", {})}

    use_layout = bool(cfg_full.get("use_parameter_layout"))
    kwargs_truth = ctx["kwargs_lens"] if use_layout else _build_kwargs_lens(truth_params)

    solver, solver_params, _ = build_nautilus_solver(ctx, cfg_full, n_images)

    y_truth = list(gw_cfg.get("source_pos", [truth_params.get("y0gw", 0.05),
                                               truth_params.get("y1gw", 1e-6)]))
    x_img_truth = [float(truth_params[f"image_x{i+1}"]) for i in range(n_images)]
    y_img_truth = [float(truth_params[f"image_y{i+1}"]) for i in range(n_images)]
    validate_helens_solver(solver, solver_params, kwargs_truth,
                            y_truth, x_img_truth, y_img_truth,
                            tol=nautilus_cfg.get("solver_validation_tol", 0.05),
                            lens_gw=lens_gw)

    def solve_fn(y0, y1, kwargs_lens):
        cx = float(kwargs_lens[0].get("center_x", 0.0))
        cy = float(kwargs_lens[0].get("center_y", 0.0))
        return _solve_images(solver, solver_params, y0, y1, kwargs_lens, cx, cy,
                             n_images, lens_gw=lens_gw)

    if use_layout:
        from .parameter_layout import (
            build_mass_parameter_entries, build_priors_registry, unpack_to_kwargs,
        )
        mass_model = ctx["lens_mass_model"]
        entries = build_mass_parameter_entries(mass_model, kwargs_lens=ctx["kwargs_lens"])
        registry = build_priors_registry(entries, mass_model=mass_model, user_priors=None)
        default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
        default_dists.update(_gw_extra_defaults(bounds, ("T_star", "dL", "y0gw", "y1gw")))
        n_mass = len(mass_model.func_list)
        parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
        qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)

        def make_kwargs_lens(full):
            kl, _, _ = unpack_to_kwargs(full, entries, n_mass=n_mass,
                                        n_source=0, n_lens_light=0)
            return kl
    else:
        registry_fixed = {}
        default_dists = dict(_GW_DEFAULT_DISTS)
        parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
        qphi_prefixes = frozenset({"lens"}) if parametrization == "q_phi" else frozenset()

        def make_kwargs_lens(full):
            return _build_kwargs_lens(full)

    if qphi_prefixes:
        swap_ellipticity_for_qphi(default_dists, qphi_prefixes)

    cfg_priors = cfg_full.get("priors", {})
    warn_if_ellipticity_prior_keys_unused(
        cfg_priors, parametrization, qphi_prefixes or frozenset({"lens"})
    )
    scipy_overrides, cfg_fixed = parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}
        full = expand_qphi_params(full, qphi_prefixes)
        kwargs_lens = make_kwargs_lens(full)
        x_pos, y_pos = solve_fn(float(full["y0gw"]), float(full["y1gw"]), kwargs_lens)
        if x_pos is None:
            return -1e300
        return _gw_loglike_from_images(
            x_pos, y_pos, kwargs_lens, lens_gw,
            float(full["T_star"]), float(full["dL"]),
            gw_obs, error_scales,
        )

    print("Warming up GW source-plane log_likelihood (triggers JAX compilation)...")
    try:
        src_truth = list(gw_cfg.get("source_pos", [0.05, 1e-6]))
        warmup_src = (truth_params.get("y0gw", src_truth[0]),
                      truth_params.get("y1gw", src_truth[1]))
        lv = log_likelihood({**truth_params,
                             "y0gw": warmup_src[0], "y1gw": warmup_src[1]})
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def build_em_gw_source_plane_problem(ctx, cfg):
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    gw_cfg = cfg_full.get("gw", {})
    nautilus_cfg = cfg_full.get("nautilus", {})

    lens_gw = ctx["lens_gw"]
    lens_image = ctx["lens_image"]
    noise = ctx["noise_inf"]
    gw_obs = ctx["gw_obs"]
    em_obs = ctx["em_obs"]
    n_images = len([k for k in ctx.get("truth_params", {}) if k.startswith("image_x")])
    error_scales = gw_cfg.get("error_scales", {})
    truth_params = ctx.get("truth_params", {})

    bounds = {**DEFAULT_PRIORS_GW_SOURCE_PLANE, **gw_cfg.get("source_plane_bounds", {})}

    use_layout = bool(cfg_full.get("use_parameter_layout"))
    kwargs_truth = ctx["kwargs_lens"] if use_layout else _build_kwargs_lens(truth_params)

    solver, solver_params, _ = build_nautilus_solver(ctx, cfg_full, n_images)

    y_truth = list(gw_cfg.get("source_pos", [truth_params.get("y0gw", 0.05),
                                               truth_params.get("y1gw", 1e-6)]))
    x_img_truth = [float(truth_params[f"image_x{i+1}"]) for i in range(n_images)]
    y_img_truth = [float(truth_params[f"image_y{i+1}"]) for i in range(n_images)]
    validate_helens_solver(solver, solver_params, kwargs_truth,
                            y_truth, x_img_truth, y_img_truth,
                            tol=nautilus_cfg.get("solver_validation_tol", 0.05),
                            lens_gw=lens_gw)

    def solve_fn(y0, y1, kwargs_lens):
        cx = float(kwargs_lens[0].get("center_x", 0.0))
        cy = float(kwargs_lens[0].get("center_y", 0.0))
        return _solve_images(solver, solver_params, y0, y1, kwargs_lens, cx, cy,
                             n_images, lens_gw=lens_gw)

    em_data = jnp.array(em_obs["data"])

    if use_layout:
        from .parameter_layout import (
            build_parameter_layout, build_priors_registry, unpack_to_kwargs,
        )
        from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

        em_sec = cfg_full.get("em") or {}
        ks_tmpl = em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
        kll_tmpl = em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
        entries, _ = build_parameter_layout(
            lens_image,
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=ks_tmpl,
            kwargs_lens_light=kll_tmpl,
        )
        registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
        default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
        default_dists.update({
            "T_star": _GW_DEFAULT_DISTS["T_star"](bounds["T_star"]),
            "dL": _GW_DEFAULT_DISTS["dL"](bounds["dL"]),
            "noise_sigma_bkg": EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None),
        })
        n_mass = len(lens_image.MassModel.func_list)
        n_source = len(lens_image.SourceModel.func_list)
        n_lens_light = len(lens_image.LensLightModel.func_list)
        parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
        qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)
    else:
        registry_fixed = {}
        default_dists = {**_GW_DEFAULT_DISTS, **EM_EXTRA_DEFAULT_DISTS}
        parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
        qphi_prefixes = frozenset({"lens"}) if parametrization == "q_phi" else frozenset()

    if qphi_prefixes:
        swap_ellipticity_for_qphi(default_dists, qphi_prefixes)

    cfg_priors = cfg_full.get("priors", {})
    warn_if_ellipticity_prior_keys_unused(
        cfg_priors, parametrization, qphi_prefixes or frozenset({"lens"})
    )
    scipy_overrides, cfg_fixed = parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}
        full = expand_qphi_params(full, qphi_prefixes)

        if use_layout:
            kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
                full, entries, n_mass=n_mass,
                n_source=n_source, n_lens_light=n_lens_light,
            )
            y0 = float(kwargs_source[0]["center_x"])
            y1 = float(kwargs_source[0]["center_y"])
        else:
            kwargs_lens = _build_kwargs_lens(full)
            y0, y1 = float(full["y0gw"]), float(full["y1gw"])
            kwargs_source = [{
                "amp": float(full["source_amp"]),
                "R_sersic": float(full["source_R_sersic"]),
                "n_sersic": float(full["source_n"]),
                "e1": float(full["source_e1"]),
                "e2": float(full["source_e2"]),
                "center_x": y0,
                "center_y": y1,
            }]
            kwargs_lens_light = [{
                "amp": float(full["light_amp"]),
                "R_sersic": float(full["light_R_sersic"]),
                "n_sersic": float(full["light_n"]),
                "e1": float(full["light_e1"]),
                "e2": float(full["light_e2"]),
                "center_x": float(full["light_center_x"]),
                "center_y": float(full["light_center_y"]),
            }]

        x_pos, y_pos = solve_fn(y0, y1, kwargs_lens)
        if x_pos is None:
            return -1e300

        loglike_gw = _gw_loglike_from_images(
            x_pos, y_pos, kwargs_lens, lens_gw,
            float(full["T_star"]), float(full["dL"]),
            gw_obs, error_scales,
        )

        sigma_bkg = float(full["noise_sigma_bkg"])
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        loglike_em = float(
            jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                             + jnp.log(2 * jnp.pi * model_var)))
        )
        return loglike_gw + loglike_em

    print("Warming up EM+GW source-plane log_likelihood (triggers JAX compilation)...")
    try:
        warmup_params = dict(truth_params)
        if not use_layout:
            src_truth = list(gw_cfg.get("source_pos", [0.05, 1e-6]))
            warmup_params.setdefault("y0gw", src_truth[0])
            warmup_params.setdefault("y1gw", src_truth[1])
        lv = log_likelihood(warmup_params)
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def build_em_only_problem(ctx, cfg):
    """Backward-compatible alias for shared EM-only Nautilus builder."""
    return build_em_only_nautilus_problem(ctx, cfg)
