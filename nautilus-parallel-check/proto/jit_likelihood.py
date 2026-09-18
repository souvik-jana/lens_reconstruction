"""Prototype nautilus-source problem builders, all three modes.

Each builder mirrors its gwemfish counterpart one-for-one -- same priors, same
fixed-parameter handling, same q/phi reparametrisation, same solver, same
validation -- and differs only in that the per-call arithmetic goes through the
compiled cores in ``jit_gw_core`` instead of being executed eagerly.

Everything that runs once at build time is *imported* from gwemfish rather than
copied, so the prior construction and parameter layout cannot drift from the real
ones. Only the per-call bodies are re-implemented.

Set ``jit=False`` and each builder delegates straight to the real gwemfish
builder, which is what the benchmark's baseline variant uses.
"""

import warnings

from gwemfish.config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE
from gwemfish.ellipticity_reparam import (
    expand_qphi_params,
    mass_qphi_prefixes_from_entries,
    swap_ellipticity_for_qphi,
    validate_parametrization,
)
from gwemfish.nautilus_common import (
    EM_EXTRA_DEFAULT_DISTS,
    build_em_only_nautilus_problem,
    build_nautilus_prior,
    layout_defaults_from_registry,
    parse_cfg_priors,
)
from gwemfish.nautilus_source_inference import (
    _GW_DEFAULT_DISTS,
    _gw_extra_defaults,
    build_em_gw_source_plane_problem,
    build_gw_source_plane_problem,
    build_nautilus_solver,
    validate_helens_solver,
)
from gwemfish.parameter_layout import (
    build_mass_parameter_entries,
    build_parameter_layout,
    build_priors_registry,
    unpack_to_kwargs,
)
from gwemfish.priors import DEFAULT_PRIORS_GW_SOURCE_PLANE
from gwemfish.simple_pipeline import _deep_merge_dict, make_default_cfg

from .jit_gw_core import check_jittable, em_loglike_fn, gw_loglike_fn, solve_fn

REJECT = -1e300


def common_setup(ctx, cfg):
    """Everything the two GW builders share up to the point they diverge."""
    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    check_jittable(cfg_full)

    gw_cfg = cfg_full.get("gw", {})
    nautilus_cfg = cfg_full.get("nautilus", {})
    truth_params = ctx.get("truth_params", {})
    n_images = len([k for k in truth_params if k.startswith("image_x")])

    solver, solver_params, _ = build_nautilus_solver(ctx, cfg_full, n_images)

    y_truth = list(gw_cfg.get("source_pos", [truth_params.get("y0gw", 0.05),
                                             truth_params.get("y1gw", 1e-6)]))
    validate_helens_solver(
        solver, solver_params, ctx["kwargs_lens"], y_truth,
        [float(truth_params[f"image_x{i + 1}"]) for i in range(n_images)],
        [float(truth_params[f"image_y{i + 1}"]) for i in range(n_images)],
        tol=nautilus_cfg.get("solver_validation_tol", 0.05),
        lens_gw=ctx["lens_gw"],
    )

    return {
        "cfg_full": cfg_full,
        "gw_cfg": gw_cfg,
        "truth_params": truth_params,
        "n_images": n_images,
        "bounds": {**DEFAULT_PRIORS_GW_SOURCE_PLANE,
                   **gw_cfg.get("source_plane_bounds", {})},
        "solve": solve_fn(solver, solver_params, ctx["lens_gw"], n_images),
        "gw_loglike": gw_loglike_fn(ctx["lens_gw"], ctx["gw_obs"],
                                    gw_cfg.get("error_scales", {})),
    }


def finalize_prior(cfg_full, default_dists, bounds, registry_fixed, qphi_prefixes):
    if qphi_prefixes:
        swap_ellipticity_for_qphi(default_dists, qphi_prefixes)
    scipy_overrides, cfg_fixed = parse_cfg_priors(
        cfg_full.get("priors", {}), default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)
    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")
    return prior, fixed_params


def warm_up(log_likelihood, params, label):
    print(f"Warming up {label} log_likelihood (compiles once, then reuses)...")
    try:
        print(f"  warm-up log_likelihood = {log_likelihood(params):.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")


def build_gw_only(ctx, cfg, jit=True):
    if not jit:
        return build_gw_source_plane_problem(ctx, cfg)

    s = common_setup(ctx, cfg)
    cfg_full, n_images = s["cfg_full"], s["n_images"]

    if not bool(cfg_full.get("use_parameter_layout")):
        raise NotImplementedError(
            "prototype supports use_parameter_layout=True only; the legacy "
            "lens_theta_E/lens_e1 naming is not reimplemented here."
        )

    mass_model = ctx["lens_mass_model"]
    entries = build_mass_parameter_entries(mass_model, kwargs_lens=ctx["kwargs_lens"])
    registry = build_priors_registry(entries, mass_model=mass_model, user_priors=None)
    default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
    default_dists.update(_gw_extra_defaults(s["bounds"], ("T_star", "dL", "y0gw", "y1gw")))
    n_mass = len(mass_model.func_list)
    parametrization = validate_parametrization(
        cfg_full.get("lens_mass_parametrization", "e1e2"))
    qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)

    prior, fixed_params = finalize_prior(
        cfg_full, default_dists, s["bounds"], registry_fixed, qphi_prefixes)
    solve, gw_loglike = s["solve"], s["gw_loglike"]

    def log_likelihood(params):
        full = expand_qphi_params({**fixed_params, **params}, qphi_prefixes)
        kwargs_lens, _, _ = unpack_to_kwargs(full, entries, n_mass=n_mass,
                                             n_source=0, n_lens_light=0)
        x_pos, y_pos, n_distinct = solve(
            full["y0gw"], full["y1gw"], kwargs_lens,
            kwargs_lens[0].get("center_x", 0.0), kwargs_lens[0].get("center_y", 0.0))
        if int(n_distinct) != n_images:
            return REJECT
        return float(gw_loglike(x_pos, y_pos, kwargs_lens,
                                full["T_star"], full["dL"]))

    warm_up(log_likelihood, warmup_params(ctx, s), "GW-only source (jit)")
    return prior, log_likelihood, list(prior.keys)


def build_em_gw(ctx, cfg, jit=True):
    if not jit:
        return build_em_gw_source_plane_problem(ctx, cfg)

    s = common_setup(ctx, cfg)
    cfg_full, n_images = s["cfg_full"], s["n_images"]

    if not bool(cfg_full.get("use_parameter_layout")):
        raise NotImplementedError(
            "prototype supports use_parameter_layout=True only; the legacy "
            "source_amp/light_amp naming is not reimplemented here."
        )

    lens_image = ctx["lens_image"]
    em_sec = cfg_full.get("em") or {}
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE,
        kwargs_lens_light=em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT,
    )
    registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
    default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
    default_dists.update({
        "T_star": _GW_DEFAULT_DISTS["T_star"](s["bounds"]["T_star"]),
        "dL": _GW_DEFAULT_DISTS["dL"](s["bounds"]["dL"]),
        "noise_sigma_bkg": EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None),
    })
    n_mass = len(lens_image.MassModel.func_list)
    n_source = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)
    parametrization = validate_parametrization(
        cfg_full.get("lens_mass_parametrization", "e1e2"))
    qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)

    prior, fixed_params = finalize_prior(
        cfg_full, default_dists, s["bounds"], registry_fixed, qphi_prefixes)
    solve, gw_loglike = s["solve"], s["gw_loglike"]
    em_loglike = em_loglike_fn(lens_image, ctx["noise_inf"], ctx["em_obs"]["data"])

    def log_likelihood(params):
        full = expand_qphi_params({**fixed_params, **params}, qphi_prefixes)
        kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
            full, entries, n_mass=n_mass,
            n_source=n_source, n_lens_light=n_lens_light,
        )
        x_pos, y_pos, n_distinct = solve(
            kwargs_source[0]["center_x"], kwargs_source[0]["center_y"], kwargs_lens,
            kwargs_lens[0].get("center_x", 0.0), kwargs_lens[0].get("center_y", 0.0))
        if int(n_distinct) != n_images:
            return REJECT
        loglike_gw = gw_loglike(x_pos, y_pos, kwargs_lens,
                                full["T_star"], full["dL"])
        loglike_em = em_loglike(kwargs_lens, kwargs_source, kwargs_lens_light,
                                full["noise_sigma_bkg"])
        return float(loglike_gw + loglike_em)

    warm_up(log_likelihood, dict(s["truth_params"]), "EM+GW source (jit)")
    return prior, log_likelihood, list(prior.keys)


def build_em_only(ctx, cfg, jit=True):
    """EM-only: no lens equation solved, so only the pixel likelihood is jitted.

    ``lens_image.model`` and ``noise.C_D_model`` are already compiled inside
    herculens, which is why this mode starts at 5.3 ms/call rather than 110.
    """
    prior, real_loglike, param_names = build_em_only_nautilus_problem(ctx, cfg)
    if not jit:
        return prior, real_loglike, param_names

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    check_jittable(cfg_full)

    lens_image = ctx["lens_image"]
    em_sec = cfg_full.get("em") or {}
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE,
        kwargs_lens_light=em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT,
    )
    registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
    default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
    default_dists["noise_sigma_bkg"] = EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None)

    n_mass = len(lens_image.MassModel.func_list)
    n_source = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)
    parametrization = validate_parametrization(
        cfg_full.get("lens_mass_parametrization", "e1e2"))
    qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)

    _, fixed_params = finalize_prior(
        cfg_full, default_dists, DEFAULT_PRIORS_GW_SOURCE_PLANE,
        registry_fixed, qphi_prefixes)
    em_loglike = em_loglike_fn(lens_image, ctx["noise_inf"], ctx["em_obs"]["data"])

    def log_likelihood(params):
        full = expand_qphi_params({**fixed_params, **params}, qphi_prefixes)
        kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
            full, entries, n_mass=n_mass,
            n_source=n_source, n_lens_light=n_lens_light,
        )
        return float(em_loglike(kwargs_lens, kwargs_source, kwargs_lens_light,
                                full["noise_sigma_bkg"]))

    warm_up(log_likelihood, dict(ctx.get("truth_params", {})), "EM-only (jit)")
    return prior, log_likelihood, param_names


def warmup_params(ctx, s):
    src = list(s["gw_cfg"].get("source_pos", [0.05, 1e-6]))
    truth = s["truth_params"]
    return {**truth,
            "y0gw": truth.get("y0gw", src[0]),
            "y1gw": truth.get("y1gw", src[1])}


BUILDERS = {
    "GW-only": build_gw_only,
    "EM+GW": build_em_gw,
    "EM-only": build_em_only,
}


def build_problem(ctx, cfg, mode, jit=True):
    return BUILDERS[mode](ctx, cfg, jit=jit)
