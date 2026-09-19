"""Tutorial configs and priors, lifted verbatim, shared by every prototype script.

Copied from ``tutorial/tutorial_gw_only.py``, ``tutorial_em_gw.py`` and
``tutorial_em_only.py`` so the benchmark measures the setup people actually run,
not a toy. The only change is the sampling budget: the tutorials use
``n_live=2000, n_eff=5000, n_like_max=500_000``, which takes hours per variant.
BUDGET below scales that down the same way ``quick_check_gw_only.py`` already
does, so its measured 740.1 s GW-only number is directly comparable.
"""

import copy

import numpyro.distributions as dist

from gwemfish import (
    make_default_cfg,
    prune_gw_images,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.config import DEFAULT_KWARGS_SOURCE

BUDGET = {"n_live": 200, "n_eff": 500, "n_like_max": 4000}


def base_cfg(mode):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1

    if mode == "GW-only":
        cfg["lens_mass_parametrization"] = "q_phi"
        cfg["gw"]["n_images"] = 4
        cfg["gw"]["source_pos"] = (0.02, 0.01)
        cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
        cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    else:
        cfg["gw"]["n_images"] = 2
        cfg["source_plane"]["n_images"] = 2
        cfg["gw"]["source_pos"] = (0.2, 0.01)

    if mode == "EM+GW":
        # Deviation from tutorial_em_gw.py, and it is required for the mode to
        # mean anything. The tutorial leaves the EM source at its default
        # (0.05, 0.1) while putting the GW source at (0.2, 0.01), but
        # build_em_gw_source_plane_problem uses the EM source centre AS the GW
        # source position (nautilus_source_inference.py:442-443) -- one physical
        # source seen two ways. With the two positions disagreeing, the
        # likelihood solves the lens equation at the EM position and compares
        # against time delays generated at the GW position. Measured at truth:
        # logL = -472283 (GW term -4.73e5) instead of +976, and truth is not even
        # the maximum -- random prior draws score 1.6e5 log-units better. That is
        # what made nautilus return n_eff = 1 after 500,000 calls.
        src = cfg["gw"]["source_pos"]
        kwargs_source = copy.deepcopy(DEFAULT_KWARGS_SOURCE)
        kwargs_source[0]["center_x"] = float(src[0])
        kwargs_source[0]["center_y"] = float(src[1])
        cfg["em"]["kwargs_source"] = kwargs_source

    cfg["nautilus"] = {**BUDGET, "resume": False, "prior_check": True, "verbose": False}
    return cfg


def build_ctx(mode):
    cfg = base_cfg(mode)
    ctx = setup_em_observation(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    if mode != "GW-only":
        ctx = prune_gw_images(ctx, n_keep=2)
    return ctx


def source_bounds(ctx, mode):
    """Prior box on the GW source position, exactly as each tutorial sets it."""
    if mode == "GW-only":
        return {"y0gw": (0.01992, 0.02005), "y1gw": (0.0091, 0.0106)}
    src = ctx["cfg"]["gw"]["source_pos"]
    hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
    return {"y0gw": (float(src[0]) - hw, float(src[0]) + hw),
            "y1gw": (float(src[1]) - hw, float(src[1]) + hw)}


def priors_for_mode(ctx, mode):
    truth = ctx["truth_params"]
    if mode == "EM-only":
        return {
            "lens1_ra_0": float(truth["lens1_ra_0"]),
            "lens1_dec_0": float(truth["lens1_dec_0"]),
        }

    bounds = source_bounds(ctx, mode)
    common = {
        "lens1_ra_0": float(truth["lens1_ra_0"]),
        "lens1_dec_0": float(truth["lens1_dec_0"]),
        "y0gw": dist.Uniform(*bounds["y0gw"]),
        "y1gw": dist.Uniform(*bounds["y1gw"]),
    }

    if mode == "EM+GW":
        return {**common,
                "dL": float(truth["dL"]),
                "T_star": float(truth["T_star"])}

    return {
        **common,
        "lens1_gamma1": float(truth["lens1_gamma1"]),
        "lens1_gamma2": float(truth["lens1_gamma2"]),
        "lens0_phi": float(truth["lens0_phi"]),
        "lens0_q": dist.Uniform(0.75, 0.85),
        "lens0_theta_E": float(truth["lens0_theta_E"]),
        "lens0_center_x": float(truth["lens0_center_x"]),
        "lens0_center_y": float(truth["lens0_center_y"]),
        "lens0_gamma": dist.Uniform(1.5, 2.4),
        "T_star": float(truth["T_star"]),
        "dL": float(truth["dL"]),
    }


def cfg_for_run(ctx, mode, **nautilus_overrides):
    """cfg sub-dict to hand a builder or run_inference: priors + source box + budget."""
    cfg = {"priors": priors_for_mode(ctx, mode),
           "nautilus": {**ctx["cfg"]["nautilus"], **nautilus_overrides}}
    if mode != "EM-only":
        cfg["gw"] = {"source_plane_bounds": source_bounds(ctx, mode)}
    return cfg
