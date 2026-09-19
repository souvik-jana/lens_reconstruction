"""Shared setup for the q/phi coverage matrix: one 4-image system, three modes.

Lifted from tutorial/tutorial_{gw_only,em_gw,em_only}.py, with two deliberate
changes so all three modes describe the *same* lens:

  - GW source at (0.02, 0.01) with n_images=4 everywhere. The EM+GW and EM-only
    tutorials use (0.2, 0.01) with n_images=2 and prune to 2; four images give
    3 time delays + 4 dL_eff = 7 GW observables instead of 3, which is what makes
    the GW-only B variant (T_star and dL free) determinable at all.
  - lens_mass_parametrization="q_phi" throughout -- the point of the exercise.

Lens model is the default ["EPL", "SHEAR"], so lens1_* are the shear components
(gamma1, gamma2) and the shear centre (ra_0, dec_0).
"""

import numpy as np
import numpyro.distributions as dist

from gwemfish import (
    make_default_cfg,
    setup_em_observation,
    setup_gw_observation,
)

SOURCE_POS = (0.02, 0.01)
N_IMAGES = 4

# Must stay under the caustic margin, which gwemfish measures at 0.1667 for this
# lens. The tutorials use 0.8, and gwemfish's own [diag] flags it: a source drawn
# outside the caustic produces a different image count and hits the image-count
# penalty, which NUTS reports as divergences. With 0.8, hmc-informed-source took
# 2133 s; the box is the cause, not the sampler.
SOURCE_BOX_HALF_WIDTH = 0.15

# Tiny on purpose. The matrix asks "does q/phi reach this sampler and convert
# correctly", not "has it converged".
BUDGETS = {
    "fisher": {"n_fisher_samples": 2000},
    # max_tree_depth 10 (the default) allows up to 2**10 leapfrog steps per
    # sample, each of which runs the lens-equation solver -- that is what made
    # hmc-informed-source take 2133 s. 5 caps it at 32. The posterior suffers,
    # which is fine: recovery is informational here, flow is what is asserted.
    "mcmc": {"num_chains": 2, "num_warmup": 100, "num_samples": 200,
             "max_tree_depth": 5},
    "nautilus": {"n_live": 100, "n_eff": 100, "n_like_max": 5000,
                 "resume": False, "prior_check": False, "verbose": False},
}


def base_cfg(parametrization="q_phi"):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = parametrization
    cfg["gw"]["n_images"] = N_IMAGES
    cfg["gw"]["source_pos"] = SOURCE_POS
    cfg["gw"]["source_box_half_width"] = SOURCE_BOX_HALF_WIDTH
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["inference"].update(BUDGETS["mcmc"])
    cfg["inference"]["n_fisher_samples"] = BUDGETS["fisher"]["n_fisher_samples"]
    cfg["nautilus"] = dict(BUDGETS["nautilus"])
    return cfg


def build_ctx(parametrization="q_phi"):
    """One system, shared by every mode. No pruning -- we want all four images."""
    cfg = base_cfg(parametrization)
    ctx = setup_em_observation(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    return ctx


def check_four_images(ctx):
    """Fail loudly rather than silently analysing a 2-image system.

    The source position has to sit inside the caustic to produce four images;
    if it does not the solver returns two and every count below is wrong.
    """
    truth = ctx["truth_params"]
    n_found = len([k for k in truth if k.startswith("image_x")])
    x = [float(truth[f"image_x{i+1}"]) for i in range(n_found)]
    y = [float(truth[f"image_y{i+1}"]) for i in range(n_found)]

    print(f"GW source {SOURCE_POS} -> {n_found} images (ctx['n_images']={ctx['n_images']})")
    for i, (xi, yi) in enumerate(zip(x, y), 1):
        print(f"    image {i}: ({xi:+.6f}, {yi:+.6f})")
    print(f"  time delays available : {max(0, n_found - 1)}")
    print(f"  dL_eff available      : {n_found}")
    print(f"  GW observables total  : {max(0, n_found - 1) + n_found}")

    if n_found != N_IMAGES or ctx["n_images"] != N_IMAGES:
        raise SystemExit(
            f"\nSTOP: expected {N_IMAGES} images, got {n_found}. The source at "
            f"{SOURCE_POS} is not inside the caustic for this lens, so the "
            f"observable counting the free-parameter choices rely on does not hold."
        )
    return n_found


# ---- free/fixed layouts -------------------------------------------------
# Every GW-only variant also fixes these; only the T_star/dL vs gamma choice differs.
GW_ONLY_ALWAYS_FIXED = [
    "lens0_phi", "lens0_center_x", "lens0_center_y", "lens0_theta_E",
    "lens1_gamma1", "lens1_gamma2", "lens1_ra_0", "lens1_dec_0",
]
SHEAR_CENTRE = ["lens1_ra_0", "lens1_dec_0"]


def priors_for(ctx, layout):
    """layout: 'EM-only' | 'GW-only-A' | 'GW-only-B' | 'EM+GW'."""
    truth = ctx["truth_params"]
    fix = lambda keys: {k: float(truth[k]) for k in keys}

    src = ctx["cfg"]["gw"]["source_pos"]
    hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
    src_box = {
        "y0gw": dist.Uniform(float(src[0]) - hw, float(src[0]) + hw),
        "y1gw": dist.Uniform(float(src[1]) - hw, float(src[1]) + hw),
    }

    if layout == "EM-only":
        # phi is FREE here -- EM data constrains orientation, and this is the
        # first time the phi sampler has ever run.
        return fix(SHEAR_CENTRE)

    if layout == "GW-only-A":
        # distances pinned, slope free
        return {**fix(GW_ONLY_ALWAYS_FIXED + ["T_star", "dL"]), **src_box}

    if layout == "GW-only-B":
        # slope pinned, distances free. T_star enters the time delay only as
        # T_star * delta-phi(lens), so it trades against lens0_gamma; pinning
        # gamma is what makes T_star recoverable.
        return {**fix(GW_ONLY_ALWAYS_FIXED + ["lens0_gamma"]), **src_box}

    if layout == "EM+GW":
        # only the shear centre pinned; phi, T_star and dL all free
        return {**fix(SHEAR_CENTRE), **src_box}

    raise ValueError(f"unknown layout {layout!r}")


def source_plane_bounds(ctx):
    src = ctx["cfg"]["gw"]["source_pos"]
    hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
    return {"y0gw": (float(src[0]) - hw, float(src[0]) + hw),
            "y1gw": (float(src[1]) - hw, float(src[1]) + hw)}
