"""Phase 2a: stock gwemfish absolute TD + dL_eff inference on the geometry-ratios system.

Same truth as geometry_ratios_lenstronomy.py / tutorial_gw_only_pool.py.
Free: T_star, dL, lens0_q, lens0_gamma, y0gw, y1gw (geometry + cosmology).
Fixed: theta_E, phi, shear, centres.

Outputs samples for geometry_ratios_overlay.py (geometry-only rough match).

    uv run python examples/scripts/geometry_ratios_gwemfish_absolute.py
"""

import copy
import json
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from lenstronomy.Util.param_util import phi_q2_ellipticity

from gwemfish import (
    make_default_cfg,
    plot_source_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===================== SWITCHES =====================
RUN_DERIV_APPROX_SOURCE = True
RUN_NAUTILUS_SOURCE = True
NAUTILUS_RESUME = True
POOL = 9

# ===================== TRUTH (match geometry_ratios_lenstronomy) =====================
LENS_Q = 0.5
LENS_PHI_DEG = 30.0
Y0_TRUTH = 0.02
Y1_TRUTH = 0.00001
SIGMA_TD = 0.001
SIGMA_DL_EFF = 0.1

# Match geometry_ratios_lenstronomy_absolute BOUNDS
Y0_LO, Y0_HI = -0.001, 0.05
Y1_LO, Y1_HI = -0.008, 0.028

OUT_ROOT = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_gwemfish_absolute"
)
os.makedirs(OUT_ROOT, exist_ok=True)

# Per-method output dirs (sampler / method tag)
OUT_DERIV = os.path.join(OUT_ROOT, "deriv_approx_source")
OUT_NAUTILUS = os.path.join(OUT_ROOT, "nautilus_source")
NAUTILUS_CHECKPOINT = os.path.join(OUT_NAUTILUS, "nautilus_checkpoint.hdf5")


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(to_serializable(data), f, indent=2)


def save_samples(path, samples):
    np.savez(path, **{k: np.asarray(v) for k, v in samples.items()})


CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
CFG["lens_mass_parametrization"] = "q_phi"
CFG["lens"]["kwargs_lens"] = copy.deepcopy(CFG["lens"]["kwargs_lens"])
_e1, _e2 = phi_q2_ellipticity(np.deg2rad(LENS_PHI_DEG), LENS_Q)
CFG["lens"]["kwargs_lens"][0]["e1"] = float(_e1)
CFG["lens"]["kwargs_lens"][0]["e2"] = float(_e2)
CFG["gw"]["n_images"] = 4
CFG["gw"]["source_box_half_width"] = 0.4
CFG["gw"]["source_pos"] = (Y0_TRUTH, Y1_TRUTH)
CFG["gw"]["solver_params"]["backend"] = "helens"
# CFG["gw"]["solver_params"]["backend"] = "jaxtronomy"
# CFG["gw"]["solver_params"]["jaxtronomy"]["solver"] = "lenstronomy"  # "analytical"
CFG["gw"]["error_scales"]["sigma_td"] = SIGMA_TD
CFG["gw"]["error_scales"]["sigma_dL_eff"] = SIGMA_DL_EFF
CFG["em"]["kwargs_source"][0]["center_x"] = float(Y0_TRUTH)
CFG["em"]["kwargs_source"][0]["center_y"] = float(Y1_TRUTH)
CFG["inference"]["num_chains"] = 4
CFG["inference"]["num_samples"] = 2000
CFG["inference"]["num_warmup"] = 1000
CFG["inference"]["diagnostics"] = "warn"
# CFG["inference"]["diagnostics_thresholds"] = {
#     "inversion_residual": 1e-5,
#     "inversion_residual_physical": 1e3,
# }
CFG["nautilus"] = {
    "n_live": 5000,
    "n_eff": 5000,
    "n_like_max": 800_000,
    "filepath": NAUTILUS_CHECKPOINT,
    "resume": NAUTILUS_RESUME,
    "prior_check": True,
    "verbose": True,
    "pool": POOL,
    "jit": True,
    "equal_weight": False,  # full dead points + weights (lenstronomy-style)
}
CFG["output"]["output_dir"] = OUT_ROOT

if __name__ == "__main__":
    print("=" * 72)
    print("  geometry_ratios_gwemfish_absolute (TD + dL_eff, free T_star/dL)")
    print(
        f"  q={LENS_Q} phi={LENS_PHI_DEG} deg  source=({Y0_TRUTH}, {Y1_TRUTH})"
    )
    print(f"  out root: {OUT_ROOT}")
    print("=" * 72)

    ctx = setup_em_observation(cfg=CFG)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth_params = ctx["truth_params"]
    src = ctx["cfg"]["gw"]["source_pos"]
    print(f"Simulated {ctx['n_images']} GW images.")

    plot_system_observation(
        ctx,
        cfg={
            "output": {
                "output_dir": OUT_ROOT,
                "save_system_plot_path": "system_observation.png",
            }
        },
    )
    plt.close("all")

    priors = {
        "lens1_gamma1": float(truth_params["lens1_gamma1"]),
        "lens1_gamma2": float(truth_params["lens1_gamma2"]),
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        "lens0_theta_E": float(truth_params["lens0_theta_E"]),
        "lens0_center_x": float(truth_params["lens0_center_x"]),
        "lens0_center_y": float(truth_params["lens0_center_y"]),
        "lens0_phi": float(truth_params["lens0_phi"]),
        "T_star": dist.Uniform(4e6, 1.6e7),
        "dL": dist.Uniform(10000.0, 28000.0),
        "lens0_q": dist.Uniform(0.3, 0.75),
        "lens0_gamma": dist.Uniform(1.2, 2.8),
        "y0gw": dist.Uniform(Y0_LO, Y0_HI),
        "y1gw": dist.Uniform(Y1_LO, Y1_HI),
    }
    ctx["cfg"]["priors"] = dict(priors)
    ctx["cfg"]["gw"]["source_plane_bounds"] = {
        "y0gw": (Y0_LO, Y0_HI),
        "y1gw": (Y1_LO, Y1_HI),
    }

    save_json(
        {
            "note": (
                "Stock gwemfish absolute TD+dL_eff. Geometry marginals "
                "(lens0_q, lens0_gamma, y0gw, y1gw) are meant for rough "
                "comparison to geometry_ratios_lenstronomy (no T_star/dL)."
            ),
            "sigma_td": SIGMA_TD,
            "sigma_dL_eff": SIGMA_DL_EFF,
            "source_pos": list(src),
            "lens_q": LENS_Q,
            "lens_phi_deg": LENS_PHI_DEG,
        },
        os.path.join(OUT_ROOT, "run_info.json"),
    )

    if RUN_DERIV_APPROX_SOURCE:
        print("\n--- GW-only: deriv-approx-source ---\n")
        os.makedirs(OUT_DERIV, exist_ok=True)
        samples, truths = run_inference(
            ctx,
            mode="GW-only",
            method="deriv-approx-source",
            cfg={
                "priors": priors,
                "output": {
                    "output_dir": OUT_DERIV,
                    "json_tag": "deriv_approx_source",
                },
                "inference": {"informed": True},
            },
        )
        truths.setdefault("y0gw", float(src[0]))
        truths.setdefault("y1gw", float(src[1]))
        save_samples(os.path.join(OUT_DERIV, "samples.npz"), samples)
        plot_source_posterior(
            samples,
            truths=truths,
            cfg={
                "output": {"output_dir": OUT_DERIV},
                "plot": {
                    "plot_mode": "combined",
                    "save_path": "source_posterior_combined.png",
                },
            },
        )
        plt.close("all")
        print(f"  -> {OUT_DERIV}")

    if RUN_NAUTILUS_SOURCE:
        print("\n--- GW-only: nautilus-source ---\n")
        os.makedirs(OUT_NAUTILUS, exist_ok=True)
        samples, truths = run_inference(
            ctx,
            mode="GW-only",
            method="nautilus-source",
            cfg={
                "priors": priors,
                "output": {
                    "output_dir": OUT_NAUTILUS,
                    "json_tag": "nautilus_source",
                },
                "nautilus": {
                    "filepath": NAUTILUS_CHECKPOINT,
                    "resume": NAUTILUS_RESUME,
                    "pool": POOL,
                },
            },
        )
        truths.setdefault("y0gw", float(src[0]))
        truths.setdefault("y1gw", float(src[1]))
        n = min(
            len(np.asarray(v))
            for k, v in samples.items()
            if k != "weights"
        )
        if n < 100:
            raise RuntimeError(
                f"nautilus returned {n} samples — hit n_like_max during exploration"
            )
        save_samples(os.path.join(OUT_NAUTILUS, "samples.npz"), samples)
        plot_source_posterior(
            samples,
            truths=truths,
            cfg={
                "output": {"output_dir": OUT_NAUTILUS},
                "plot": {
                    "plot_mode": "combined",
                    "save_path": "source_posterior_combined.png",
                },
            },
        )
        plt.close("all")
        print(f"  -> {OUT_NAUTILUS}")

    print(f"\nDone -> {OUT_ROOT}")
