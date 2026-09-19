"""
GW-only, q_phi parametrization, fisher-source: Taylor-Gaussian approximation
N(u0, inv(-H0)) around truth -- cheapest method (Hessian evaluated once at u0, no
NUTS, no nested sampling). Same system/priors as
gw_only_deriv_approx_source_vs_nautilus_source_qphi.py / gw_only_hmc_informed_source_qphi.py,
for the 4-way comparison.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi")
FISHER_DIR = os.path.join(OUTPUT_DIR, "fisher_source")

# ===================== TRUTH SYSTEM (same as the other q_phi comparison scripts) =============
THETA_E = 1.0
GAMMA_TRUTH = 2.0
Q_TRUTH = 0.6
PHI_TRUTH = 0.0
ZL, ZS = 0.7, 1.5
SOURCE_POS = (0.02, 0.001)

GAMMA_BOUNDS = (0.8, 2.8)
E1_BOUNDS = (0.04, 0.48)
Q_BOUNDS = (
    (1 - E1_BOUNDS[1]) / (1 + E1_BOUNDS[1]),
    (1 - E1_BOUNDS[0]) / (1 + E1_BOUNDS[0]),
)
TSTAR_BOUNDS = (2958497.8814427527, 29584978.814427525)
DL_BOUNDS = (5606.859711766309, 21193.92971047665)
# Same ASYMMETRIC box as lenstronomy_nautilus/cfg.json and lenstronomy_nautilus_qphi.py
# (NOT a symmetric truth-centered box -- source_box_half_width only supports that).
Y0_BOUNDS = (0.00375, 0.045)
Y1_BOUNDS = (1e-5, 0.003)
FRAC_TD = 0.002
FRAC_DLEFF = 0.1

N_FISHER_SAMPLES = 20000  # cheap (plain Gaussian draw), no reason to keep this small

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots
from herculens.Util.param_util import phi_q2_ellipticity

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import run_inference, setup_gw_observation, prune_gw_images, plot_source_posterior

os.makedirs(FISHER_DIR, exist_ok=True)

E1_TRUTH, E2_TRUTH = phi_q2_ellipticity(PHI_TRUTH, Q_TRUTH)

BASE_CFG = {
    "em": {"enabled": False},
    "use_parameter_layout": True,
    "lens_mass_parametrization": "q_phi",
    "lens": {
        "lens_model_list": ["EPL"],
        "kwargs_lens": [
            {
                "theta_E": THETA_E,
                "e1": float(E1_TRUTH),
                "e2": float(E2_TRUTH),
                "gamma": GAMMA_TRUTH,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
        "zl": ZL,
        "zs": ZS,
    },
    "gw": {
        "source_pos": SOURCE_POS,
        "n_images": 4,
        "source_plane_bounds": {"y0gw": Y0_BOUNDS, "y1gw": Y1_BOUNDS},
        "solver_params": {"backend": "helens"},
        "error_scales": {
            "sigma_td": FRAC_TD,
            "sigma_td_floor": 1e-8,
            "sigma_dL_eff": FRAC_DLEFF,
            "epsilon": 1e-4,
        },
    },
}

ctx = setup_gw_observation({}, cfg=BASE_CFG)
if len(ctx["x_img_gw"]) > 4:
    ctx = prune_gw_images(ctx, n_keep=4)
tp = ctx["truth_params"]
print(f"  n images: {len(ctx['x_img_gw'])}")

ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens0_phi":      float(tp["lens0_phi"]),
    "lens0_gamma":    dist.Uniform(*GAMMA_BOUNDS),
    "lens0_q":        dist.Uniform(*Q_BOUNDS),
    "T_star":         dist.Uniform(*TSTAR_BOUNDS),
    "dL":             dist.Uniform(*DL_BOUNDS),
    # Explicit override: the source-plane model's default y0gw/y1gw prior is a
    # SYMMETRIC truth-centered box (source_box_half_width), which cannot express
    # the asymmetric box lenstronomy_nautilus/lenstronomy_nautilus_qphi.py use.
    # Setting these directly wins over that default (see
    # _build_inference_probmodel_source_plane's priors_combined merge order).
    "y0gw":           dist.Uniform(*Y0_BOUNDS),
    "y1gw":           dist.Uniform(*Y1_BOUNDS),
}

gw_src = ctx["cfg"]["gw"]["source_pos"]
truths_source = {
    k: float(tp[k]) for k in tp
    if not (k.startswith("image_x") or k.startswith("image_y"))
}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

print("\n--- GW-only inference: fisher-source, q_phi ---\n")

samples, truths = run_inference(
    ctx,
    mode="GW-only",
    method="fisher-source",
    cfg={
        "inference": {"n_fisher_samples": N_FISHER_SAMPLES},
        "output": {
            "output_dir": FISHER_DIR,
            "json_path": "pipeline_outputs.json",
            "json_tag": "fisher-source-qphi",
        },
    },
)
print("Free parameters sampled:", sorted(k for k in samples if k not in ("obs",)))

np.savez(os.path.join(FISHER_DIR, "samples.npz"), **{k: np.asarray(v) for k, v in samples.items()})
print(f"Saved: {os.path.join(FISHER_DIR, 'samples.npz')}")

plot_keys = [k for k in samples if k != "obs" and np.std(np.asarray(samples[k])) > 0]
dropped = sorted(set(samples) - set(plot_keys) - {"obs"})
if dropped:
    print(f"Dropping zero-variance column(s) from the corner plot: {dropped}")

plot_source_posterior(
    {k: samples[k] for k in plot_keys}, truths=truths_source,
    cfg={
        "output": {"output_dir": FISHER_DIR},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

print(f"\nDone. Corner: {os.path.join(FISHER_DIR, 'corner_all_params.png')}")
