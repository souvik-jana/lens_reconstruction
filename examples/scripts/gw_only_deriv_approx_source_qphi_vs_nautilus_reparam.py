"""
GW-only, deriv-approx-source, lens_mass_parametrization="q_phi", compared against
the existing lenstronomy+nautilus run in lensing-degeneracies/gw-only-analysis/
geometry-analysis-reparam/runs/centre/full/lenstronomy_nautilus/ (NOT re-run here --
its samples.npz is loaded directly).

Same system as gwemfish_infer_reparam.py / lenstronomy_infer_reparam.py in that repo:
EPL only (no shear), theta_E=1.0, gamma=2.0, q=0.6, phi=0.0, zl=0.7, zs=1.5, source
at (0.02, 0.001) ("centre"), n_images=4.

Prior translation (nautilus's box -> our q_phi box):
  - nautilus fixed e2=0.0 and freed e1 ~ Uniform(0.04, 0.48) at that fixed e2. Since
    e2 = (1-q)/(1+q)*sin(2*phi), e2=0 for ALL q iff phi=0 (or pi/2) -- so "e2 fixed
    at 0, e1 free" is exactly "phi fixed at 0 (truth), q free" expressed through
    e1/e2 instead of q/phi. We fix lens0_phi=0.0 and free lens0_q over the box that
    e1 in [0.04, 0.48] maps to at phi=0 (e1=(1-q)/(1+q) there, so q=(1-e1)/(1+e1)):
    q in [(1-0.48)/(1+0.48), (1-0.04)/(1+0.04)] = [0.3514, 0.9231].
  - lens0_gamma, T_star, dL, y0gw, y1gw bounds copied verbatim from that run's
    cfg.json / gwemfish_infer_reparam.py.

Comparison uses lens0_e1 (numpyro.deterministic from our sampled lens0_q at fixed
phi=0) against nautilus's directly-sampled e1 -- same physical quantity, sampled
two different ways.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_qphi_vs_nautilus_reparam")

REPARAM_RUN_DIR = (
    "/Users/souvikjana/Documents/lensing-degeneracies/gw-only-analysis/"
    "geometry-analysis-reparam/runs/centre/full/lenstronomy_nautilus"
)
NAUTILUS_SAMPLES_NPZ = os.path.join(REPARAM_RUN_DIR, "samples.npz")

# ===================== TRUTH SYSTEM (same as gwemfish_infer_reparam.py) =====================
THETA_E = 1.0
GAMMA_TRUTH = 2.0
Q_TRUTH = 0.6
PHI_TRUTH = 0.0
ZL, ZS = 0.7, 1.5
SOURCE_POS = (0.02, 0.001)

# ===================== PRIOR BOUNDS (copied from lenstronomy_nautilus/cfg.json) =============
GAMMA_BOUNDS = (0.8, 2.8)
E1_BOUNDS = (0.04, 0.48)                 # nautilus's free param, at fixed e2=0 (phi=0)
Q_BOUNDS = (
    (1 - E1_BOUNDS[1]) / (1 + E1_BOUNDS[1]),   # e1 upper -> q lower
    (1 - E1_BOUNDS[0]) / (1 + E1_BOUNDS[0]),   # e1 lower -> q upper
)
TSTAR_BOUNDS = (2958497.8814427527, 29584978.814427525)
DL_BOUNDS = (5606.859711766309, 21193.92971047665)
Y0_BOUNDS = (0.00375, 0.045)
Y1_BOUNDS = (1e-5, 0.003)
FRAC_TD = 0.002
FRAC_DLEFF = 0.1

# --- SMOKE-TEST SETTINGS --- (raise for a publication-quality run)
NUM_WARMUP = 2000
NUM_SAMPLES = 4000
NUM_CHAINS = 4

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CHAINS}"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

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
from gwemfish.corner_plot_utils import plot_multi_comparison_corner

os.makedirs(OUTPUT_DIR, exist_ok=True)

E1_TRUTH, E2_TRUTH = phi_q2_ellipticity(PHI_TRUTH, Q_TRUTH)
print(f"  EPL truth: theta_E={THETA_E} gamma={GAMMA_TRUTH} q={Q_TRUTH} phi={PHI_TRUTH}"
      f" -> e1={float(E1_TRUTH):+.6f} e2={float(E2_TRUTH):+.6f}")
print(f"  q bounds derived from nautilus e1 bounds {E1_BOUNDS} at phi=0: "
      f"({Q_BOUNDS[0]:.4f}, {Q_BOUNDS[1]:.4f})")

BASE_CFG = {
    "em": {"enabled": False},
    "use_parameter_layout": True,   # EPL-only (no shear component) -> lens0_* names
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
        "source_box_half_width": 0.05,
        "error_scales": {
            "sigma_td": FRAC_TD,
            "sigma_td_floor": 1e-8,
            "sigma_dL_eff": FRAC_DLEFF,
            "epsilon": 1e-4,
        },
    },
    "inference": {
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
    },
}

ctx = setup_gw_observation({}, cfg=BASE_CFG)
if len(ctx["x_img_gw"]) > 4:
    ctx = prune_gw_images(ctx, n_keep=4)
tp = ctx["truth_params"]
print(f"  n images: {len(ctx['x_img_gw'])}")
print(f"  truth_params lens0_q={tp['lens0_q']:.6f}  lens0_phi={tp['lens0_phi']:.6f}"
      f"  (from lens0_e1={tp['lens0_e1']:+.6f} lens0_e2={tp['lens0_e2']:+.6f})")

ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens0_phi":      float(tp["lens0_phi"]),  # fixed=0.0, matches nautilus's fixed e2=0
    "lens0_gamma":    dist.Uniform(*GAMMA_BOUNDS),
    "lens0_q":        dist.Uniform(*Q_BOUNDS),
    "T_star":         dist.Uniform(*TSTAR_BOUNDS),
    "dL":             dist.Uniform(*DL_BOUNDS),
}
ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": Y0_BOUNDS, "y1gw": Y1_BOUNDS}

gw_src = ctx["cfg"]["gw"]["source_pos"]
truths_source = {
    k: float(tp[k]) for k in tp
    if not (k.startswith("image_x") or k.startswith("image_y"))
}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

print("\n--- GW-only inference: deriv-approx-source, lens_mass_parametrization=q_phi ---\n")

samples, truths = run_inference(
    ctx,
    mode="GW-only",
    method="deriv-approx-source",
    cfg={
        "inference": {"informed": True},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": "pipeline_outputs.json",
            "json_tag": "deriv-approx-source-qphi",
        },
        "gw": {"source_plane_bounds": {"y0gw": Y0_BOUNDS, "y1gw": Y1_BOUNDS}},
    },
)
print("\nFree parameters sampled:", sorted(k for k in samples if k not in ("obs",)))

plot_source_posterior(
    samples,
    truths=truths_source,
    cfg={
        "output": {"output_dir": OUTPUT_DIR},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

# --------------------------------------------------------------------------
# Load the EXISTING lenstronomy+nautilus run (not re-run) and compare on the
# shared physical parameters: gamma, e1, T_star, dL, y0gw, y1gw.
# --------------------------------------------------------------------------
print(f"\n--- Loading existing nautilus run (no rerun): {NAUTILUS_SAMPLES_NPZ} ---")
nautilus_raw = np.load(NAUTILUS_SAMPLES_NPZ)
n_nautilus = min(len(nautilus_raw[k]) for k in nautilus_raw.files)
print(f"  {n_nautilus} weighted nested-sampling points loaded")

# Weighted nested-sampling points -> equal-weight resample (same convention as
# geometry_infer_epl.py's multi-method comparison section).
w = np.asarray(nautilus_raw["weights"])
rng = np.random.default_rng(0)
idx = rng.choice(len(w), size=len(w), p=w / w.sum())
nautilus_eqw = {k: np.asarray(nautilus_raw[k])[idx] for k in nautilus_raw.files if k != "weights"}

# lens0_phi is fixed (not sampled), so add_qphi_columns_to_samples (which needs
# BOTH q and phi live) never backfills lens0_e1/e2 here -- compute e1 from the
# sampled lens0_q at the known fixed phi=0.0 truth directly.
e1_from_q, _ = phi_q2_ellipticity(float(tp["lens0_phi"]), np.asarray(samples["lens0_q"]))
ours_renamed = {
    "gamma": np.asarray(samples["lens0_gamma"]),
    "e1": np.asarray(e1_from_q),
    "T_star": np.asarray(samples["T_star"]),
    "dL": np.asarray(samples["dL"]),
    "y0gw": np.asarray(samples["y0gw"]),
    "y1gw": np.asarray(samples["y1gw"]),
}
shared_keys = sorted(k for k in ours_renamed if k in nautilus_eqw)
truths_compare = {"gamma": GAMMA_TRUTH, "e1": float(E1_TRUTH), "T_star": float(tp["T_star"]),
                   "dL": float(tp["dL"]), "y0gw": float(gw_src[0]), "y1gw": float(gw_src[1])}

print("\n" + "=" * 78)
print(f"{'param':<10}{'deriv-approx-source (q_phi)':>30}{'nautilus (reparam, e1)':>30}")
print("-" * 78)
for k in shared_keys:
    d, n = ours_renamed[k], nautilus_eqw[k]
    print(f"{k:<10}{d.mean():>18.6g} +/- {d.std():<9.3g}"
          f"{n.mean():>18.6g} +/- {n.std():<9.3g}"
          f"  (truth={truths_compare[k]:.6g})")
print("=" * 78)

plot_multi_comparison_corner(
    [ours_renamed, nautilus_eqw],
    {"all": shared_keys},
    labels=["deriv-approx-source (q_phi)", "nautilus (reparam, e1-based)"],
    colors=["C0", "C1"],
    truths_dict={"all": truths_compare},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_qphi_vs_nautilus_reparam.png"),
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
