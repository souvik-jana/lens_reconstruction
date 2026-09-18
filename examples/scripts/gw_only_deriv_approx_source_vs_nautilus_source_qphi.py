"""
GW-only, q_phi parametrization, apples-to-apples: gwemfish deriv-approx-source vs
gwemfish nautilus-source -- same probmodel family, same priors, same fixed-to-truth
set (lens0_theta_E, lens0_center_x/y, lens0_phi), unlike the earlier comparison
against the external lenstronomy+nautilus (e1-based) run.

Same system as gwemfish_infer_reparam.py / lenstronomy_infer_reparam.py in
lensing-degeneracies/gw-only-analysis/geometry-analysis-reparam/: EPL only (no
shear), theta_E=1.0, gamma=2.0, q=0.6, phi=0.0, zl=0.7, zs=1.5, source at
(0.02, 0.001) ("centre"), n_images=4.

nautilus-source runs as a SMOKE TEST here (n_eff=500) to get a quick sense of the
posterior before committing to a real n_eff. deriv-approx-source uses more chains
and samples than the earlier smoke run since it is cheap (Fisher evaluated once at
u0, then Hessian-informed NUTS on the Taylor/banana model -- see gwemfish-infer
skill's cost table).
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi")

NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_source_qphi_smoke.hdf5")
NAUTILUS_RESUME = False  # first run at this n_eff / these priors

# ===================== TRUTH SYSTEM (same as gwemfish_infer_reparam.py) =====================
THETA_E = 1.0
GAMMA_TRUTH = 2.0
Q_TRUTH = 0.6
PHI_TRUTH = 0.0
ZL, ZS = 0.7, 1.5
SOURCE_POS = (0.02, 0.001)

# ===================== PRIOR BOUNDS (same box for both methods) =============================
GAMMA_BOUNDS = (0.8, 2.8)
E1_BOUNDS = (0.04, 0.48)                 # same nautilus-reparam e1 box, at fixed phi=0
Q_BOUNDS = (
    (1 - E1_BOUNDS[1]) / (1 + E1_BOUNDS[1]),
    (1 - E1_BOUNDS[0]) / (1 + E1_BOUNDS[0]),
)
TSTAR_BOUNDS = (2958497.8814427527, 29584978.814427525)
DL_BOUNDS = (5606.859711766309, 21193.92971047665)
# Same ASYMMETRIC box as lenstronomy_nautilus/cfg.json and lenstronomy_nautilus_qphi.py
# (a previous version of this script used a symmetric +/-0.02 truth-centered box,
# which does NOT match -- fixed per explicit user correction). Set directly via
# cfg['priors']['y0gw']/['y1gw'] for deriv-approx-source, since its default box
# (source_box_half_width) can only express a symmetric one; nautilus-source's
# source_plane_bounds already supports an asymmetric box natively.
Y0_BOUNDS = (0.00375, 0.045)
Y1_BOUNDS = (1e-5, 0.003)
FRAC_TD = 0.002
FRAC_DLEFF = 0.1

# --- deriv-approx-source: more chains/samples than the earlier smoke run (cheap method) ---
NUM_WARMUP = 5000
NUM_SAMPLES = 8000
NUM_CHAINS = 8

# --- nautilus-source: smoke run (n_eff=500) to get a quick sense before scaling up ---
NAUTILUS_N_LIVE = 500
NAUTILUS_N_EFF = 500

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
print(f"  q bounds (from e1 in {E1_BOUNDS} at phi=0): ({Q_BOUNDS[0]:.4f}, {Q_BOUNDS[1]:.4f})")
print(f"  y0gw bounds: {Y0_BOUNDS}   y1gw bounds: {Y1_BOUNDS}")

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
        "source_plane_bounds": {"y0gw": Y0_BOUNDS, "y1gw": Y1_BOUNDS},  # nautilus-source's box
        # helens triangle-search backend: jaxtronomy's "auto"/default path threw
        # repeated "invalid value encountered in multiply" warnings and produced a
        # badly weight-skewed nautilus-source posterior (67/500 effective samples);
        # jaxtronomy's "analytical" solver crashed outright (ValueError: Signs are
        # not different, in lenstronomy's brentq_nojit root refinement) on some
        # proposed point during nautilus's exploration of this near-axis "centre"
        # source. gwemfish_infer_reparam.py (lensing-degeneracies) uses helens for
        # this exact system successfully.
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
print(f"  truth_params lens0_q={tp['lens0_q']:.6f}  lens0_phi={tp['lens0_phi']:.6f}"
      f"  (from lens0_e1={tp['lens0_e1']:+.6f} lens0_e2={tp['lens0_e2']:+.6f})")

# Same priors dict for BOTH methods -- fix theta_E, center, and phi to truth; free
# lens0_gamma, lens0_q (q_phi mode), T_star, dL; y0gw/y1gw explicitly set to the
# same asymmetric box as lenstronomy_nautilus (overrides deriv-approx-source's
# default symmetric source_box_half_width box; nautilus-source's source_plane_bounds
# already applies the same box via cfg["gw"] above).
ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens0_phi":      float(tp["lens0_phi"]),
    "lens0_gamma":    dist.Uniform(*GAMMA_BOUNDS),
    "lens0_q":        dist.Uniform(*Q_BOUNDS),
    "T_star":         dist.Uniform(*TSTAR_BOUNDS),
    "dL":             dist.Uniform(*DL_BOUNDS),
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

# --------------------------------------------------------------------------
# deriv-approx-source
# --------------------------------------------------------------------------
print("\n--- GW-only inference: deriv-approx-source, q_phi (more chains/samples) ---\n")
deriv_dir = os.path.join(OUTPUT_DIR, "deriv_approx_source")
os.makedirs(deriv_dir, exist_ok=True)

samples_deriv, truths_deriv = run_inference(
    ctx,
    mode="GW-only",
    method="deriv-approx-source",
    cfg={
        "inference": {
            "informed": True,
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS,
        },
        "output": {
            "output_dir": deriv_dir,
            "json_path": "pipeline_outputs.json",
            "json_tag": "deriv-approx-source-qphi",
        },
    },
)
print("\nderiv-approx-source free parameters:", sorted(k for k in samples_deriv if k not in ("obs",)))

# Save BEFORE plotting -- a plotting failure must not lose an expensive run.
np.savez(os.path.join(deriv_dir, "samples.npz"), **{k: np.asarray(v) for k, v in samples_deriv.items()})
print(f"Saved: {os.path.join(deriv_dir, 'samples.npz')}")

# `corner` can't histogram a zero-variance column (e.g. if a fixed-truth key leaks
# through, or lens0_e1/e2 ever get backfilled with phi fixed) -- drop any such key
# defensively rather than crash after an expensive run.
_plot_keys = [k for k in samples_deriv if k != "obs" and np.std(np.asarray(samples_deriv[k])) > 0]
_dropped = sorted(set(samples_deriv) - set(_plot_keys) - {"obs"})
if _dropped:
    print(f"Dropping zero-variance column(s) from deriv-approx-source corner: {_dropped}")

plot_source_posterior(
    {k: samples_deriv[k] for k in _plot_keys}, truths=truths_source,
    cfg={
        "output": {"output_dir": deriv_dir},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

# --------------------------------------------------------------------------
# nautilus-source (smoke: n_eff=500)
# --------------------------------------------------------------------------
print(f"\n--- GW-only inference: nautilus-source, q_phi (smoke, n_eff={NAUTILUS_N_EFF}) ---\n")
nautilus_dir = os.path.join(OUTPUT_DIR, "nautilus_source")
os.makedirs(nautilus_dir, exist_ok=True)

samples_nautilus, truths_nautilus = run_inference(
    ctx,
    mode="GW-only",
    method="nautilus-source",
    cfg={
        "nautilus": {
            "n_live": NAUTILUS_N_LIVE,
            "n_eff": NAUTILUS_N_EFF,
            "filepath": NAUTILUS_CHECKPOINT,
            "resume": NAUTILUS_RESUME,
            "verbose": True,
        },
        "output": {
            "output_dir": nautilus_dir,
            "json_path": "pipeline_outputs.json",
            "json_tag": "nautilus-source-qphi-smoke",
        },
    },
)
n_nautilus = min(len(np.asarray(v)) for v in samples_nautilus.values())
print(f"  nautilus-source returned {n_nautilus} samples")
if n_nautilus < 100:
    # Kish n_eff (nautilus's stopping criterion) reached ~1500+ here (well past the
    # n_eff=500 target), but sampler.posterior(equal_weight=True)'s floor(w/w_max)
    # thinning is far stricter than Kish ESS for a likelihood this peaked relative
    # to the prior box -- accepted per user request as a smoke/shape-only check;
    # do not trust this count for anything quantitative (see gwemfish-infer skill,
    # Question 4, trap 2). Raise n_eff substantially (~10-20x) for a usable count.
    print(f"  WARNING: only {n_nautilus} equal-weight samples -- shape-only smoke "
          "result, not quantitatively reliable. Raise n_eff for a real run.")
print("nautilus-source free parameters:", sorted(k for k in samples_nautilus if k not in ("obs",)))

# Save BEFORE plotting -- a plotting failure must not lose an expensive run.
np.savez(os.path.join(nautilus_dir, "samples.npz"), **{k: np.asarray(v) for k, v in samples_nautilus.items()})
print(f"Saved: {os.path.join(nautilus_dir, 'samples.npz')}")

_plot_keys_n = [k for k in samples_nautilus if k != "obs" and np.std(np.asarray(samples_nautilus[k])) > 0]
_dropped_n = sorted(set(samples_nautilus) - set(_plot_keys_n) - {"obs"})
if _dropped_n:
    print(f"Dropping zero-variance column(s) from nautilus-source corner: {_dropped_n}")

plot_source_posterior(
    {k: samples_nautilus[k] for k in _plot_keys_n}, truths=truths_source,
    cfg={
        "output": {"output_dir": nautilus_dir},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

# --------------------------------------------------------------------------
# Apples-to-apples comparison: identical keys, no renaming needed (same probmodel
# family, same flat-name convention, same priors).
# --------------------------------------------------------------------------
shared_keys = sorted(k for k in samples_deriv if k in samples_nautilus)
print("\n" + "=" * 78)
print(f"{'param':<14}{'deriv-approx-source':>26}{'nautilus-source (smoke)':>30}")
print("-" * 78)
for k in shared_keys:
    d = np.asarray(samples_deriv[k])
    n = np.asarray(samples_nautilus[k])
    truth_str = f"  (truth={truths_source[k]:.6g})" if k in truths_source else ""
    print(f"{k:<14}{d.mean():>16.6g} +/- {d.std():<7.3g}"
          f"{n.mean():>18.6g} +/- {n.std():<9.3g}{truth_str}")
print("=" * 78)

plot_multi_comparison_corner(
    [samples_deriv, samples_nautilus],
    {"all": shared_keys},
    labels=["deriv-approx-source", "nautilus-source (smoke, n_eff=500)"],
    colors=["C0", "C1"],
    truths_dict={"all": {k: truths_source[k] for k in shared_keys if k in truths_source}},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_deriv_approx_vs_nautilus_qphi.png"),
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
print(f"Nautilus checkpoint: {NAUTILUS_CHECKPOINT} (raise n_eff and set resume=True to refine)")
