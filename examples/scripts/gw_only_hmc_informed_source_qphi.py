"""
GW-only, q_phi parametrization, hmc-informed-source (full likelihood, Hessian-
informed NUTS, solves the lens equation inside the model on every leapfrog step).

Same system/priors as gw_only_deriv_approx_source_vs_nautilus_source_qphi.py, for
a genuine apples-to-apples 3-way comparison (deriv-approx-source, nautilus-source,
hmc-informed-source) once all three have run. This is the most expensive of the
three per the gwemfish-infer skill's cost table -- the solver is evaluated at
every leapfrog step (up to ~1024 per sample at max_tree_depth=10), unlike
deriv-approx-source (u0 only) or nautilus-source (once per likelihood call, no
gradients). Uses SMOKE settings (small warmup/samples/chains) to get a timing
sense first -- do not scale up until this completes and the per-sample cost is
known.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi")
HMC_DIR = os.path.join(OUTPUT_DIR, "hmc_informed_source")

# ===================== TRUTH SYSTEM (same as the deriv-approx/nautilus comparison) ===========
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
# Same ASYMMETRIC box as lenstronomy_nautilus/lenstronomy_nautilus_qphi.py, set
# explicitly via cfg['priors'] below since source_box_half_width can only express
# a symmetric truth-centered box.
TSTAR_BOUNDS = (2958497.8814427527, 29584978.814427525)
DL_BOUNDS = (5606.859711766309, 21193.92971047665)
Y0_BOUNDS = (0.00375, 0.045)
Y1_BOUNDS = (1e-5, 0.003)
FRAC_TD = 0.002
FRAC_DLEFF = 0.1

# --- SMOKE SETTINGS: this method is far more expensive per sample than
# deriv-approx-source (solver on every leapfrog step, not once at u0) -- start
# small to get a timing sense before committing to a bigger run.
NUM_WARMUP = 200
NUM_SAMPLES = 200
NUM_CHAINS = 2

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CHAINS}"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import time

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

os.makedirs(HMC_DIR, exist_ok=True)

E1_TRUTH, E2_TRUTH = phi_q2_ellipticity(PHI_TRUTH, Q_TRUTH)
print(f"  EPL truth: theta_E={THETA_E} gamma={GAMMA_TRUTH} q={Q_TRUTH} phi={PHI_TRUTH}"
      f" -> e1={float(E1_TRUTH):+.6f} e2={float(E2_TRUTH):+.6f}")

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
        # Matches the deriv-approx-source / nautilus-source comparison's choice --
        # "auto"/jaxtronomy threw warnings and jaxtronomy's analytical solver
        # crashed on this near-axis "centre" source; helens ran cleanly for both.
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

print(f"\n--- GW-only inference: hmc-informed-source, q_phi "
      f"(SMOKE: {NUM_WARMUP} warmup / {NUM_SAMPLES} samples / {NUM_CHAINS} chains) ---\n")

t0 = time.time()
samples, truths = run_inference(
    ctx,
    mode="GW-only",
    method="hmc-informed-source",
    cfg={
        "inference": {
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS,
        },
        "output": {
            "output_dir": HMC_DIR,
            "json_path": "pipeline_outputs.json",
            "json_tag": "hmc-informed-source-qphi",
        },
    },
)
elapsed = time.time() - t0
n_total_samples = NUM_SAMPLES * NUM_CHAINS
print(f"\nElapsed: {elapsed:.0f}s for {n_total_samples} post-warmup samples "
      f"({elapsed / max(n_total_samples, 1):.3f} s/sample) -- use this to size the next run.")
print("Free parameters sampled:", sorted(k for k in samples if k not in ("obs",)))

# Save BEFORE plotting -- a plotting failure must not lose an expensive HMC run.
np.savez(os.path.join(HMC_DIR, "samples.npz"), **{k: np.asarray(v) for k, v in samples.items()})
print(f"Saved: {os.path.join(HMC_DIR, 'samples.npz')}")

# lens0_phi is fixed to exactly 0.0 (truth), so the derived lens0_e2 = (1-q)/(1+q)*
# sin(2*0) is EXACTLY 0.0 for every sample -- a zero-variance column that `corner`
# cannot histogram (unlike deriv-approx-source/nautilus-source, whose samples never
# contain lens0_e1/e2 at all in this fixed-phi setup, since neither runs
# compute_qphi_ellipticity; hmc-informed-source runs the real model, so it does).
plot_keys = [k for k in samples if k != "obs" and np.std(np.asarray(samples[k])) > 0]
dropped = sorted(set(samples) - set(plot_keys) - {"obs"})
if dropped:
    print(f"Dropping zero-variance column(s) from the corner plot: {dropped}")
samples_for_plot = {k: samples[k] for k in plot_keys}

plot_source_posterior(
    samples_for_plot, truths=truths_source,
    cfg={
        "output": {"output_dir": HMC_DIR},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

print(f"\nDone. Corner: {os.path.join(HMC_DIR, 'corner_all_params.png')}")
