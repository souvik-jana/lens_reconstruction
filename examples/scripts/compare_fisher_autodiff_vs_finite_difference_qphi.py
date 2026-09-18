"""
Cross-check gwemfish's autodiff Fisher/Hessian (ctx["fisher"]["H0"], computed via
jax.hessian in fisher.py's compute_fisher) against an independent finite-difference
Hessian of the SAME likelihood function (ctx["likelihood"]["likelihood_function_vec"])
at the SAME expansion point (ctx["likelihood"]["u0"]). No source code changes --
uses only the public ctx entries run_inference already populates.

Same system/priors as gw_only_fisher_source_qphi.py (q_phi, GW-only, "centre" EPL).
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi")

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
SOURCE_BOX_HALF_WIDTH = 0.02
Y0_BOUNDS = (SOURCE_POS[0] - SOURCE_BOX_HALF_WIDTH, SOURCE_POS[0] + SOURCE_BOX_HALF_WIDTH)
Y1_BOUNDS = (SOURCE_POS[1] - SOURCE_BOX_HALF_WIDTH, SOURCE_POS[1] + SOURCE_BOX_HALF_WIDTH)
FRAC_TD = 0.002
FRAC_DLEFF = 0.1

# Relative finite-difference step (fraction of |u0_i|, floor for near-zero params).
REL_STEP = 1e-4

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

from gwemfish import run_inference, setup_gw_observation, prune_gw_images
from gwemfish.corner_plot_utils import plot_multi_comparison_corner

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
        "source_box_half_width": SOURCE_BOX_HALF_WIDTH,
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

ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens0_phi":      float(tp["lens0_phi"]),
    "lens0_gamma":    dist.Uniform(*GAMMA_BOUNDS),
    "lens0_q":        dist.Uniform(*Q_BOUNDS),
    "T_star":         dist.Uniform(*TSTAR_BOUNDS),
    "dL":             dist.Uniform(*DL_BOUNDS),
}

print("\n--- Running fisher-source once to populate ctx['likelihood'] / ctx['fisher'] ---\n")
_ = run_inference(
    ctx, mode="GW-only", method="fisher-source",
    cfg={"inference": {"n_fisher_samples": 10},
         "output": {"output_dir": OUTPUT_DIR, "json_path": "pipeline_outputs.json",
                     "json_tag": "fisher-source-for-fd-check"}},
)

keys = ctx["likelihood"]["keys_to_include"]
u0 = np.asarray(ctx["likelihood"]["u0"], dtype=np.float64)
logdensity = ctx["likelihood"]["likelihood_function_vec"]  # u (np/jax array) -> scalar log density
H0_autodiff = np.asarray(ctx["fisher"]["H0"], dtype=np.float64)

print(f"\nkeys_to_include: {keys}")
print(f"u0: {u0}")

n = len(u0)
h = np.maximum(np.abs(u0) * REL_STEP, 1e-8)  # per-parameter absolute step
print(f"finite-difference steps h: {dict(zip(keys, h))}")


def f(u):
    return float(logdensity(np.asarray(u, dtype=np.float64)))


f0 = f(u0)
print(f"logdensity(u0) = {f0:.6f}")

H_fd = np.zeros((n, n))

# Diagonal: standard central 3-point second derivative.
for i in range(n):
    up = u0.copy(); up[i] += h[i]
    um = u0.copy(); um[i] -= h[i]
    H_fd[i, i] = (f(up) - 2.0 * f0 + f(um)) / (h[i] ** 2)

# Off-diagonal: standard 4-point mixed partial stencil.
for i in range(n):
    for j in range(i + 1, n):
        upp = u0.copy(); upp[i] += h[i]; upp[j] += h[j]
        upm = u0.copy(); upm[i] += h[i]; upm[j] -= h[j]
        ump = u0.copy(); ump[i] -= h[i]; ump[j] += h[j]
        umm = u0.copy(); umm[i] -= h[i]; umm[j] -= h[j]
        val = (f(upp) - f(upm) - f(ump) + f(umm)) / (4.0 * h[i] * h[j])
        H_fd[i, j] = val
        H_fd[j, i] = val

print("\n" + "=" * 100)
print("Hessian comparison: gwemfish autodiff (jax.hessian) vs finite-difference")
print("=" * 100)
print(f"\n{'':<14}" + "".join(f"{k:>16}" for k in keys) + "   (autodiff row)")
for i, ki in enumerate(keys):
    print(f"{ki:<14}" + "".join(f"{H0_autodiff[i, j]:>16.4g}" for j in range(n)))
print(f"\n{'':<14}" + "".join(f"{k:>16}" for k in keys) + "   (finite-diff row)")
for i, ki in enumerate(keys):
    print(f"{ki:<14}" + "".join(f"{H_fd[i, j]:>16.4g}" for j in range(n)))

diag_autodiff = np.diag(H0_autodiff)
diag_fd = np.diag(H_fd)
rel_diag_diff = np.abs(diag_fd - diag_autodiff) / np.maximum(np.abs(diag_autodiff), 1e-300)

print("\n" + "-" * 100)
print(f"{'param':<14}{'H_ii autodiff':>18}{'H_ii finite-diff':>20}{'rel diff':>14}")
for i, k in enumerate(keys):
    print(f"{k:<14}{diag_autodiff[i]:>18.6g}{diag_fd[i]:>20.6g}{rel_diag_diff[i]:>14.3%}")
print("-" * 100)

eig_autodiff = np.sort(np.linalg.eigvalsh(H0_autodiff))
eig_fd = np.sort(np.linalg.eigvalsh(H_fd))
print(f"\nEigenvalues, autodiff:    {eig_autodiff}")
print(f"Eigenvalues, finite-diff: {eig_fd}")

cond_autodiff = np.abs(eig_autodiff).max() / max(np.abs(eig_autodiff).min(), 1e-300)
cond_fd = np.abs(eig_fd).max() / max(np.abs(eig_fd).min(), 1e-300)
print(f"\nCondition number, autodiff:    {cond_autodiff:.4g}")
print(f"Condition number, finite-diff: {cond_fd:.4g}")

# 1-sigma widths sqrt(diag(inv(-H))) -- what the Fisher covariance actually reports.
try:
    cov_autodiff = np.linalg.inv(-H0_autodiff)
except np.linalg.LinAlgError:
    cov_autodiff = np.linalg.pinv(-H0_autodiff)
try:
    cov_fd = np.linalg.inv(-H_fd)
except np.linalg.LinAlgError:
    cov_fd = np.linalg.pinv(-H_fd)
sig_autodiff = np.sqrt(np.abs(np.diag(cov_autodiff)))
sig_fd = np.sqrt(np.abs(np.diag(cov_fd)))
print(f"\n{'param':<14}{'sigma autodiff':>18}{'sigma finite-diff':>20}")
for i, k in enumerate(keys):
    print(f"{k:<14}{sig_autodiff[i]:>18.6g}{sig_fd[i]:>20.6g}")

max_rel = float(np.max(rel_diag_diff))
print(f"\nMax relative difference in Hessian diagonal: {max_rel:.3%}")
print("PASS: autodiff and finite-difference Hessians agree closely" if max_rel < 0.05
      else "NOTE: noticeable disagreement -- check step size / near-degenerate direction")

# --------------------------------------------------------------------------
# Corner plot: N(u0, inv(-H)) Gaussian draws from each Hessian, overlaid.
# --------------------------------------------------------------------------
N_SAMPLES = 20000
rng = np.random.default_rng(0)
samples_autodiff_arr = rng.multivariate_normal(u0, cov_autodiff, size=N_SAMPLES)
samples_fd_arr = rng.multivariate_normal(u0, cov_fd, size=N_SAMPLES)

samples_autodiff = {k: samples_autodiff_arr[:, i] for i, k in enumerate(keys)}
samples_fd = {k: samples_fd_arr[:, i] for i, k in enumerate(keys)}

truths_dict = {k: float(u0[i]) for i, k in enumerate(keys)}

plot_multi_comparison_corner(
    [samples_autodiff, samples_fd],
    {"all": keys},
    labels=["Fisher (autodiff H0)", "Fisher (finite-difference H)"],
    colors=["C0", "C1"],
    truths_dict={"all": truths_dict},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_fisher_autodiff_vs_finite_difference_corner.png"),
)
print(f"\nSaved corner: "
      f"{os.path.join(OUTPUT_DIR, 'compare_fisher_autodiff_vs_finite_difference_corner.png')}")
