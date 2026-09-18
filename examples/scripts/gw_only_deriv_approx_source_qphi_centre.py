"""
GW-only, deriv-approx-source, lens_mass_parametrization="q_phi".

System taken from lensing-degeneracies/gw-only-analysis/geometry-analysis/scripts/
geometry_infer_epl.py: single EPL lens (no external shear), theta_E=1.0, gamma=2.0,
q=0.6, phi=0.0, zl=0.7, zs=1.5, source at the "centre" position (0.02, 0.001) arcsec
(SOURCES["centre"] there) -- a near-axis quad.

Demonstrates the new (q, phi) reparametrization end to end: simulate the truth in
e1/e2 (as gwemfish always does at the kwargs_lens level), then run inference with
cfg["lens_mass_parametrization"]="q_phi" so lens_q/lens_phi are the free lens-mass
parameters instead of lens_e1/lens_e2 (which still appear in the output, as
numpyro.deterministic sites -- see src/gwemfish/ellipticity_reparam.py).

Free: lens_gamma, lens_q, y0gw, y1gw, T_star, dL (6 -- above the ~5-parameter budget
a GW-only quad supports, see gwemfish-infer skill; watch the Fisher-conditioning
diagnostic). Fixed to truth: lens_theta_E, lens_center_x/y, gamma1/gamma2 (no
shear), lens_phi.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_qphi_centre")

# ===================== TRUTH SYSTEM (geometry_infer_epl.py) =====================
THETA_E = 1.0
GAMMA_TRUTH = 2.0
Q_TRUTH = 0.6
PHI_TRUTH = 0.0
ZL, ZS = 0.7, 1.5
GW_SOURCE_POS = (0.02, 0.001)  # SOURCES["centre"]

# Truth-centered box half-width for y0gw/y1gw (source is close to the axis).
SOURCE_BOX_HALF_WIDTH = 0.02

# --- SMOKE-TEST SETTINGS --- (raise for a publication-quality run)
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 2

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CHAINS}"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scienceplots
from herculens.Util.param_util import phi_q2_ellipticity

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import run_inference, setup_gw_observation, plot_source_posterior

os.makedirs(OUTPUT_DIR, exist_ok=True)

E1_TRUTH, E2_TRUTH = phi_q2_ellipticity(PHI_TRUTH, Q_TRUTH)

BASE_CFG = {
    "em": {"enabled": False},
    "lens_mass_parametrization": "q_phi",
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": THETA_E,
                "e1": float(E1_TRUTH),
                "e2": float(E2_TRUTH),
                "gamma": GAMMA_TRUTH,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": ZL,
        "zs": ZS,
    },
    "gw": {
        "source_pos": GW_SOURCE_POS,
        "n_images": 4,
        # jaxtronomy's closed-form solver for the truth solve (gwemfish-simulate skill).
        "solver_params": {"jaxtronomy": {"solver": "analytical"}},
        "source_box_half_width": SOURCE_BOX_HALF_WIDTH,
    },
    "inference": {
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
    },
}

print(f"  EPL truth: theta_E={THETA_E} gamma={GAMMA_TRUTH} q={Q_TRUTH} phi={PHI_TRUTH}"
      f" -> e1={float(E1_TRUTH):+.6f} e2={float(E2_TRUTH):+.6f}")
print(f"  source (centre): {GW_SOURCE_POS}")

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp = ctx["truth_params"]
print(f"  n images: {len(ctx['x_img_gw'])}")
print(f"  truth_params lens_q={tp['lens_q']:.6f}  lens_phi={tp['lens_phi']:.6f}"
      f"  (from lens_e1={tp['lens_e1']:+.6f} lens_e2={tp['lens_e2']:+.6f})")

# Fix theta_E, center, shear, and lens_phi to truth; T_star/dL now free (default
# priors: Uniform(10, 1e12) / Uniform(1e-5, 50000)) alongside lens_gamma/lens_q
# (q_phi mode) and y0gw/y1gw -- 6 free params total, above the quad's ~5-parameter
# budget (7 GW observables), so watch the Fisher-conditioning diagnostic.
ctx["cfg"]["priors"] = {
    "lens_theta_E":  float(tp["lens_theta_E"]),
    "lens_center_x": float(tp["lens_center_x"]),
    "lens_center_y": float(tp["lens_center_y"]),
    "lens_gamma1":   float(tp["lens_gamma1"]),
    "lens_gamma2":   float(tp["lens_gamma2"]),
    "lens_phi":      float(tp["lens_phi"]),
}

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
    },
)

print("\nFree parameters sampled:", sorted(k for k in samples if k not in ("obs",)))
assert "lens_q" in samples, "q_phi mode did not activate"
# lens_phi is fixed to truth above (cfg["priors"]["lens_phi"]) -- add_qphi_columns_to_samples
# only backfills lens_e1/lens_e2 when BOTH lens_q and lens_phi are sampled, so with phi fixed
# they are not derivable from the samples dict alone and won't appear here; that's expected.
assert "lens_phi" not in samples, "lens_phi unexpectedly free"

plot_source_posterior(
    samples,
    truths=truths_source,
    cfg={
        "output": {"output_dir": OUTPUT_DIR},
        "plot": {"plot_mode": "combined", "save_path": "corner_all_params.png"},
    },
)

print(f"\nDone. Corner plot (all params together): "
      f"{os.path.join(OUTPUT_DIR, 'corner_all_params.png')}")
