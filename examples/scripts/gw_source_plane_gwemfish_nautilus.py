"""
GW-only source-plane inference of (y0gw, y1gw) using gwemfish's OWN
``run_inference(..., method="nautilus-source")`` framework, configured to be
as apples-to-apples as possible with the hand-written cross-check in
``gw_source_plane_custom_nautilus.py``.

Both scripts:
  - use the identical fixed ground-truth EPL+SHEAR quad system dumped to
    ``examples/outputs/gw_source_plane_shared_truth/truth.json`` by
    ``gw_source_plane_shared_truth.py``,
  - fix every lens-mass parameter, T_star, and dL to their exact truth
    values, leaving only y0gw/y1gw free,
  - use the same Uniform(truth - halfwidth, truth + halfwidth) priors on
    y0gw/y1gw (halfwidths 0.02 and 0.004),
  - use jaxtronomy's LensModel/LensEquationSolver as the source-plane solver
    backend (``solver_backend="jaxtronomy"``) with its default
    ``arrival_time_sort=True``,
  - use nautilus with n_live=500, n_eff=3000,
  - checkpoint the nautilus run to /tmp (not the mounted outputs dir) to
    avoid a PermissionError on unlink() for HDF5 files written directly into
    the mounted output folder,
  - and are resumable across multiple bash-call-capped invocations via
    nautilus's own ``resume=True`` HDF5 checkpoint mechanism.

Unlike the hand-written script, this one does NOT reimplement the physics --
it goes through gwemfish's ``build_gw_source_plane_problem`` (via
``run_inference(mode="GW-only", method="nautilus-source")``), so any
difference in the resulting posterior directly reflects differences (or
bugs) in gwemfish's own inference plumbing rather than in the underlying
lensing/GW physics.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import json

import numpy as np
import numpyro.distributions as dist

from gwemfish import setup_gw_observation, make_default_cfg, run_inference

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/gw_source_plane_shared_truth")
POSTERIOR_OUT_PATH = os.path.join(OUTPUT_DIR, "gwemfish_posterior.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint to /tmp -- writing the nautilus .hdf5 checkpoint directly into the
# mounted examples/outputs/... folder can raise PermissionError on unlink()
# (quirk of that mount). Only the final JSON posterior goes to the mounted path.
NAUTILUS_CHECKPOINT = "/tmp/gwemfish_nautilus_checkpoint.hdf5"

# ----------------------------------------------------------------------------
# 1. Rebuild the EXACT same ctx as gw_source_plane_shared_truth.py.
#    GW-only physics has no RNG dependence, so re-running this identical
#    config reproduces the same T_star/dL/gw_obs bit-for-bit as truth.json.
# ----------------------------------------------------------------------------
CFG = make_default_cfg()
CFG["em"]["enabled"] = False             # GW-only, no EM/imaging needed
CFG["use_parameter_layout"] = False       # flat lens_theta_E, lens_e1, ... names
CFG["lens"]["lens_model_list"] = ["EPL", "SHEAR"]

KWARGS_EPL = {
    "theta_E": 1.2,
    "e1": 0.0,
    "e2": 0.1,
    "gamma": 2.0,
    "center_x": 0.0,
    "center_y": 0.0,
}
KWARGS_SHEAR = {
    "gamma1": 0.1,
    "gamma2": 0.0,
    "ra_0": 0.0,
    "dec_0": 0.0,
}
CFG["lens"]["kwargs_lens"] = [KWARGS_EPL, KWARGS_SHEAR]

GW_SOURCE_POS = (0.05, 1e-6)
CFG["gw"]["source_pos"] = GW_SOURCE_POS
CFG["gw"]["n_images"] = 4
CFG["gw"]["error_scales"] = {
    "sigma_td": 0.05,
    "sigma_dL_eff": 0.2,
    "epsilon": 0.005,
    "sigma_td_floor": 1.0,
}

print(f"Lens model list: {CFG['lens']['lens_model_list']}")
print(f"GW source_pos (arcsec): {GW_SOURCE_POS}")

ctx = setup_gw_observation({}, cfg=CFG)
tp = ctx["truth_params"]
gw_obs = ctx["gw_obs"]

n_img = len(ctx["x_img_gw"])
print(f"\nNumber of GW images solved: {n_img}")
if n_img != 4:
    raise RuntimeError(f"Expected a quad (4 images) but got {n_img}.")
print("Confirmed: 4 images (quad).")

# Sanity check against truth.json (bit-for-bit reproducibility check).
TRUTH_JSON_PATH = os.path.join(OUTPUT_DIR, "truth.json")
with open(TRUTH_JSON_PATH, "r") as f:
    truth_ref = json.load(f)

t_star_match = np.isclose(float(tp["T_star"]), float(truth_ref["truth_params"]["T_star"]))
dL_match = np.isclose(float(tp["dL"]), float(truth_ref["truth_params"]["dL"]))
print(f"\nT_star matches truth.json: {t_star_match} "
      f"({tp['T_star']} vs {truth_ref['truth_params']['T_star']})")
print(f"dL matches truth.json: {dL_match} "
      f"({tp['dL']} vs {truth_ref['truth_params']['dL']})")
if not (t_star_match and dL_match):
    raise RuntimeError(
        "Rebuilt ctx does not reproduce truth.json bit-for-bit -- check CFG "
        "construction against gw_source_plane_shared_truth.py."
    )

# ----------------------------------------------------------------------------
# 2. Priors: fix everything to truth except y0gw/y1gw. Read truth values off
#    ctx (not hardcoded) so there is zero chance of a copy-paste mismatch.
# ----------------------------------------------------------------------------
Y0_HALFWIDTH = 0.02
Y1_HALFWIDTH = 0.004
y0_truth = float(GW_SOURCE_POS[0])
y1_truth = float(GW_SOURCE_POS[1])

ctx["cfg"]["priors"] = {
    "lens_theta_E":  float(tp["lens_theta_E"]),
    "lens_e1":       float(tp["lens_e1"]),
    "lens_e2":       float(tp["lens_e2"]),
    "lens_gamma":    float(tp["lens_gamma"]),
    "lens_center_x": float(tp["lens_center_x"]),
    "lens_center_y": float(tp["lens_center_y"]),
    "lens_gamma1":   float(tp["lens_gamma1"]),
    "lens_gamma2":   float(tp["lens_gamma2"]),
    "T_star":        float(tp["T_star"]),
    "dL":            float(tp["dL"]),
    "y0gw":          dist.Uniform(y0_truth - Y0_HALFWIDTH, y0_truth + Y0_HALFWIDTH),
    "y1gw":          dist.Uniform(y1_truth - Y1_HALFWIDTH, y1_truth + Y1_HALFWIDTH),
}

print("\n--- Fixed-to-truth priors (from ctx['truth_params']) ---")
for k in ("lens_theta_E", "lens_e1", "lens_e2", "lens_gamma", "lens_center_x",
          "lens_center_y", "lens_gamma1", "lens_gamma2", "T_star", "dL"):
    print(f"  {k}: {ctx['cfg']['priors'][k]}")
print(f"  y0gw ~ Uniform({y0_truth - Y0_HALFWIDTH}, {y0_truth + Y0_HALFWIDTH})")
print(f"  y1gw ~ Uniform({y1_truth - Y1_HALFWIDTH}, {y1_truth + Y1_HALFWIDTH})")

# ----------------------------------------------------------------------------
# 3. Nautilus settings: jaxtronomy solver backend (matches sibling script),
#    checkpoint to /tmp, resume=True.
# ----------------------------------------------------------------------------
ctx["cfg"]["nautilus"] = {
    "solver_backend": "jaxtronomy",
    "n_live": 500,
    "n_eff": 3000,
    "filepath": NAUTILUS_CHECKPOINT,
    "resume": True,
    "verbose": True,
    # This script deliberately uses the legacy flat naming (use_parameter_layout
    # = False above), which has no compiled likelihood path, so the default
    # jit=True would refuse rather than silently run slow.
    "jit": False,
}

resume_msg = ("resume " + NAUTILUS_CHECKPOINT
              if os.path.isfile(NAUTILUS_CHECKPOINT) else "fresh run")
print(f"\nNautilus: {resume_msg}")
print(f"Nautilus cfg: {ctx['cfg']['nautilus']}")

# ----------------------------------------------------------------------------
# 4. Run inference via gwemfish's own framework.
# ----------------------------------------------------------------------------
print("\n--- GW-only inference: nautilus-source (gwemfish run_inference) ---\n")

samples, truths = run_inference(
    ctx,
    mode="GW-only",
    method="nautilus-source",
    cfg={
        "priors": ctx["cfg"]["priors"],
        "nautilus": ctx["cfg"]["nautilus"],
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": "pipeline_outputs.json",
            "json_tag": "gwemfish_nautilus_source",
        },
    },
)

# ----------------------------------------------------------------------------
# 5. Save samples + report.
# ----------------------------------------------------------------------------
y0gw_samples = np.asarray(samples["y0gw"])
y1gw_samples = np.asarray(samples["y1gw"])

posterior_out = {
    "y0gw": y0gw_samples.tolist(),
    "y1gw": y1gw_samples.tolist(),
}
with open(POSTERIOR_OUT_PATH, "w") as f:
    json.dump(posterior_out, f)

print("\n===== RESULTS =====")
print(f"Number of equal-weighted posterior samples: {len(y0gw_samples)}")
print(f"y0gw: mean={np.mean(y0gw_samples):.6f}, std={np.std(y0gw_samples):.6f} "
      f"(truth={y0_truth})")
print(f"y1gw: mean={np.mean(y1gw_samples):.8f}, std={np.std(y1gw_samples):.8f} "
      f"(truth={y1_truth})")
print(f"\nSaved posterior samples to: {POSTERIOR_OUT_PATH}")
print("Done.")
