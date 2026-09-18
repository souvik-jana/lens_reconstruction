"""
Test 2: does switching the solver backend from "jaxtronomy" to "helens" (pure
JAX, no jax.pure_callback / host round-trip) fix nautilus-source speed on its
own, single process, no multiprocessing? No src/gwemfish/ changes.

Same GW-only setup as bench_nautilus_source_pool.py, just backend swapped.
Reports: (1) whether the helens solver passes the same accuracy validation
gwemfish already runs internally (tol=0.05 arcsec) against the jaxtronomy
truth, and (2) per-call and small-sampler-run timing vs the jaxtronomy
baseline (216 ms/call serial; 521.9s for a single-process n_like_max=2000 run).
"""

import os
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX devices: {jax.devices()}")

import numpy as np
import numpyro.distributions as dist

from gwemfish import make_default_cfg, setup_em_observation, setup_gw_observation
from gwemfish.nautilus_source_inference import build_gw_source_plane_problem

N_CALLS = 60
N_LIKE_MAX = 2000
JAXTRONOMY_SERIAL_MS_PER_CALL = 216.0
JAXTRONOMY_D_BASELINE_S = 521.9

Y0_LO, Y0_HI = 0.01992, 0.02005
Y1_LO, Y1_HI = 0.0091, 0.0106

CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
CFG["lens_mass_parametrization"] = "q_phi"
CFG["gw"]["n_images"] = 4
CFG["gw"]["source_box_half_width"] = 0.8
CFG["gw"]["source_pos"] = (0.02, 0.01)
CFG["gw"]["solver_params"]["backend"] = "helens"
CFG["gw"]["error_scales"]["sigma_td"] = 0.001
CFG["gw"]["error_scales"]["sigma_dL_eff"] = 0.1

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
truth_params = ctx["truth_params"]

PRIORS = {
    "lens1_gamma1": float(truth_params["lens1_gamma1"]),
    "lens1_gamma2": float(truth_params["lens1_gamma2"]),
    "lens1_ra_0": float(truth_params["lens1_ra_0"]),
    "lens1_dec_0": float(truth_params["lens1_dec_0"]),
    "lens0_phi": float(truth_params["lens0_phi"]),
    "lens0_q": dist.Uniform(0.75, 0.85),
    "lens0_theta_E": float(truth_params["lens0_theta_E"]),
    "lens0_center_x": float(truth_params["lens0_center_x"]),
    "lens0_center_y": float(truth_params["lens0_center_y"]),
    "lens0_gamma": dist.Uniform(1.5, 2.4),
    "T_star": float(truth_params["T_star"]),
    "dL": float(truth_params["dL"]),
    "y0gw": dist.Uniform(Y0_LO, Y0_HI),
    "y1gw": dist.Uniform(Y1_LO, Y1_HI),
}
ctx["cfg"]["priors"] = dict(PRIORS)
ctx["cfg"]["gw"]["source_plane_bounds"] = {
    "y0gw": (Y0_LO, Y0_HI),
    "y1gw": (Y1_LO, Y1_HI),
}

print("\n[helens] Building GW source-plane problem (includes internal solver "
      "validation against jaxtronomy/lenstronomy truth, tol=0.05 arcsec)...")
try:
    prior, log_likelihood, param_names = build_gw_source_plane_problem(ctx, {"priors": PRIORS})
except Exception as exc:
    print(f"  FAILED to build problem with helens backend: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

n_dim = prior.dimensionality()
print(f"  params: {param_names}  (n_dim={n_dim})")

rng = np.random.default_rng(0)
unit_points = rng.uniform(size=(N_CALLS, n_dim))
points = [prior.unit_to_dictionary(u) for u in unit_points]

t0 = time.perf_counter()
vals = [log_likelihood(p) for p in points]
t_serial = time.perf_counter() - t0
per_call = t_serial / N_CALLS * 1000
print(f"\n[I] helens serial: {N_CALLS} calls in {t_serial:.2f}s -> {per_call:.1f} ms/call")
print(f"    jaxtronomy baseline was {JAXTRONOMY_SERIAL_MS_PER_CALL:.1f} ms/call "
      f"-> {JAXTRONOMY_SERIAL_MS_PER_CALL/per_call:.2f}x")
print(f"    sample log-likelihoods: {vals[:3]}")

print(f"\n[J] real nautilus.Sampler(pool=None), helens backend, n_like_max={N_LIKE_MAX}...")
import nautilus

t0 = time.perf_counter()
sampler = nautilus.Sampler(
    prior, log_likelihood, n_live=200, pool=None, filepath=None, resume=False,
)
sampler.run(verbose=False, n_like_max=N_LIKE_MAX)
t_j = time.perf_counter() - t0
print(f"    OK: {t_j:.1f}s wall  (jaxtronomy D baseline was {JAXTRONOMY_D_BASELINE_S}s)")
print(f"    speedup: {JAXTRONOMY_D_BASELINE_S/t_j:.2f}x")

print("\nDone.")
