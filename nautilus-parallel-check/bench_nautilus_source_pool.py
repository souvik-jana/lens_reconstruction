"""
Benchmark whether nautilus-source's likelihood can be parallelized as-is,
without touching src/gwemfish/.

Reuses tutorial/tutorial_gw_only.py's GW-only setup verbatim, then calls
build_gw_source_plane_problem() directly (bypassing run_inference) to:
  A. time the raw likelihood, single process
  B/C. test multiprocessing.Pool the same way nautilus builds its pool
       internally (nautilus/pool.py: Pool(n, initializer=..., initargs=(likelihood,))),
       under both the "spawn" start method (macOS/py3.9 default -- what
       `pool=N` does today with zero source changes) and "fork" (workers
       inherit already-built JAX/solver state via copy-on-write instead of
       pickling the closure)
  D/E. run the real nautilus.Sampler with a small n_like_max budget, single
       process vs pool=N, to compare wall-clock on the same task.
"""

import multiprocessing as mp
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
N_WORKERS = 4
N_LIKE_MAX = 2000

CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
CFG["lens_mass_parametrization"] = "q_phi"
CFG["gw"]["n_images"] = 4
CFG["gw"]["source_box_half_width"] = 0.8
CFG["gw"]["source_pos"] = (0.02, 0.01)
CFG["gw"]["solver_params"]["backend"] = "jaxtronomy"
CFG["gw"]["solver_params"]["jaxtronomy"]["solver"] = "lenstronomy"
CFG["gw"]["error_scales"]["sigma_td"] = 0.001
CFG["gw"]["error_scales"]["sigma_dL_eff"] = 0.1

Y0_LO, Y0_HI = 0.01992, 0.02005
Y1_LO, Y1_HI = 0.0091, 0.0106


def bench_pool(log_likelihood, points, t_serial, start_method, label):
    ctx_mp = mp.get_context(start_method)
    print(f"\n[{label}] multiprocessing.Pool(start_method={start_method!r}, n={N_WORKERS}) ...")
    try:
        t0 = time.perf_counter()
        with ctx_mp.Pool(N_WORKERS) as pool:
            pool.map(log_likelihood, points)
        t_par = time.perf_counter() - t0
        print(f"    OK: {len(points)} calls in {t_par:.2f}s -> {t_par/len(points)*1000:.1f} ms/call")
        print(f"    speedup vs serial: {t_serial/t_par:.2f}x")
        return t_par
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return None


def bench_sampler_pool(prior, log_likelihood, pool_arg, label):
    import nautilus

    print(f"\n[{label}] nautilus.Sampler(pool={pool_arg!r}), n_like_max={N_LIKE_MAX} ...")
    try:
        t0 = time.perf_counter()
        sampler = nautilus.Sampler(
            prior, log_likelihood, n_live=200, pool=pool_arg,
            filepath=None, resume=False,
        )
        sampler.run(verbose=False, n_like_max=N_LIKE_MAX)
        t_run = time.perf_counter() - t0
        print(f"    OK: {t_run:.1f}s wall")
        return t_run
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return None


# multiprocessing's "spawn" start method re-imports this module in each
# worker process; everything that builds ctx/JAX state must be guarded so it
# only runs once, in the parent -- a correctness requirement of `spawn`, not
# a style choice.
if __name__ == "__main__":
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

    print("\nBuilding GW source-plane problem (prior, log_likelihood)...")
    prior, log_likelihood, param_names = build_gw_source_plane_problem(ctx, {"priors": PRIORS})
    n_dim = prior.dimensionality()
    print(f"  params: {param_names}  (n_dim={n_dim})")

    rng = np.random.default_rng(0)
    unit_points = rng.uniform(size=(N_CALLS, n_dim))
    points = [prior.unit_to_dictionary(u) for u in unit_points]

    t0 = time.perf_counter()
    vals = [log_likelihood(p) for p in points]
    t_serial = time.perf_counter() - t0
    print(f"\n[A] serial: {N_CALLS} calls in {t_serial:.2f}s -> {t_serial/N_CALLS*1000:.1f} ms/call")
    print(f"    sample log-likelihoods: {vals[:3]}")

    bench_pool(log_likelihood, points, t_serial, "spawn", "B-spawn")
    bench_pool(log_likelihood, points, t_serial, "fork", "C-fork")

    print("\n[D] real nautilus.Sampler, single process (pool=None) baseline...")
    t_d = bench_sampler_pool(prior, log_likelihood, None, "D-serial")

    print("\n[E] real nautilus.Sampler, pool=N (default/spawn start method, zero source changes)...")
    t_e = bench_sampler_pool(prior, log_likelihood, N_WORKERS, "E-pool-spawn")

    mp.set_start_method("fork", force=True)
    print("\n[F] real nautilus.Sampler, pool=N (start method forced to fork)...")
    t_f = bench_sampler_pool(prior, log_likelihood, N_WORKERS, "F-pool-fork")

    print("\n--- summary ---")
    print(f"  A serial likelihood:         {t_serial:.2f}s ({N_CALLS} calls)")
    print(f"  D sampler, pool=None:        {t_d}")
    print(f"  E sampler, pool={N_WORKERS} (spawn): {t_e}")
    print(f"  F sampler, pool={N_WORKERS} (fork):  {t_f}")
