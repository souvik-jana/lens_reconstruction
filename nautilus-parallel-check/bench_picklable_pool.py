"""
Test 1: does a picklable-by-reconstruction likelihood make `pool=N` actually
work for nautilus-source? No src/gwemfish/ changes.

`build_gw_source_plane_problem`'s log_likelihood is a local closure -- pickle
can't send it to a spawned worker (bench_nautilus_source_pool.py already
proved this fails). Trick: instead of pickling the closure itself, pickle only
plain config (cfg_full, priors), and have each worker rebuild its own
log_likelihood ONCE (cached in a process-global) the first time it's called.
That sidesteps pickling entirely -- only the (picklable) config crosses the
process boundary.

Cost of this trick: every worker independently re-runs setup_em_observation +
setup_gw_observation + build_gw_source_plane_problem (JAX warm-up, EM sim,
solver validation) once. Measures that cost against the parallel speedup.
"""

import multiprocessing as mp
import os
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import numpyro.distributions as dist

from gwemfish import make_default_cfg, setup_em_observation, setup_gw_observation
from gwemfish.nautilus_source_inference import build_gw_source_plane_problem

N_CALLS = 60
N_WORKERS = 4
N_LIKE_MAX = 2000

Y0_LO, Y0_HI = 0.01992, 0.02005
Y1_LO, Y1_HI = 0.0091, 0.0106


def base_cfg():
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = "q_phi"
    cfg["gw"]["n_images"] = 4
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["source_pos"] = (0.02, 0.01)
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "lenstronomy"
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    return cfg


# process-global cache: each worker builds its own log_likelihood once, on
# first call, and reuses it for every subsequent call in that same process.
_WORKER_LOGLIKE = None


def _worker_build(cfg_full, priors_cfg):
    global _WORKER_LOGLIKE
    if _WORKER_LOGLIKE is None:
        t0 = time.perf_counter()
        ctx = setup_em_observation(cfg=cfg_full)
        ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
        ctx["cfg"]["priors"] = priors_cfg
        _, loglike, _ = build_gw_source_plane_problem(ctx, {"priors": priors_cfg})
        dt = time.perf_counter() - t0
        print(f"    [pid {os.getpid()}] worker rebuild: {dt:.1f}s", flush=True)
        _WORKER_LOGLIKE = loglike
    return _WORKER_LOGLIKE


class PicklableGWSourceLikelihood:
    """Pickles as plain config only; each worker rebuilds the real
    (unpicklable) closure itself, once, on first call."""

    def __init__(self, cfg_full, priors_cfg):
        self.cfg_full = cfg_full
        self.priors_cfg = priors_cfg

    def __call__(self, params):
        loglike = _worker_build(self.cfg_full, self.priors_cfg)
        return loglike(params)


if __name__ == "__main__":
    CFG = base_cfg()
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

    print("\nBuilding prior (parent process; closure discarded, only bounds kept)...")
    prior, _discarded_loglike, param_names = build_gw_source_plane_problem(ctx, {"priors": PRIORS})
    n_dim = prior.dimensionality()
    print(f"  params: {param_names}  (n_dim={n_dim})")

    rng = np.random.default_rng(0)
    unit_points = rng.uniform(size=(N_CALLS, n_dim))
    points = [prior.unit_to_dictionary(u) for u in unit_points]

    picklable_loglike = PicklableGWSourceLikelihood(ctx["cfg"], PRIORS)

    print(f"\n[G] Pool(spawn, n={N_WORKERS}).map(picklable_loglike, ...) -- "
          f"two passes over the SAME pool to separate one-time worker "
          f"rebuild cost from steady-state per-call cost")
    ctx_mp = mp.get_context("spawn")
    try:
        with ctx_mp.Pool(N_WORKERS) as pool:
            t0 = time.perf_counter()
            out1 = pool.map(picklable_loglike, points)
            t_pass1 = time.perf_counter() - t0

            t0 = time.perf_counter()
            out2 = pool.map(picklable_loglike, points)
            t_pass2 = time.perf_counter() - t0
        print(f"    pass 1 (cold, incl. worker rebuild): {t_pass1:.2f}s "
              f"({t_pass1/N_CALLS*1000:.1f} ms/call)")
        print(f"    pass 2 (warm workers):                {t_pass2:.2f}s "
              f"({t_pass2/N_CALLS*1000:.1f} ms/call)")
        print(f"    sample results match direction: {out1[:2]} / {out2[:2]}")
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")

    print(f"\n[H] real nautilus.Sampler(pool={N_WORKERS}, spawn) with picklable "
          f"likelihood, n_like_max={N_LIKE_MAX} (compare to D=521.9s serial baseline)")
    import nautilus

    try:
        t0 = time.perf_counter()
        sampler = nautilus.Sampler(
            prior, picklable_loglike, n_live=200, pool=N_WORKERS,
            filepath=None, resume=False,
        )
        sampler.run(verbose=False, n_like_max=N_LIKE_MAX)
        t_h = time.perf_counter() - t0
        print(f"    OK: {t_h:.1f}s wall  (D baseline was 521.9s serial)")
        print(f"    speedup: {521.9/t_h:.2f}x")
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")

    print("\nDone.")
