"""Make a nautilus likelihood survive the trip to a worker process.

``nautilus.Sampler(pool=N)`` cannot be used with gwemfish today, and not because
of nautilus: every builder returns a function defined *inside* another function,
and pickle can only serialise functions it can find by name at module level:

    AttributeError: Can't get local object
                    'build_gw_source_plane_problem.<locals>.log_likelihood'

The usual workaround is to ship the config and have each worker re-simulate the
system from scratch, which the earlier bench measured at 7.2-8.2 s per worker and
which silently produces *different data* if any seed drifts. Neither is necessary:
the whole ctx pickles in 44 KB, instantly. So the worker is handed the exact same
simulated system and only rebuilds the cheap part -- the prior and the likelihood
closure -- which costs 0.01-0.6 s for most modes.
"""

import multiprocessing as mp
import os
import time

# Written into ctx by a previous fisher / deriv-approx run, and not picklable:
# both hold jitted closures (simple_pipeline.py:2097-2098 and :2114). Nautilus
# never reads either. Without this strip, any multi-method script would fail at
# pool creation rather than at a point anyone would connect to the cause.
UNPICKLABLE_CTX_KEYS = ("fisher", "likelihood")

WORKER = {}


def strip_ctx(ctx):
    return {k: v for k, v in ctx.items() if k not in UNPICKLABLE_CTX_KEYS}


def check_priors_picklable(priors):
    """Fail early, by name, instead of deep inside pool startup.

    numpyro ``dist.Uniform`` / ``dist.Normal`` pickle fine. A lambda or any other
    callable prior does not, and the error it raises from inside the pool names
    nothing useful.
    """
    import pickle

    bad = []
    for name, spec in (priors or {}).items():
        try:
            pickle.dumps(spec)
        except Exception:
            bad.append(name)
    if bad:
        raise TypeError(
            f"cannot use pool: these priors are not picklable: {bad}. "
            "Callable/lambda priors cannot cross a process boundary -- use a "
            "numpyro distribution or a fixed float instead."
        )


def worker_init():
    """Pin each worker to one thread before JAX starts up.

    Four workers each spawning as many BLAS threads as there are cores turns a
    speed-up into a slowdown. gwemfish already has the helper for this; nautilus
    only applies ``threadpool_limits`` around its own neural-network sections,
    not around likelihood calls.
    """
    from gwemfish.jax_config import setup_jax

    setup_jax(ncpus=1)


class PicklableLikelihood:
    """Carries the ctx to each worker; the worker builds its own closure once."""

    def __init__(self, ctx, cfg, mode, jit=True):
        self.ctx = strip_ctx(ctx)
        self.cfg = cfg
        self.mode = mode
        self.jit = jit
        check_priors_picklable(cfg.get("priors"))

    def __call__(self, params):
        key = (self.mode, self.jit)
        if key not in WORKER:
            worker_init()
            from .jit_likelihood import build_problem

            t0 = time.perf_counter()
            _, loglike, _ = build_problem(self.ctx, self.cfg, self.mode, jit=self.jit)
            WORKER[key] = loglike
            print(f"    [pid {os.getpid()}] built {self.mode} likelihood in "
                  f"{time.perf_counter() - t0:.1f}s "
                  f"(start method: {mp.get_start_method()})", flush=True)
        return WORKER[key](params)


def force_spawn():
    """``nautilus/pool.py`` does a bare ``from multiprocessing import Pool``.

    That takes the platform default, which is ``fork`` on Linux -- and forking a
    process with JAX's thread pool already running deadlocks. The earlier bench
    hit exactly this ("os.fork() ... JAX is multithreaded, so this will likely
    lead to a deadlock", then a hang). Setting spawn globally is what makes
    ``pool=N`` safe on both macOS and Linux.
    """
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
