"""Run a prototype nautilus problem, with or without jit, with or without a pool.

Mirrors ``gwemfish.nautilus_common.run_nautilus`` but forwards the two kwargs it
does not expose -- ``pool`` and ``n_batch`` -- and records nautilus' own
convergence diagnostics rather than anything hand-rolled.
"""

import time

import numpy as np

from .picklable import PicklableLikelihood, force_spawn
from .jit_likelihood import build_problem

RUN_KWARG_KEYS = ("n_eff", "n_like_max", "discard_exploration", "timeout")


def run(ctx, cfg, mode, jit=True, pool=None, n_live=200, filepath=None,
        verbose=False, seed=None, resume=False, **run_kwargs):
    """Returns (samples_dict, diagnostics_dict).

    Pass ``seed`` when comparing variants. Nautilus draws its own randomness, so
    two runs of the *same* likelihood land on different log_z and n_eff -- which
    makes an unseeded log_z comparison a measure of sampler noise, not of whether
    the likelihood changed.
    """
    import nautilus

    prior, loglike, param_names = build_problem(ctx, cfg, mode, jit=jit)

    if pool:
        # Only an *int* makes nautilus create the pool with
        # initializer=initialize_worker (nautilus/pool.py:59-61), so the
        # likelihood is pickled once per worker instead of once per map call.
        force_spawn()
        loglike = PicklableLikelihood(ctx, cfg, mode, jit=jit)

    t0 = time.perf_counter()
    sampler = nautilus.Sampler(
        prior, loglike,
        n_live=n_live,
        pool=pool,
        seed=seed,
        filepath=filepath,
        resume=resume,
    )
    sampler.run(verbose=verbose, **run_kwargs)
    wall = time.perf_counter() - t0

    points, log_w, log_l = sampler.posterior(equal_weight=True)
    samples = {name: np.array(points[:, j]) for j, name in enumerate(param_names)}

    # nautilus' own diagnostics, captured from the start rather than bolted on
    # after a run has already finished uninstrumented.
    diagnostics = {
        "wall_seconds": wall,
        "log_z": float(sampler.log_z),
        "n_eff": float(sampler.n_eff),
        "n_like": int(sampler.n_like),
        "n_posterior_samples": int(points.shape[0]),
        "mean_log_l": float(np.mean(log_l)),
    }
    return samples, diagnostics


def settings(budget, **overrides):
    run_kwargs = {k: budget[k] for k in RUN_KWARG_KEYS if k in budget}
    run_kwargs.update(overrides)
    return run_kwargs
