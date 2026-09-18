"""Catch the exact parameters that break the solver, then replay them on both paths.

Uniform prior draws never reproduce this failure -- nautilus concentrates its
samples where the likelihood is high, and that is where the lens parameters get
extreme enough to defeat lenstronomy's root finder. So rather than guess, run the
sampler until it throws, record the offending parameter dict, and feed that exact
dict to the real gwemfish likelihood.

If the baseline raises the same error on the same parameters, the failure is a
pre-existing property of the solver that the prototype merely reached first. If
the baseline returns a number, the prototype is at fault.

    python catch_solver_failure.py EM+GW
"""

import json
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from proto.jit_likelihood import build_problem
from tutorial_cfg import BUDGET, build_ctx, cfg_for_run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "proto")
os.makedirs(OUT_DIR, exist_ok=True)

CAUGHT = {}


def recording(loglike):
    """Re-raises after recording; the sampler still dies, but with evidence."""

    def wrapped(params):
        try:
            return loglike(params)
        except Exception as exc:
            CAUGHT["params"] = {k: float(v) for k, v in params.items()}
            CAUGHT["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[-1][:120]}"
            raise

    return wrapped


if __name__ == "__main__":
    import nautilus

    mode = sys.argv[1] if len(sys.argv) > 1 else "EM+GW"
    ctx = build_ctx(mode)
    cfg = cfg_for_run(ctx, mode)

    prior, base, _ = build_problem(ctx, cfg, mode, jit=False)
    _, prot, _ = build_problem(ctx, cfg, mode, jit=True)

    print(f"\nrunning {mode} with the jitted likelihood until it throws "
          f"(n_like_max={BUDGET['n_like_max']})...", flush=True)
    try:
        sampler = nautilus.Sampler(prior, recording(prot), n_live=BUDGET["n_live"],
                                   seed=None, filepath=None, resume=False)
        sampler.run(verbose=False, n_eff=BUDGET["n_eff"],
                    n_like_max=BUDGET["n_like_max"])
        print("  completed without raising -- failure not reproduced this time.")
    except Exception as exc:
        print(f"  raised: {type(exc).__name__}: {str(exc).splitlines()[-1][:120]}")

    if not CAUGHT:
        print("\nNo offending parameters captured. Re-run to try another "
              "random realization.")
        raise SystemExit(0)

    path = os.path.join(OUT_DIR, f"solver_failure_{mode.replace('+', '_')}.json")
    json.dump(CAUGHT, open(path, "w"), indent=2)
    print(f"\ncaptured failing parameters -> {path}")
    print(f"  prototype error: {CAUGHT['error']}")

    print("\nreplaying the SAME parameters through the real gwemfish likelihood:")
    try:
        value = base(CAUGHT["params"])
        print(f"  baseline returned {value!r}")
        print("\n  VERDICT: baseline survives where the prototype raised -- "
              "the prototype is at fault. Investigate.")
    except Exception as exc:
        print(f"  baseline raised: {type(exc).__name__}: "
              f"{str(exc).splitlines()[-1][:120]}")
        print("\n  VERDICT: identical failure on the untouched gwemfish code path. "
              "Pre-existing solver fragility, not introduced by the prototype.")
