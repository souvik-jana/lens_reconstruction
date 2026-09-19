"""Head-to-head benchmark: today's nautilus-source vs the prototype.

Four variants per mode, all on the same ctx, the same priors, the same budget and
the same seed, so the only thing that differs is how the likelihood is evaluated:

    A baseline   what run_inference does today  (eager, single process)
    B jit        compiled likelihood            (eager -> compiled)
    C pool       today's likelihood, 4 workers  (parallel only)
    D jit+pool   both

Run one mode:   python bench_proto.py GW-only
Run all three:  python bench_proto.py
"""

import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from proto.run_proto import run, settings
from tutorial_cfg import BUDGET, build_ctx, cfg_for_run

N_WORKERS = 4
# Same seed for every variant, so a log_z difference means the likelihood
# changed rather than that nautilus happened to draw a different realization.
SEED = 1234
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "proto")
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    ("A_baseline", dict(jit=False, pool=None)),
    ("B_jit", dict(jit=True, pool=None)),
    ("C_pool", dict(jit=False, pool=N_WORKERS)),
    ("D_jit_pool", dict(jit=True, pool=N_WORKERS)),
]


def bench_mode(mode):
    print(f"\n{'#' * 70}\n# {mode}\n{'#' * 70}")
    ctx = build_ctx(mode)
    cfg = cfg_for_run(ctx, mode)
    run_kwargs = settings(BUDGET)

    out = {}
    for name, opts in VARIANTS:
        print(f"\n--- {mode} / {name} ---")
        t0 = time.perf_counter()
        samples, diag = run(ctx, cfg, mode, n_live=BUDGET["n_live"], seed=SEED,
                            **opts, **run_kwargs)
        diag["total_seconds"] = time.perf_counter() - t0
        diag["samples"] = {k: [float(np.median(v)),
                               float(np.percentile(v, 16)),
                               float(np.percentile(v, 84))]
                           for k, v in samples.items()}
        out[name] = diag
        print(f"    wall {diag['wall_seconds']:7.1f}s  n_like {diag['n_like']:6d}  "
              f"n_eff {diag['n_eff']:7.1f}  log_z {diag['log_z']:.4f}  "
              f"samples {diag['n_posterior_samples']}")
        np.savez(os.path.join(OUT_DIR, f"{mode.replace('+', '_')}_{name}.npz"), **samples)

    base = out["A_baseline"]["wall_seconds"]
    print(f"\n{mode} summary (baseline = {base:.1f}s)")
    print(f"  {'variant':<12} {'wall':>9} {'speedup':>9} {'n_like':>8} {'log_z':>10}")
    for name, _ in VARIANTS:
        d = out[name]
        print(f"  {name:<12} {d['wall_seconds']:>8.1f}s "
              f"{base / d['wall_seconds']:>8.2f}x {d['n_like']:>8d} "
              f"{d['log_z']:>10.4f}")

    # With the seed pinned, log_z is the sharpest check that the likelihood was
    # not silently altered: the evidence integrates the whole likelihood surface,
    # so a change shows up here even when the marginals look identical.
    base_z = out["A_baseline"]["log_z"]
    dz = max(abs(out[name]["log_z"] - base_z) for name, _ in VARIANTS)
    print(f"  max |delta log_z| vs baseline: {dz:.4f}")
    return out


# The one place the "no __main__ boilerplate" rule has to give: `spawn` starts a
# worker by importing this module, so without the guard every worker re-runs the
# whole benchmark. Observed before the guard was added -- three nested copies of
# the benchmark competing for the same cores, which makes any timing meaningless.
if __name__ == "__main__":
    modes = sys.argv[1:] or ["GW-only", "EM+GW", "EM-only"]
    results = {m: bench_mode(m) for m in modes}

    path = os.path.join(OUT_DIR, "timings.json")
    existing = json.load(open(path)) if os.path.exists(path) else {}
    existing.update(results)
    json.dump(existing, open(path, "w"), indent=2)
    print(f"\nWrote {path}")
