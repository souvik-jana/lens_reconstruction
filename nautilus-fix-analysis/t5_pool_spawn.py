"""T5 -- does pool=N actually work on this machine (darwin), not just on Linux?

The REPORT's pool numbers were measured on Linux, where the trap was `fork`
deadlocking JAX. macOS already defaults to `spawn`, so the failure mode here is
the opposite one: every worker re-imports and rebuilds, and the startup cost has
to be paid back inside the run. Tiny budget -- this measures that pool workers
start, agree, and finish, not throughput.
"""

import multiprocessing as mp
import time

import numpy as np
from common import save_json

from proto.run_proto import run
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"
BUDGET = {"n_eff": 60, "n_like_max": 1200}
SEED = 7

# The one place the "no __main__ boilerplate" rule has to give: `spawn` re-imports
# the main module in every worker, so an unguarded script body would re-run the
# whole test inside each child.
if __name__ == "__main__":
    print(f"default start method: {mp.get_start_method()}")
    print(f"cpu_count: {mp.cpu_count()}")

    ctx = build_ctx(MODE)
    sub = cfg_for_run(ctx, MODE)

    # The identity check has to hold pool fixed: nautilus dispatches points in
    # different batches with and without a pool, so a pool-vs-nopool log_z
    # difference measures batching, not the likelihood. eager_pool2 vs jit_pool2
    # is the pair the REPORT's end-to-end claim is about.
    variants = [("jit_nopool", True, None), ("jit_pool2", True, 2),
                ("eager_pool2", False, 2)]
    results = {}
    for label, jit, pool in variants:
        print(f"\n{'=' * 60}\n{label}: jit={jit} pool={pool}\n{'=' * 60}", flush=True)
        t0 = time.perf_counter()
        samples, diag = run(ctx, sub, MODE, jit=jit, pool=pool, n_live=50,
                            seed=SEED, verbose=False, **BUDGET)
        diag["start_method"] = mp.get_start_method()
        diag["wall_total"] = time.perf_counter() - t0
        results[label] = {"diag": diag,
                          "means": {k: float(np.mean(v)) for k, v in samples.items()}}
        print(f"  wall {diag['wall_seconds']:.1f}s  n_like {diag['n_like']}  "
              f"n_eff {diag['n_eff']:.1f}  log_z {diag['log_z']:.4f}  "
              f"samples {diag['n_posterior_samples']}")

    a = results["eager_pool2"]["diag"]
    b = results["jit_pool2"]["diag"]
    same_run = (a["n_like"] == b["n_like"]
                and abs(a["log_z"] - b["log_z"]) < 1e-4
                and abs(a["n_eff"] - b["n_eff"]) < 1e-6)
    print(f"\nsame seed, eager-pool vs jit-pool identical run: {same_run}")
    print(f"  n_like {a['n_like']} vs {b['n_like']}")
    print(f"  n_eff  {a['n_eff']:.4f} vs {b['n_eff']:.4f}")
    print(f"  log_z  {a['log_z']:.6f} vs {b['log_z']:.6f}")
    print(f"  jit speedup inside the pool: {a['wall_seconds'] / b['wall_seconds']:.2f}x")

    nopool = results["jit_nopool"]["diag"]
    print(f"  pool(2) vs no-pool wall: {nopool['wall_seconds']:.1f}s vs "
          f"{b['wall_seconds']:.1f}s (tiny budget -- startup-dominated)")

    save_json("t5_pool_spawn.json", {"results": results, "identical_run": same_run,
                                     "budget": BUDGET, "seed": SEED})
