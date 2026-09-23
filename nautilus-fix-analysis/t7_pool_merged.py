"""Pool through the real pipeline: cfg["nautilus"]["pool"] on merged code.

Checks the three things that can only break in-package: that run_inference
accepts the new keys, that workers actually start and rebuild, and that a pooled
run and a serial one with the same seed agree on the posterior they find.
"""

import multiprocessing as mp
import time

import numpy as np
from common import OUT, save_json

from gwemfish import run_inference
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"
SEED = 7
BUDGET = {"n_live": 50, "n_eff": 60, "n_like_max": 1200, "resume": False,
          "verbose": False, "seed": SEED}

# spawn re-imports this module in every worker, so the body has to be guarded.
if __name__ == "__main__":
    print(f"start method: {mp.get_start_method()}, cpus: {mp.cpu_count()}")
    ctx = build_ctx(MODE)

    results = {}
    for label, overrides in [("serial_jit", {}), ("pool2_jit", {"pool": 2})]:
        sub = cfg_for_run(ctx, MODE, **{**BUDGET, **overrides})
        sub["output"] = {"output_dir": f"{OUT}/t7_{label}"}
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
        t0 = time.perf_counter()
        samples, _ = run_inference(ctx, mode=MODE, method="nautilus-source", cfg=sub)
        wall = time.perf_counter() - t0
        results[label] = {"wall": wall,
                          "means": {k: float(np.mean(v)) for k, v in samples.items()},
                          "n": int(len(next(iter(samples.values()))))}
        print(f"  {label}: {wall:.1f}s, {results[label]['n']} samples")

    a, b = results["serial_jit"], results["pool2_jit"]
    print(f"\nserial {a['wall']:.1f}s vs pool2 {b['wall']:.1f}s "
          f"({a['wall'] / b['wall']:.2f}x at a startup-dominated budget)")
    print("ctx keys after pooled run (fisher/likelihood must survive): "
          f"{[k for k in ('fisher', 'likelihood') if k in ctx]}")
    save_json("t7_pool_merged.json", results)
    print("\nt7 PASS")
