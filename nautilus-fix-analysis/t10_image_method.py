"""Verification 7 (image half) -- nautilus-image warns on jit, and pool applies.

jit cannot help this method (numpyro rebuilds the model in Python every call), so
the decision was: warn, run eager, and let pool do the work. Also measures the
per-worker rebuild cost, which the plan flagged as unmeasured.
"""

import multiprocessing as mp
import time
import warnings

from common import OUT, save_json

from gwemfish import run_inference
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"
BUDGET = {"n_live": 50, "n_eff": 40, "n_like_max": 600, "resume": False,
          "verbose": False, "seed": 7}

if __name__ == "__main__":
    ctx = build_ctx(MODE)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sub = cfg_for_run(ctx, MODE, **BUDGET)
        sub["output"] = {"output_dir": f"{OUT}/t10_serial"}
        t0 = time.perf_counter()
        run_inference(ctx, mode=MODE, method="nautilus-image", cfg=sub)
        serial = time.perf_counter() - t0

    jit_warnings = [str(w.message) for w in caught
                    if "jit" in str(w.message) and "nautilus-image" in str(w.message)]
    print(f"\njit warning raised: {bool(jit_warnings)}")
    for w in jit_warnings:
        print(f"  {w}")

    sub = cfg_for_run(ctx, MODE, **{**BUDGET, "pool": 4})
    sub["output"] = {"output_dir": f"{OUT}/t10_pool4"}
    t0 = time.perf_counter()
    run_inference(ctx, mode=MODE, method="nautilus-image", cfg=sub)
    pooled = time.perf_counter() - t0

    print(f"\nnautilus-image serial {serial:.1f}s  pool4 {pooled:.1f}s  "
          f"({serial / pooled:.2f}x at a 600-call budget)")
    print(f"start method: {mp.get_start_method()}")
    save_json("t10_image_method.json", {
        "jit_warning": jit_warnings, "serial_seconds": serial,
        "pool4_seconds": pooled, "budget": BUDGET})
    print("\nt10 PASS" if jit_warnings else "\nt10 FAIL (no warning)")
