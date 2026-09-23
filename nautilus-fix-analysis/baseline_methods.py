"""Step 0 / verification 6 -- fisher-source and deriv-approx-source numbers, before and after.

These two methods must not move by a single bit. Both are deterministic given
cfg["inference"]["rng_key"], so a straight comparison of the saved arrays is a
valid regression test. NUTS is cut down (300/600/1 chain) so this finishes in
minutes; determinism is what matters here, not posterior quality.

  .venv/bin/python baseline_methods.py before
  .venv/bin/python baseline_methods.py after
  .venv/bin/python baseline_methods.py compare
"""

import os
import sys
import time

import numpy as np
from common import OUT, save_json

from gwemfish import run_inference
from tutorial_cfg import build_ctx, cfg_for_run

TAG = sys.argv[1] if len(sys.argv) > 1 else "before"
MODE = "GW-only"
METHODS = ["fisher-source", "deriv-approx-source"]
NUTS = {"num_warmup": 300, "num_samples": 600, "num_chains": 1,
        "n_fisher_samples": 500, "rng_key": 123}


def digest(samples):
    """Per-parameter summary that changes if anything about the run changes."""
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "first": float(np.asarray(v).ravel()[0]),
                "last": float(np.asarray(v).ravel()[-1]),
                "n": int(np.asarray(v).size)}
            for k, v in sorted(samples.items())}


def run_all():
    ctx = build_ctx(MODE)
    sub = cfg_for_run(ctx, MODE)
    sub = {**sub, "inference": NUTS,
           "output": {"output_dir": os.path.join(OUT, f"baseline_{TAG}")}}

    out = {}
    for method in METHODS:
        print(f"\n{'=' * 60}\n{method}\n{'=' * 60}", flush=True)
        t0 = time.perf_counter()
        samples, _ = run_inference(ctx, mode=MODE, method=method, cfg=sub)
        wall = time.perf_counter() - t0
        out[method] = {"wall_seconds": wall, "digest": digest(samples)}
        print(f"  {method}: {wall:.1f}s, {len(samples)} params", flush=True)
    return out


def compare():
    import json

    with open(os.path.join(OUT, "baseline_before.json")) as f:
        before = json.load(f)
    with open(os.path.join(OUT, "baseline_after.json")) as f:
        after = json.load(f)

    ok = True
    for method in METHODS:
        b, a = before[method]["digest"], after[method]["digest"]
        if list(b) != list(a):
            print(f"{method}: FAIL -- parameter set changed")
            ok = False
            continue
        worst = max(((abs(a[k][s] - b[k][s]), k, s)
                     for k in b for s in ("mean", "std", "first", "last")),
                    default=(0.0, "", ""))
        print(f"{method}: max |delta| {worst[0]:.3e} at {worst[1]}.{worst[2]}  "
              f"wall {before[method]['wall_seconds']:.1f}s -> "
              f"{after[method]['wall_seconds']:.1f}s")
        ok &= worst[0] == 0.0
    print(f"\nbit-identical: {'PASS' if ok else 'FAIL'}")


if TAG == "compare":
    compare()
else:
    save_json(f"baseline_{TAG}.json", run_all())
