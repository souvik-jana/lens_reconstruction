"""T2 -- is the prototype's ctx strip list still complete on current HEAD?

picklable.py strips ctx["fisher"] and ctx["likelihood"] before shipping ctx to a
worker, because a prior fisher / deriv-approx run leaves jitted closures there.
If any other key became unpicklable since, pool startup would fail at a point
nobody would connect to the cause -- so check every key, after a real fisher run.
"""

import pickle
import time

from common import save_json

from gwemfish import run_inference
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"

ctx = build_ctx(MODE)
sub = cfg_for_run(ctx, MODE)


def key_report(ctx, label):
    rows = {}
    for k, v in ctx.items():
        try:
            rows[k] = {"ok": True, "bytes": len(pickle.dumps(v))}
        except Exception as exc:
            rows[k] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    bad = sorted(k for k, r in rows.items() if not r["ok"])
    total = sum(r.get("bytes", 0) for r in rows.values())
    print(f"\n{label}: {len(rows)} keys, {total / 1024:.1f} KB picklable, "
          f"unpicklable: {bad}")
    return rows, bad, total


rows_before, bad_before, size_before = key_report(ctx, "ctx fresh (after setup only)")

print("\nrunning fisher-source to populate ctx['fisher'] / ctx['likelihood']...", flush=True)
sub_fisher = {**sub, "inference": {"n_fisher_samples": 200},
              "output": {"output_dir": "outputs", "save_samples_path": None,
                         "save_truths_path": None}}
t0 = time.perf_counter()
run_inference(ctx, mode=MODE, method="fisher-source", cfg=sub_fisher)
print(f"fisher-source done in {time.perf_counter() - t0:.1f}s")

rows_after, bad_after, size_after = key_report(ctx, "ctx after fisher-source")

STRIPPED = ("fisher", "likelihood")
leftover = [k for k in bad_after if k not in STRIPPED]
missing_from_ctx = [k for k in STRIPPED if k not in ctx]

stripped_ctx = {k: v for k, v in ctx.items() if k not in STRIPPED}
try:
    n = len(pickle.dumps(stripped_ctx))
    strip_ok, strip_err = True, None
    print(f"\nstripped ctx pickles: {n / 1024:.1f} KB")
except Exception as exc:
    n, strip_ok, strip_err = 0, False, f"{type(exc).__name__}: {exc}"
    print(f"\nstripped ctx FAILS to pickle: {strip_err}")

print(f"strip list {STRIPPED}: keys present {[k for k in STRIPPED if k in ctx]}, "
      f"absent {missing_from_ctx}")
print(f"unpicklable keys NOT covered by the strip list: {leftover}")
print(f"\nT2 {'PASS' if strip_ok and not leftover else 'FAIL'}")

save_json("t2_ctx_pickle.json", {
    "mode": MODE,
    "fresh": {"unpicklable": bad_before, "kb": size_before / 1024},
    "after_fisher": {"unpicklable": bad_after, "kb": size_after / 1024,
                     "per_key": rows_after},
    "strip_list": list(STRIPPED),
    "leftover_unpicklable": leftover,
    "stripped_ctx_kb": n / 1024,
    "stripped_ctx_pickles": strip_ok,
    "stripped_ctx_error": strip_err,
    "pass": bool(strip_ok and not leftover),
})
