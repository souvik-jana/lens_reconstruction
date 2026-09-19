"""Correctness + compile-count check for the prototype likelihoods.

Speed is worthless if the number changed. For each mode this draws random points
from the prior, evaluates the real gwemfish likelihood and the prototype one on
exactly the same points, and reports the largest disagreement. It also counts XLA
compilations per call, which is the number the whole exercise is about: the
baseline recompiles 6 programs on every call, the prototype should compile zero
after warm-up.
"""

import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax._src.interpreters.pxla as pxla

from proto.jit_likelihood import build_problem
from tutorial_cfg import build_ctx, cfg_for_run

N_DRAWS = 200
MODES = ["GW-only", "EM+GW", "EM-only"]

COMPILES = [0]
ORIGINAL_FROM_HLO = pxla.UnloadedMeshExecutable.from_hlo


def counting_from_hlo(*args, **kwargs):
    COMPILES[0] += 1
    return ORIGINAL_FROM_HLO(*args, **kwargs)


pxla.UnloadedMeshExecutable.from_hlo = counting_from_hlo


def compiles_per_call(loglike, points):
    for p in points[:3]:
        loglike(p)
    COMPILES[0] = 0
    for p in points[:10]:
        loglike(p)
    return COMPILES[0] / 10


def ms_per_call(loglike, points):
    for p in points[:3]:
        loglike(p)
    t0 = time.perf_counter()
    for p in points:
        loglike(p)
    return (time.perf_counter() - t0) / len(points) * 1000


def check_mode(mode):
    print(f"\n{'=' * 64}\n{mode}\n{'=' * 64}")
    ctx = build_ctx(mode)
    sub = cfg_for_run(ctx, mode)

    prior, loglike_base, names = build_problem(ctx, sub, mode, jit=False)
    _, loglike_jit, _ = build_problem(ctx, sub, mode, jit=True)

    rng = np.random.default_rng(0)
    points = [prior.unit_to_dictionary(u)
              for u in rng.uniform(size=(N_DRAWS, prior.dimensionality()))]

    base = np.array([loglike_base(p) for p in points])
    prot = np.array([loglike_jit(p) for p in points])

    finite = np.isfinite(base) & (base > -1e299)
    n_reject = int((~finite).sum())
    reject_agree = bool(np.all(prot[~finite] <= -1e299)) if n_reject else True
    max_abs = float(np.max(np.abs(prot[finite] - base[finite]))) if finite.any() else 0.0
    max_rel = (float(np.max(np.abs((prot[finite] - base[finite]) / base[finite])))
               if finite.any() else 0.0)

    timing_points = [p for p, f in zip(points, finite) if f][:30] or points[:30]
    print(f"  params ({prior.dimensionality()}): {names}")
    print(f"  accepted {int(finite.sum())}/{N_DRAWS}, rejected {n_reject} "
          f"(reject agrees: {reject_agree})")
    print(f"  max |delta logL|  = {max_abs:.3e}")
    print(f"  max relative diff = {max_rel:.3e}")
    print(f"  compiles/call  baseline {compiles_per_call(loglike_base, timing_points):.1f}"
          f"  ->  prototype {compiles_per_call(loglike_jit, timing_points):.1f}")
    t_base = ms_per_call(loglike_base, timing_points)
    t_jit = ms_per_call(loglike_jit, timing_points)
    print(f"  ms/call        baseline {t_base:.1f}  ->  prototype {t_jit:.2f}"
          f"   ({t_base / t_jit:.1f}x)")

    return {"mode": mode, "max_abs": max_abs, "max_rel": max_rel,
            "reject_agree": reject_agree, "ms_base": t_base, "ms_jit": t_jit}


results = [check_mode(m) for m in MODES]

print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
print(f"{'mode':<10} {'max|dlogL|':>12} {'max rel':>10} {'ms base':>9} "
      f"{'ms jit':>8} {'speedup':>8}  ok")
ok_all = True
for r in results:
    # Relative, not absolute: XLA fuses the same float64 arithmetic in a different
    # order, so a logL of -4.7e5 disagrees in its last bits. 1e-10 relative is ~5
    # orders of magnitude tighter than anything nested sampling can resolve.
    ok = r["max_rel"] < 1e-10 and r["reject_agree"]
    ok_all &= ok
    print(f"{r['mode']:<10} {r['max_abs']:>12.3e} {r['max_rel']:>10.2e} "
          f"{r['ms_base']:>9.1f} {r['ms_jit']:>8.2f} "
          f"{r['ms_base'] / r['ms_jit']:>7.1f}x  {'PASS' if ok else 'FAIL'}")
print(f"\noverall: {'PASS' if ok_all else 'FAIL'}")
