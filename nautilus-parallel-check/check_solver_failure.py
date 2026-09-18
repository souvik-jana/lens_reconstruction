"""Does the prototype introduce the 'Signs are not different' solver failure?

That error comes from lenstronomy's ``brentq_nojit`` root finder, reached through
jaxtronomy's analytical EPL+shear solver, which gwemfish calls on the host via
``jax.pure_callback`` (image_finders.py:141). It is plain numpy code, so in
principle jit cannot reach it differently -- but "in principle" is not evidence.

This evaluates the real gwemfish likelihood and the prototype one on exactly the
same prior draws and records, per point, whether each raised. If jit were at
fault there would be a point where the baseline succeeds and the prototype
raises. If the failure is a property of the parameter values, both raise on the
same points.

    python check_solver_failure.py EM+GW
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from proto.jit_likelihood import build_problem
from tutorial_cfg import build_ctx, cfg_for_run

N_DRAWS = 3000


def evaluate(loglike, point):
    """Returns (value, error_type). Deliberately catches everything: the whole
    question is which exception each path raises, so none may be swallowed
    silently."""
    try:
        return loglike(point), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"


mode = sys.argv[1] if len(sys.argv) > 1 else "EM+GW"
ctx = build_ctx(mode)
cfg = cfg_for_run(ctx, mode)

prior, base, _ = build_problem(ctx, cfg, mode, jit=False)
_, prot, _ = build_problem(ctx, cfg, mode, jit=True)

rng = np.random.default_rng(12345)
points = [prior.unit_to_dictionary(u)
          for u in rng.uniform(size=(N_DRAWS, prior.dimensionality()))]

both_ok = base_only = prot_only = both_fail = 0
mismatch = []
errors = {}

for i, p in enumerate(points):
    vb, eb = evaluate(base, p)
    vp, ep = evaluate(prot, p)
    if eb is None and ep is None:
        both_ok += 1
    elif eb is None and ep is not None:
        prot_only += 1
        mismatch.append((i, "prototype raised, baseline did not", ep))
    elif eb is not None and ep is None:
        base_only += 1
        mismatch.append((i, "baseline raised, prototype did not", eb))
    else:
        both_fail += 1
        errors[eb] = errors.get(eb, 0) + 1
        if eb.split(":")[0] != ep.split(":")[0]:
            mismatch.append((i, "both raised, different exception", f"{eb} | {ep}"))

print(f"\n{'=' * 70}\n{mode}: {N_DRAWS} prior draws\n{'=' * 70}")
print(f"  both succeeded                : {both_ok}")
print(f"  both raised (same kind)       : {both_fail}")
print(f"  ONLY prototype raised         : {prot_only}")
print(f"  ONLY baseline raised          : {base_only}")

if errors:
    print("\n  exceptions seen (identical on both paths):")
    for e, n in sorted(errors.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>5}x  {e}")

if mismatch:
    print(f"\n  MISMATCHES ({len(mismatch)}):")
    for i, what, detail in mismatch[:10]:
        print(f"    draw {i}: {what}\n      {detail}")
    print("\n  VERDICT: the prototype changes failure behaviour -- investigate.")
else:
    print("\n  VERDICT: no draw where the two paths differ. The failure, where it "
          "occurs, is a property of the parameter values and of lenstronomy's "
          "host-side root finder -- not of the jit wrapper.")
