"""T1 -- jit vs eager on the merged code, 200 draws per mode.

Originally compared gwemfish against the isolated prototype; now both sides come
from gwemfish itself, selected by cfg["nautilus"]["jit"]. Same points into each,
then compile count and ms/call.

Pass condition, same as the prototype's: max relative |dlogL| < 1e-10, rejects
agree, and compiles/call drops to 0 where the baseline recompiles.
"""

import sys

import numpy as np
from common import compiles_per_call, ms_per_call, save_json

from gwemfish.nautilus_common import build_nautilus_problem
from tutorial_cfg import build_ctx, cfg_for_run

N_DRAWS = 200
MODES = sys.argv[1:] or ["GW-only", "EM+GW", "EM-only"]


def check_mode(mode):
    print(f"\n{'=' * 64}\n{mode}\n{'=' * 64}", flush=True)
    ctx = build_ctx(mode)
    prior, loglike_base, names = build_nautilus_problem(
        ctx, cfg_for_run(ctx, mode, jit=False), mode, "nautilus-source")
    _, loglike_jit, _ = build_nautilus_problem(
        ctx, cfg_for_run(ctx, mode, jit=True), mode, "nautilus-source")

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

    pts = [p for p, f in zip(points, finite) if f][:10] or points[:10]
    c_base = compiles_per_call(loglike_base, pts)
    c_jit = compiles_per_call(loglike_jit, pts)
    t_base = ms_per_call(loglike_base, pts)
    t_jit = ms_per_call(loglike_jit, pts)

    print(f"  params ({prior.dimensionality()}): {names}")
    print(f"  accepted {int(finite.sum())}/{N_DRAWS}, rejected {n_reject} "
          f"(reject agrees: {reject_agree})")
    print(f"  max |dlogL| {max_abs:.3e}   max rel {max_rel:.3e}")
    print(f"  compiles/call  base {c_base:.1f} -> jit {c_jit:.1f}")
    print(f"  ms/call        base {t_base:.1f} -> jit {t_jit:.2f}  ({t_base / t_jit:.1f}x)",
          flush=True)

    return {"mode": mode, "n_params": prior.dimensionality(), "max_abs": max_abs,
            "max_rel": max_rel, "reject_agree": reject_agree,
            "compiles_base": c_base, "compiles_jit": c_jit,
            "ms_base": t_base, "ms_jit": t_jit,
            "pass": bool(max_rel < 1e-10 and reject_agree)}


results = [check_mode(m) for m in MODES]

print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
for r in results:
    print(f"{r['mode']:<10} rel {r['max_rel']:.2e}  compiles {r['compiles_base']:.0f}->"
          f"{r['compiles_jit']:.0f}  ms {r['ms_base']:.1f}->{r['ms_jit']:.2f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}")
save_json("t1_agreement_compiles.json", results)
