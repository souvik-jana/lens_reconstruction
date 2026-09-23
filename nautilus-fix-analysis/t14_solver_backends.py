"""Does cfg['nautilus']['jit'] work with every solver backend, not just analytical?

Everything so far used jaxtronomy + 'analytical'. The compiled core wraps
solver.solve in jax.jit, which only works if the solver is traceable. Checks the
three choices on the tutorial GW-only system: jit vs eager agreement, compiles
per call, and ms per call. A backend that cannot be traced fails at the warm-up
call of the jit build, which this reports rather than hides.
"""

import numpy as np
from common import compiles_per_call, ms_per_call, save_json

from gwemfish.nautilus_common import build_nautilus_problem
from tutorial_cfg import build_ctx, cfg_for_run

N_DRAWS = 20
BACKENDS = [("jaxtronomy", "analytical"), ("jaxtronomy", "lenstronomy"), ("helens", None)]


def solver_cfg(backend, solver):
    sp = {"backend": backend}
    if solver:
        sp["jaxtronomy"] = {"solver": solver}
    return sp


results = {}
for backend, solver in BACKENDS:
    label = f"{backend}/{solver}" if solver else backend
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    ctx = build_ctx("GW-only")
    cfg_e = cfg_for_run(ctx, "GW-only", jit=False)
    cfg_j = cfg_for_run(ctx, "GW-only", jit=True)
    for c in (cfg_e, cfg_j):
        c["gw"]["solver_params"] = solver_cfg(backend, solver)

    prior, eager, _ = build_nautilus_problem(ctx, cfg_e, "GW-only", "nautilus-source")
    try:
        _, jitted, _ = build_nautilus_problem(ctx, cfg_j, "GW-only", "nautilus-source")
    except Exception as exc:
        results[label] = {"jit_builds": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  jit build FAILED: {type(exc).__name__}: {exc}")
        continue

    rng = np.random.default_rng(0)
    points = [prior.unit_to_dictionary(u)
              for u in rng.uniform(size=(N_DRAWS, prior.dimensionality()))]
    try:
        a = np.array([eager(p) for p in points])
        b = np.array([jitted(p) for p in points])
    except Exception as exc:
        results[label] = {"jit_builds": True, "jit_calls": False,
                          "error": f"{type(exc).__name__}: {exc}"}
        print(f"  jit call FAILED: {type(exc).__name__}: {exc}")
        continue

    ok = np.isfinite(a) & (a > -1e299)
    rel = float(np.max(np.abs((b[ok] - a[ok]) / a[ok]))) if ok.any() else float("nan")
    rejects_agree = bool(np.array_equal(a <= -1e299, b <= -1e299))
    pts = [p for p, f in zip(points, ok) if f][:10] or points[:10]
    r = {"jit_builds": True, "jit_calls": True, "accepted": int(ok.sum()),
         "max_rel": rel, "rejects_agree": rejects_agree,
         "compiles_eager": compiles_per_call(eager, pts),
         "compiles_jit": compiles_per_call(jitted, pts),
         "ms_eager": ms_per_call(eager, pts), "ms_jit": ms_per_call(jitted, pts)}
    r["pass"] = bool(rel < 1e-10 and rejects_agree)
    results[label] = r
    print(f"  accepted {r['accepted']}/{N_DRAWS}  max rel {rel:.2e}  rejects agree {rejects_agree}")
    print(f"  compiles/call {r['compiles_eager']:.1f} -> {r['compiles_jit']:.1f}")
    print(f"  ms/call {r['ms_eager']:.1f} -> {r['ms_jit']:.2f}  "
          f"({r['ms_eager'] / r['ms_jit']:.1f}x)")

print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
for label, r in results.items():
    if not r.get("jit_calls"):
        print(f"{label:<26} FAIL  {r['error'][:90]}")
        continue
    print(f"{label:<26} rel {r['max_rel']:.1e}  compiles {r['compiles_eager']:.0f}->"
          f"{r['compiles_jit']:.0f}  ms {r['ms_eager']:.1f}->{r['ms_jit']:.2f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}")
save_json("t14_solver_backends.json", results)
