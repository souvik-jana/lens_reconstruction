"""Smoke check after the builder merge: jit and eager agree, in-package.

Not the full verification -- 10 draws per mode, just enough to catch a wiring
mistake before the pool work goes in.
"""

import numpy as np
from common import compiles_per_call, ms_per_call

from gwemfish.nautilus_common import build_em_only_nautilus_problem
from gwemfish.nautilus_source_inference import (
    build_em_gw_source_plane_problem,
    build_gw_source_plane_problem,
)
from tutorial_cfg import build_ctx, cfg_for_run

BUILDERS = {"GW-only": build_gw_source_plane_problem,
            "EM+GW": build_em_gw_source_plane_problem,
            "EM-only": build_em_only_nautilus_problem}


def check(mode):
    print(f"\n{'=' * 60}\n{mode}\n{'=' * 60}", flush=True)
    ctx = build_ctx(mode)
    build = BUILDERS[mode]

    prior_e, loglike_e, names_e = build(ctx, cfg_for_run(ctx, mode, jit=False))
    prior_j, loglike_j, names_j = build(ctx, cfg_for_run(ctx, mode, jit=True))
    assert names_e == names_j, (names_e, names_j)

    rng = np.random.default_rng(0)
    points = [prior_e.unit_to_dictionary(u)
              for u in rng.uniform(size=(10, prior_e.dimensionality()))]
    a = np.array([loglike_e(p) for p in points])
    b = np.array([loglike_j(p) for p in points])
    ok = np.isfinite(a) & (a > -1e299)
    rel = float(np.max(np.abs((b[ok] - a[ok]) / a[ok]))) if ok.any() else 0.0

    print(f"  params ({len(names_e)}): {names_e}")
    print(f"  accepted {int(ok.sum())}/10, max rel diff {rel:.2e}")
    print(f"  compiles/call eager {compiles_per_call(loglike_e, points):.1f} -> "
          f"jit {compiles_per_call(loglike_j, points):.1f}")
    print(f"  ms/call       eager {ms_per_call(loglike_e, points):.1f} -> "
          f"jit {ms_per_call(loglike_j, points):.2f}")
    assert rel < 1e-10, f"{mode}: jit and eager disagree, rel {rel:.2e}"
    return len(names_e)


counts = {m: check(m) for m in BUILDERS}
print(f"\nparam counts: {counts}")
print("smoke PASS")
