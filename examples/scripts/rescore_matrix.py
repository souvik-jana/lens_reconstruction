"""Re-score the coverage matrix from saved .npz, without re-running anything.

Two things the first scoring pass got wrong, both about what a *correct* run
looks like rather than about the code under test:

1. With phi fixed, add_qphi_columns_to_samples cannot backfill e1/e2 -- it needs
   both q and phi in the samples dict. Absent e1/e2 is correct there, not a fail.
2. e1/e2 presence is method-dependent even so: hmc returns them because the
   numpyro model registers them as deterministic sites and get_samples() returns
   those, while fisher/deriv-approx return only free parameters. Both are
   correct; what matters is that when they ARE present they satisfy the identity
   -- evaluated against the fixed phi where phi is not sampled.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_coverage")
PREFIX = "lens0"
TOL = 1e-12
PHI_FREE = ("EM-only", "EM+GW")

TRUTH = json.load(open(os.path.join(OUT_DIR, "truth.json")))
PHI_TRUTH = float(TRUTH[f"{PREFIX}_phi"])
Q_TRUTH = float(TRUTH[f"{PREFIX}_q"])


def identity_error(s, phi_free):
    """max |e1 - (1-q)/(1+q)cos(2phi)| ; phi from samples if free, else truth."""
    if f"{PREFIX}_q" not in s or f"{PREFIX}_e1" not in s:
        return None
    q = np.asarray(s[f"{PREFIX}_q"], dtype=float)
    phi = np.asarray(s[f"{PREFIX}_phi"], dtype=float) if phi_free else PHI_TRUTH
    a = (1.0 - q) / (1.0 + q)
    de1 = np.abs(np.asarray(s[f"{PREFIX}_e1"], dtype=float) - a * np.cos(2 * phi))
    de2 = np.abs(np.asarray(s[f"{PREFIX}_e2"], dtype=float) - a * np.sin(2 * phi))
    return float(max(de1.max(), de2.max()))


rows = []
for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.npz"))):
    name = os.path.basename(path)[:-4]
    label, method = name.split("_", 1)
    s = dict(np.load(path))
    phi_free = label in PHI_FREE

    has_q = f"{PREFIX}_q" in s
    has_phi = f"{PREFIX}_phi" in s
    has_e = f"{PREFIX}_e1" in s and f"{PREFIX}_e2" in s
    ident = identity_error(s, phi_free)

    q = np.asarray(s.get(f"{PREFIX}_q", []), dtype=float)
    q_ok = bool(q.size and q.min() >= 0.01 and q.max() <= 1.0)

    # flow: q must be sampled; phi must be sampled iff it is free; and any e1/e2
    # that IS present must satisfy the identity.
    flow = has_q and (has_phi == phi_free)
    if has_e:
        flow = flow and ident is not None and ident < TOL
    elif phi_free:
        flow = False  # free phi must produce backfilled e1/e2

    bias = None
    if has_q and q.size and np.std(q) > 0:
        bias = float((np.median(q) - Q_TRUTH) / np.std(q))

    rows.append(dict(label=label, method=method, phi_free=phi_free, flow=flow,
                     has_q=has_q, has_phi=has_phi, has_e=has_e, ident=ident,
                     q_ok=q_ok, bias=bias, n=int(q.size)))

order = {"GW-only-A": 0, "GW-only-B": 1, "EM-only": 2, "EM+GW": 3}
rows.sort(key=lambda r: (order.get(r["label"], 9), r["method"]))

f = lambda v, sp=".2f": "-" if v is None else format(v, sp)
out = ["# q/phi coverage matrix", "",
       f"Truth: q={Q_TRUTH:.4f}, phi={PHI_TRUTH:+.4f} rad "
       f"({np.degrees(PHI_TRUTH):+.1f} deg). 4-image system, 7 GW observables.", "",
       "`phi col` and `e1/e2` are *expected* to differ by method: with phi fixed the",
       "backfill cannot run, and only hmc returns e1/e2 (as numpyro.deterministic",
       "sites). PASS means the output matches what is correct for that combination,",
       "and any e1/e2 present satisfies e1=(1-q)/(1+q)cos(2phi).", "",
       "| cell | method | phi | flow | q | phi col | e1/e2 | identity | q in range | bias_q | N |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
y = lambda b: "y" if b else "n"
for r in rows:
    out.append(f"| {r['label']} | `{r['method']}` | {'free' if r['phi_free'] else 'fixed'} | "
               f"{'PASS' if r['flow'] else '**FAIL**'} | {y(r['has_q'])} | {y(r['has_phi'])} | "
               f"{y(r['has_e'])} | {f(r['ident'], '.1e')} | {y(r['q_ok'])} | "
               f"{f(r['bias'])} | {r['n']} |")
n_pass = sum(r["flow"] for r in rows)
out += ["", f"**{n_pass}/{len(rows)} cells pass.**"]

open(os.path.join(OUT_DIR, "matrix.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
