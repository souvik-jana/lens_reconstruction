"""Compare the posteriors the three prototype variants produced.

They ran the same tutorial problem, the same full budget and the same seed, so
they should agree. This quantifies "agree" rather than asserting it from a plot:
for every parameter it reports each variant's median and 68% interval, and the
shift between variants in units of the posterior width -- which is the only scale
on which a difference is either negligible or not.

    python compare_tutorial_full.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "tutorial_full")
ORDER = ["D_jit_pool", "B_jit", "C_pool"]
COLORS = {"D_jit_pool": "C0", "B_jit": "C1", "C_pool": "C2"}

runs = {}
for path in sorted(glob.glob(os.path.join(OUT_DIR, "GW-only_*.npz"))):
    name = os.path.basename(path)[len("GW-only_"):-len(".npz")]
    runs[name] = dict(np.load(path))

if not runs:
    raise SystemExit(f"no posteriors in {OUT_DIR} -- run run_tutorial_full.py first")

names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]
print(f"variants found: {names}")

truths_path = os.path.join(OUT_DIR, "truths.json")
truths = json.load(open(truths_path)) if os.path.exists(truths_path) else {}

timings_path = os.path.join(OUT_DIR, "timings.json")
timings = json.load(open(timings_path)) if os.path.exists(timings_path) else {}

# Only parameters that were actually sampled; fixed ones are a delta function and
# would make every comparison trivially pass.
first = runs[names[0]]
params = [k for k in first if np.std(first[k]) > 0]
print(f"sampled parameters: {params}")

print(f"\n{'=' * 78}\nMARGINALS  (median [16%, 84%])\n{'=' * 78}")
for p in params:
    truth = truths.get(p)
    print(f"\n{p}" + (f"   truth = {truth:.6g}" if truth is not None else ""))
    for n in names:
        v = runs[n][p]
        lo, med, hi = np.percentile(v, [16, 50, 84])
        line = (f"  {n:<12} {med:12.6g}  [{lo:.6g}, {hi:.6g}]  "
                f"sigma={np.std(v):.3g}  N={len(v)}")
        if truth is not None:
            line += f"  bias={(med - truth) / np.std(v):+.2f}sigma"
        print(line)

print(f"\n{'=' * 78}\nAGREEMENT vs {names[0]}  (median shift / posterior sigma)\n{'=' * 78}")
ref = runs[names[0]]
print(f"  {'param':<18}" + "".join(f"{n:>14}" for n in names[1:]))
worst = 0.0
for p in params:
    sigma = np.std(ref[p])
    shifts = [abs(np.median(runs[n][p]) - np.median(ref[p])) / sigma for n in names[1:]]
    worst = max([worst] + shifts)
    print(f"  {p:<18}" + "".join(f"{s:>13.3f}s" for s in shifts))
print(f"\n  worst shift across all parameters: {worst:.3f} sigma")
print("  (Monte Carlo scatter alone gives ~1/sqrt(N_eff) ~ "
      f"{1 / np.sqrt(len(ref[params[0]])):.3f} sigma, so anything of that order "
      "is noise, not disagreement.)")

if timings:
    print(f"\n{'=' * 78}\nCOST\n{'=' * 78}")
    print(f"  {'variant':<12} {'wall':>11} {'n_like':>9} {'n_eff':>9} {'log_z':>11}")
    for n in names:
        d = timings.get(n)
        if d:
            print(f"  {n:<12} {d['wall_seconds']:>10.1f}s {d['n_like']:>9d} "
                  f"{d['n_eff']:>9.1f} {d['log_z']:>11.4f}")
    zs = [timings[n]["log_z"] for n in names if n in timings]
    if len(zs) > 1:
        print(f"\n  max |delta log_z| between variants: {max(zs) - min(zs):.4f}")

arrs = {n: np.column_stack([runs[n][p] for p in params]) for n in names}
fig = None
for n in names:
    fig = corner.corner(
        arrs[n], labels=params, color=COLORS.get(n, "C3"), fig=fig,
        truths=[truths.get(p) for p in params] if truths else None,
        truth_color="k", plot_datapoints=False, levels=(0.68, 0.95),
        hist_kwargs={"density": True},
    )
fig.legend(handles=[plt.Line2D([], [], color=COLORS.get(n, "C3"), label=n)
                    for n in names], loc="upper right", frameon=False)
path = os.path.join(OUT_DIR, "posterior_comparison.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nwrote {path}")
