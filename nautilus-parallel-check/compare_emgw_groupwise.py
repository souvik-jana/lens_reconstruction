"""EM+GW: nautilus vs fisher-source vs deriv-approx-source, one corner per group.

Same treatment as the EM-only comparison: gwemfish's own parameter grouping via
create_default_param_groups, truth lines keyed by GROUP name (corner_plot_utils.py:727
looks up truths_dict.get(group_name) -- keying by method label silently drops them),
and density-normalised 1D histograms so the three methods' peak heights are
comparable rather than reflecting their different sample counts.

EM+GW should show the GW source position much better constrained than EM-only
managed, since the time delays break the degeneracy the EM data alone cannot.

    python compare_emgw_groupwise.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from gwemfish.corner_plot_utils import (
    create_default_param_groups,
    plot_multi_comparison_corner,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_gw")

SOURCES = [
    ("nautilus-source", "EM_GW_D_jit_pool.npz", "C0"),
    ("fisher-source", "EM_GW_fisher-source.npz", "C4"),
    ("deriv-approx-source (informed)", "EM_GW_deriv-approx-source.npz", "C3"),
]

samples_dicts, labels, colors = [], [], []
for label, fname, color in SOURCES:
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"skipping {label}: {fname} not found")
        continue
    samples_dicts.append({k: np.asarray(v) for k, v in np.load(path).items()})
    labels.append(label)
    colors.append(color)

if not samples_dicts:
    raise SystemExit(f"no posteriors in {OUT_DIR} -- run run_emgw_full.py first")

truths = json.load(open(os.path.join(OUT_DIR, "truths.json")))

shared = [k for k in samples_dicts[0]
          if all(k in s and np.std(s[k]) > 0 for s in samples_dicts)]
groups = create_default_param_groups({k: samples_dicts[0][k] for k in shared})

print(f"methods: {labels}")
for name, params in groups.items():
    print(f"  {name:<22} {len(params)} params: {params}")

truths_dict = {group: {p: truths[p] for p in params if p in truths}
               for group, params in groups.items()}

plot_multi_comparison_corner(
    samples_dicts=samples_dicts,
    param_groups=groups,
    labels=labels,
    colors=colors,
    truths_dict=truths_dict,
    truth_color="k",
    save_path=os.path.join(OUT_DIR, "groupwise_{group_name}.png"),
    plot_datapoints=False,
    levels=(0.68, 0.95),
    hist_kwargs={"density": True},
)

print(f"\n{'=' * 96}\nWIDTH vs NAUTILUS, BY GROUP  (sigma_method / sigma_nautilus)\n{'=' * 96}")
ref = samples_dicts[0]
for name, params in groups.items():
    print(f"\n{name}")
    print(f"  {'param':<22}" + "".join(f"{lb:>34}" for lb in labels[1:]))
    for p in params:
        r = np.std(ref[p])
        print(f"  {p:<22}" + "".join(
            f"{np.std(s[p]) / r:>33.3f}x" for s in samples_dicts[1:]))

print(f"\n{'=' * 96}\nBIAS vs TRUTH  (median - truth, in units of that method's sigma)\n{'=' * 96}")
print(f"  {'param':<22}" + "".join(f"{lb:>34}" for lb in labels))
for p in shared:
    if p not in truths:
        continue
    cells = []
    for s in samples_dicts:
        sd = np.std(s[p])
        cells.append((np.median(s[p]) - truths[p]) / sd if sd > 0 else np.nan)
    print(f"  {p:<22}" + "".join(f"{c:>32.2f}s" for c in cells))

print(f"\nwrote {OUT_DIR}/groupwise_<group>.png")
