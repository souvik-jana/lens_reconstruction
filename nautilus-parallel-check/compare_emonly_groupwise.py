"""EM-only: nautilus vs fisher vs deriv-approx(informed), one corner per group.

23 parameters in a single corner is unreadable, so this splits them the way
gwemfish already groups them -- lens mass, source light, lens light, noise --
using create_default_param_groups and plot_multi_comparison_corner rather than
a hand-rolled layout.

deriv-approx here is the informed=True run. The plain-NUTS one is excluded on
purpose: it did not converge (r_hat ~1e15, sigmas off by 5.5e8x), so plotting it
would just be three panels of noise.

    python compare_emonly_groupwise.py
"""

import json
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from gwemfish.corner_plot_utils import (
    create_default_param_groups,
    plot_multi_comparison_corner,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_only")

SOURCES = [
    ("nautilus-source", "EM-only_D_jit_pool.npz", "C0"),
    ("fisher", "EM-only_fisher.npz", "C4"),
    ("deriv-approx (informed)", "EM-only_deriv-approx-informed.npz", "C3"),
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

truths = json.load(open(os.path.join(OUT_DIR, "truths.json")))

# Only parameters every method sampled and that actually vary -- a parameter one
# method holds fixed would show as a spike and make the panel unreadable.
shared = [k for k in samples_dicts[0]
          if all(k in s and np.std(s[k]) > 0 for s in samples_dicts)]
groups = create_default_param_groups({k: samples_dicts[0][k] for k in shared})

print(f"methods: {labels}")
for name, params in groups.items():
    print(f"  {name:<20} {len(params)} params: {params}")

# Keyed by GROUP name, not by method label: corner_plot_utils.py:727 looks up
# truths_dict.get(group_name). Keying it by label makes that lookup return None
# and the truth lines silently vanish.
truths_dict = {group: {p: truths[p] for p in params if p in truths}
               for group, params in groups.items()}

plot_multi_comparison_corner(
    samples_dicts=samples_dicts,
    param_groups=groups,
    labels=labels,
    colors=colors,
    truths_dict=truths_dict,
    # Default truth_color is 'red', which is invisible against the red
    # deriv-approx contours.
    truth_color="k",
    save_path=os.path.join(OUT_DIR, "groupwise_{group_name}.png"),
    plot_datapoints=False,
    levels=(0.68, 0.95),
    # density=True normalises each 1D histogram to unit area (corner forwards
    # hist_kwargs straight to ax.hist, corner/core.py:233). Without it the panels
    # show raw counts, so the three methods' peak heights differ purely because
    # they have different sample counts -- nautilus 12750, fisher 5000 -- which
    # looks like a disagreement in the posteriors and is not one.
    hist_kwargs={"density": True},
)

print(f"\n{'=' * 92}\nWIDTH vs NAUTILUS, BY GROUP  (sigma_method / sigma_nautilus)\n{'=' * 92}")
ref = samples_dicts[0]
for name, params in groups.items():
    print(f"\n{name}")
    print(f"  {'param':<22}" + "".join(f"{lb:>26}" for lb in labels[1:]))
    for p in params:
        r = np.std(ref[p])
        print(f"  {p:<22}" + "".join(
            f"{np.std(s[p]) / r:>25.3f}x" for s in samples_dicts[1:]))

print(f"\nwrote {OUT_DIR}/groupwise_<group>.png")
