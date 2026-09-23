"""Save every posterior we have: comparison overlays and nautilus alone.

Comparison figures answer "do the methods agree"; the nautilus-only figures are
the ones to read the actual measurement off, without the Fisher Gaussian widening
every panel.
"""

import os

import numpy as np
from common import OUT, save_json

from gwemfish import plot_posterior
from gwemfish.corner_plot_utils import plot_multi_comparison_corner

PLOT_DIR = os.path.join(OUT, "posteriors")
RUNS = {
    "fisher-source": (os.path.join(OUT, "t9_jit_pool4_span1", "samples_fisher_source.npz"),
                      "steelblue"),
    "nautilus jit (serial)": (os.path.join(OUT, "t9_jit_span1", "samples_nautilus_source.npz"),
                              "seagreen"),
    "nautilus jit + pool4": (os.path.join(OUT, "t9_jit_pool4_span1", "samples_nautilus_source.npz"),
                             "darkorange"),
}
TRUTHS = {"T_star": 7472713.020600661, "dL": 15946.707596842625,
          "lens0_q": 0.8, "y0gw": 0.02, "y1gw": 1e-05}
SOURCE = ["y0gw", "y1gw"]

os.makedirs(PLOT_DIR, exist_ok=True)


def load(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


samples = {label: load(path) for label, (path, _) in RUNS.items()}
shared = sorted(set.intersection(*(set(s) for s in samples.values())))
groups = {"all": shared, "source": SOURCE}

# 1. everything overlaid
plot_multi_comparison_corner(
    list(samples.values()), groups,
    labels=list(samples), colors=[c for _, c in RUNS.values()],
    truths_dict=TRUTHS,
    save_path=os.path.join(PLOT_DIR, "comparison_all_methods_{group_name}.png"))

# 2. the two nautilus runs against each other, no Fisher Gaussian in the way
nautilus_labels = [k for k in samples if k.startswith("nautilus")]
plot_multi_comparison_corner(
    [samples[k] for k in nautilus_labels], groups,
    labels=nautilus_labels, colors=[RUNS[k][1] for k in nautilus_labels],
    truths_dict=TRUTHS,
    save_path=os.path.join(PLOT_DIR, "nautilus_serial_vs_pool_{group_name}.png"))

# 3. each nautilus run on its own, full corner and the source-localization pair
for label in nautilus_labels:
    tag = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    for mode, params, name in (("combined", shared, "full"),
                               ("subset", SOURCE, "source")):
        plot_posterior(samples[label], TRUTHS, cfg={
            "plot": {"plot_mode": mode, "params_to_plot": params,
                     "color": RUNS[label][1], "show_titles": True,
                     "save_path": f"{tag}_{name}.png"},
            "output": {"output_dir": PLOT_DIR}})

written = sorted(os.listdir(PLOT_DIR))
print("\n".join(f"  {f}" for f in written))
save_json("t13_posteriors.json", {
    "plot_dir": PLOT_DIR, "files": written, "shared_params": shared,
    "n_samples": {k: int(len(next(iter(v.values())))) for k, v in samples.items()}})
