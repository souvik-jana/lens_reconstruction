"""
Five q_phi comparison corner plots, all from the same "centre/full" system:
  1. fisher-source + deriv-approx-source (gwemfish) -- full auto-range
  2. fisher-source + deriv-approx-source (gwemfish) -- zoomed to deriv-approx-source's own extent
  3. fisher-source + deriv-approx-source + lenstronomy-nautilus -- within lenstronomy-nautilus's prior range
  4. fisher-source + deriv-approx-source + lenstronomy-nautilus + nessai (nlive1000 pool6, not smoke) -- same prior range
  5. lenstronomy-nautilus + nessai (nlive1000 pool6) -- full auto-range
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import plot_multi_comparison_corner

OUT_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi"
LN_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_lenstronomy_nautilus_qphi"

rng = np.random.default_rng(0)


def load_samples(npz_path, json_path=None, json_key="samples_image_plane"):
    if os.path.isfile(npz_path):
        return {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    with open(json_path) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d[json_key].items()}


def load_weighted(npz_path):
    raw = np.load(npz_path)
    w = np.asarray(raw["weights"])
    idx = rng.choice(len(w), size=len(w), p=w / w.sum())
    return {k: np.asarray(raw[k])[idx] for k in raw.files if k != "weights"}


def load_unweighted(npz_path):
    raw = np.load(npz_path)
    return {k: np.asarray(raw[k]) for k in raw.files if k != "weights"}


def to_common(d):
    return {
        "gamma": d.get("gamma", d.get("lens0_gamma")),
        "q": d.get("q", d.get("lens0_q")),
        "T_star": d["T_star"], "dL": d["dL"],
        "y0gw": d["y0gw"], "y1gw": d["y1gw"],
    }


samples_fisher = to_common(load_samples(os.path.join(OUT_DIR, "fisher_source", "samples.npz")))
samples_deriv = to_common(load_samples(
    os.path.join(OUT_DIR, "deriv_approx_source", "samples.npz"),
    os.path.join(OUT_DIR, "deriv_approx_source", "pipeline_outputs_deriv_approx_source.json"),
))
samples_ln = to_common(load_weighted(os.path.join(LN_DIR, "samples.npz")))
samples_nessai = to_common(load_unweighted(os.path.join(LN_DIR, "samples_nessai_nlive1000_pool6.npz")))

TRUTHS = {
    "gamma": 2.0, "q": 0.6,
    "T_star": 14792489.407213762, "dL": 11213.719423532619,
    "y0gw": 0.02, "y1gw": 0.001,
}
shared_keys = sorted(TRUTHS.keys())

# lenstronomy-nautilus's own prior box (same box deriv-approx-source/fisher-source were set up
# to match -- see gw_only_deriv_approx_source_vs_nautilus_source_qphi.py)
LN_PRIOR_RANGE = {
    "gamma": (0.8, 2.8),
    "q": ((1 - 0.48) / (1 + 0.48), (1 - 0.04) / (1 + 0.04)),
    "T_star": (2958497.8814427527, 29584978.814427525),
    "dL": (5606.859711766309, 21193.92971047665),
    "y0gw": (0.00375, 0.045),
    "y1gw": (1e-5, 0.003),
}

# deriv-approx-source's own extent, padded 5% each side, for the "zoomed" plot
DERIV_RANGE = {}
for k in shared_keys:
    v = samples_deriv[k]
    lo, hi = np.min(v), np.max(v)
    pad = 0.05 * (hi - lo)
    DERIV_RANGE[k] = (lo - pad, hi + pad)


def make_plot(title, dicts_by_label, param_ranges, save_name):
    labels = list(dicts_by_label.keys())
    colors = ["C0", "C1", "C2", "C3"][: len(labels)]
    save_path = os.path.join(OUT_DIR, save_name)
    plot_multi_comparison_corner(
        [dicts_by_label[l] for l in labels],
        {"all": shared_keys},
        labels=labels,
        colors=colors,
        truths_dict={"all": TRUTHS},
        param_ranges=param_ranges,
        hist_kwargs={"density": True},
        levels=[0.95],
        plot_datapoints=False, plot_density=False,
        fill_contours=False, no_fill_contours=True,
        save_path=save_path,
    )
    print(f"Saved [{title}]: {save_path}")


# 1. fisher-source + deriv-approx-source, full auto-range
make_plot(
    "1. fisher+deriv-approx, full range",
    {"fisher-source": samples_fisher, "deriv-approx-source": samples_deriv},
    param_ranges=None,
    save_name="qphi_1_fisher_deriv_full.png",
)

# 2. fisher-source + deriv-approx-source, zoomed to deriv-approx-source's extent
make_plot(
    "2. fisher+deriv-approx, zoomed to deriv-approx-source",
    {"fisher-source": samples_fisher, "deriv-approx-source": samples_deriv},
    param_ranges=DERIV_RANGE,
    save_name="qphi_2_fisher_deriv_zoom_deriv.png",
)

# 3. fisher-source + deriv-approx-source + lenstronomy-nautilus, within lenstronomy-nautilus's prior range
make_plot(
    "3. fisher+deriv-approx+lenstronomy-nautilus, LN prior range",
    {
        "fisher-source": samples_fisher,
        "deriv-approx-source": samples_deriv,
        "lenstronomy-nautilus": samples_ln,
    },
    param_ranges=LN_PRIOR_RANGE,
    save_name="qphi_3_fisher_deriv_lnnautilus.png",
)

# 4. fisher-source + deriv-approx-source + lenstronomy-nautilus + nessai, within lenstronomy-nautilus's prior range
make_plot(
    "4. fisher+deriv-approx+lenstronomy-nautilus+nessai, LN prior range",
    {
        "fisher-source": samples_fisher,
        "deriv-approx-source": samples_deriv,
        "lenstronomy-nautilus": samples_ln,
        "nessai (nlive1000 pool6)": samples_nessai,
    },
    param_ranges=LN_PRIOR_RANGE,
    save_name="qphi_4_fisher_deriv_lnnautilus_nessai.png",
)

# 5. lenstronomy-nautilus + nessai, full auto-range
make_plot(
    "5. lenstronomy-nautilus+nessai, full range",
    {
        "lenstronomy-nautilus": samples_ln,
        "nessai (nlive1000 pool6)": samples_nessai,
    },
    param_ranges=None,
    save_name="qphi_5_lnnautilus_nessai.png",
)
