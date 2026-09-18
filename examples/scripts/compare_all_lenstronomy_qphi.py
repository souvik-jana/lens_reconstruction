"""
Lenstronomy-only comparison: every sampler run against the SAME pure-lenstronomy
likelihood/prior (no gwemfish) for the centre/full q_phi system (q free, phi fixed).
  1. nautilus (ground truth, n_eff=7003)
  2. emcee (w64 s8000 jitter, NOT converged)
  3. emcee (w64 s24630 jitter, NOT converged)
  4. pocomc (n4000 jitter, precondition=True)
  5. pocomc (n4000 jitter, precondition=False)
  6. nessai (nlive200 smoke)
  7. nessai (nlive1000 pool6, converged)
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import plot_multi_comparison_corner

LN_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_lenstronomy_nautilus_qphi"

rng = np.random.default_rng(0)


def load_weighted(npz_path):
    raw = np.load(npz_path)
    w = np.asarray(raw["weights"])
    idx = rng.choice(len(w), size=len(w), p=w / w.sum())
    return {k: np.asarray(raw[k])[idx] for k in raw.files if k != "weights"}


def load_unweighted(npz_path):
    raw = np.load(npz_path)
    return {k: np.asarray(raw[k]) for k in raw.files if k != "weights"}


all_dicts = {
    "nautilus (7003 n_eff)": load_weighted(os.path.join(LN_DIR, "samples.npz")),
    "emcee (w64 s8000 jitter, NOT converged)": load_unweighted(os.path.join(LN_DIR, "samples_emcee_w64_s8000_jitter.npz")),
    "emcee (w64 s24630 jitter, NOT converged)": load_unweighted(os.path.join(LN_DIR, "samples_emcee_w64_s24630_jitter.npz")),
    "pocomc (n4000 jitter, precondTrue)": load_weighted(os.path.join(LN_DIR, "samples_pocomc_n4000_jitter_precondTrue.npz")),
    "pocomc (n4000 jitter, precondFalse)": load_weighted(os.path.join(LN_DIR, "samples_pocomc_n4000_jitter_precondFalse.npz")),
    "nessai (nlive200 smoke)": load_unweighted(os.path.join(LN_DIR, "samples_nessai_nlive200.npz")),
    "nessai (nlive1000 pool6)": load_unweighted(os.path.join(LN_DIR, "samples_nessai_nlive1000_pool6.npz")),
}

TRUTHS = {
    "gamma": 2.0, "q": 0.6,
    "T_star": 14792489.407213762, "dL": 11213.719423532619,
    "y0gw": 0.02, "y1gw": 0.001,
}
shared_keys = sorted(TRUTHS.keys())

PARAM_RANGES = {
    "gamma": (0.8, 2.8),
    "q": ((1 - 0.48) / (1 + 0.48), (1 - 0.04) / (1 + 0.04)),
    "T_star": (2958497.8814427527, 29584978.814427525),
    "dL": (5606.859711766309, 21193.92971047665),
    "y0gw": (0.00375, 0.045),
    "y1gw": (1e-5, 0.003),
}

print("\n" + "=" * 150)
header = f"{'param':<8}"
for label in all_dicts:
    header += f"{label:>30}"
print(header)
print("-" * 150)
for k in shared_keys:
    row = f"{k:<8}"
    for label, d in all_dicts.items():
        row += f"{d[k].mean():>18.5g} +/- {d[k].std():<7.3g}"
    print(row)
print("=" * 150)

labels = list(all_dicts.keys())
plot_multi_comparison_corner(
    [all_dicts[l] for l in labels],
    {"all": shared_keys},
    labels=labels,
    colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
    truths_dict={"all": TRUTHS},
    param_ranges=PARAM_RANGES,
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(LN_DIR, "compare_all_lenstronomy_qphi.png"),
)
print(f"\nSaved: {os.path.join(LN_DIR, 'compare_all_lenstronomy_qphi.png')}")
