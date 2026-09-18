"""
Final 4-way comparison, all with the SAME corrected asymmetric y0gw/y1gw box and
Uniform(q) prior directly (not Uniform(e1)):
  1. fisher-source (gwemfish, q_phi)
  2. deriv-approx-source (gwemfish, q_phi)
  3. nautilus-source (gwemfish, q_phi, smoke n_eff=500 -> 65 samples)
  4. lenstronomy-nautilus-qphi (external, pure lenstronomy, n_eff=7003)
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


def load_samples(npz_path, json_path=None, json_key="samples_image_plane"):
    if os.path.isfile(npz_path):
        return {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    with open(json_path) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d[json_key].items()}


samples_fisher_raw = load_samples(os.path.join(OUT_DIR, "fisher_source", "samples.npz"))
samples_deriv_raw = load_samples(
    os.path.join(OUT_DIR, "deriv_approx_source", "samples.npz"),
    os.path.join(OUT_DIR, "deriv_approx_source", "pipeline_outputs_deriv_approx_source.json"),
)
samples_nautilus_gw_raw = load_samples(os.path.join(OUT_DIR, "nautilus_source", "samples.npz"))

ln_raw = np.load(os.path.join(LN_DIR, "samples.npz"))
w = np.asarray(ln_raw["weights"])
rng = np.random.default_rng(0)
idx = rng.choice(len(w), size=len(w), p=w / w.sum())
samples_ln = {k: np.asarray(ln_raw[k])[idx] for k in ln_raw.files if k != "weights"}

poco_raw = np.load(os.path.join(LN_DIR, "samples_pocomc_n4000_jitter_precondTrue.npz"))
wp = np.asarray(poco_raw["weights"])
idxp = rng.choice(len(wp), size=len(wp), p=wp / wp.sum())
samples_pocomc = {k: np.asarray(poco_raw[k])[idxp] for k in poco_raw.files if k != "weights"}

emcee_raw = np.load(os.path.join(LN_DIR, "samples_emcee_w64_s24630_jitter.npz"))
samples_emcee = {k: np.asarray(emcee_raw[k]) for k in emcee_raw.files if k != "weights"}

nessai_raw = np.load(os.path.join(LN_DIR, "samples_nessai_nlive1000_pool6.npz"))
samples_nessai = {k: np.asarray(nessai_raw[k]) for k in nessai_raw.files if k != "weights"}


def to_common(d):
    return {
        "gamma": d.get("gamma", d.get("lens0_gamma")),
        "q": d.get("q", d.get("lens0_q")),
        "T_star": d["T_star"], "dL": d["dL"],
        "y0gw": d["y0gw"], "y1gw": d["y1gw"],
    }


all_dicts = {
    "fisher-source": to_common(samples_fisher_raw),
    "deriv-approx-source": to_common(samples_deriv_raw),
    "nautilus-source (gwemfish, 65 smp)": to_common(samples_nautilus_gw_raw),
    "lenstronomy-nautilus-qphi (7003 n_eff)": to_common(samples_ln),
    "pocomc-lenstronomy (n4000 jitter, 12134 smp)": to_common(samples_pocomc),
    "emcee-lenstronomy (w64 s24630 jitter, 896 smp, NOT converged)": to_common(samples_emcee),
    "nessai-lenstronomy (nlive1000 pool6, 3802 smp)": to_common(samples_nessai),
}

TRUTHS = {
    "gamma": 2.0, "q": 0.6,
    "T_star": 14792489.407213762, "dL": 11213.719423532619,
    "y0gw": 0.02, "y1gw": 0.001,
}
shared_keys = sorted(TRUTHS.keys())

# deriv-approx-source's exact prior bounds (gw_only_deriv_approx_source_vs_nautilus_source_qphi.py)
# -- used here to fix every panel's axis range to the full prior box, rather than
# auto-scaling to each method's own posterior spread.
PARAM_RANGES = {
    "gamma": (0.8, 2.8),
    "q": ((1 - 0.48) / (1 + 0.48), (1 - 0.04) / (1 + 0.04)),
    "T_star": (2958497.8814427527, 29584978.814427525),
    "dL": (5606.859711766309, 21193.92971047665),
    "y0gw": (0.00375, 0.045),
    "y1gw": (1e-5, 0.003),
}

print("\n" + "=" * 130)
header = f"{'param':<8}"
for label in all_dicts:
    header += f"{label:>28}"
print(header)
print("-" * 130)
for k in shared_keys:
    row = f"{k:<8}"
    for label, d in all_dicts.items():
        row += f"{d[k].mean():>16.5g} +/- {d[k].std():<6.3g}"
    print(row)
print("=" * 130)

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
    save_path=os.path.join(OUT_DIR, "compare_all_seven_qphi_prior_bounds.png"),
)
print(f"\nSaved: {os.path.join(OUT_DIR, 'compare_all_seven_qphi_prior_bounds.png')}")
