"""
True apples-to-apples: gwemfish deriv-approx-source (q_phi, corrected asymmetric
box) vs lenstronomy_nautilus_qphi.py (q free via Uniform(q), phi fixed, same
asymmetric box, n_eff=7003). Both sample Uniform(q) directly now -- the earlier
comparison against the original lenstronomy_nautilus used Uniform(e1) instead.
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

DERIV_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi/deriv_approx_source"
LN_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_lenstronomy_nautilus_qphi"
OUT_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi"

DERIV_NPZ = os.path.join(DERIV_DIR, "samples.npz")
DERIV_JSON = os.path.join(DERIV_DIR, "pipeline_outputs_deriv_approx_source.json")
LN_NPZ = os.path.join(LN_DIR, "samples.npz")


def load_samples(npz_path, json_path=None, json_key="samples_image_plane"):
    if os.path.isfile(npz_path):
        return {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    with open(json_path) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d[json_key].items()}


samples_deriv_raw = load_samples(DERIV_NPZ, DERIV_JSON)
ln_raw = np.load(LN_NPZ)
w = np.asarray(ln_raw["weights"])
rng = np.random.default_rng(0)
idx = rng.choice(len(w), size=len(w), p=w / w.sum())
samples_ln = {k: np.asarray(ln_raw[k])[idx] for k in ln_raw.files if k != "weights"}

# Rename gwemfish's lens0_gamma/lens0_q -> gamma/q to match lenstronomy_nautilus_qphi.
samples_deriv = {
    "gamma": samples_deriv_raw["lens0_gamma"],
    "q": samples_deriv_raw["lens0_q"],
    "T_star": samples_deriv_raw["T_star"],
    "dL": samples_deriv_raw["dL"],
    "y0gw": samples_deriv_raw["y0gw"],
    "y1gw": samples_deriv_raw["y1gw"],
}

TRUTHS = {
    "gamma": 2.0, "q": 0.6,
    "T_star": 14792489.407213762, "dL": 11213.719423532619,
    "y0gw": 0.02, "y1gw": 0.001,
}

shared_keys = sorted(set(samples_deriv) & set(samples_ln))
print("shared keys:", shared_keys)

print("\n" + "=" * 90)
print(f"{'param':<10}{'deriv-approx-source':>24}{'lenstronomy-nautilus-qphi':>28}")
print("-" * 90)
for k in shared_keys:
    d, n = samples_deriv[k], samples_ln[k]
    truth_str = f"  (truth={TRUTHS[k]:.6g})" if k in TRUTHS else ""
    print(f"{k:<10}{d.mean():>15.6g} +/- {d.std():<6.3g}"
          f"{n.mean():>19.6g} +/- {n.std():<6.3g}{truth_str}")
print("=" * 90)

plot_multi_comparison_corner(
    [samples_deriv, samples_ln],
    {"all": shared_keys},
    labels=["deriv-approx-source (gwemfish, q_phi)", "lenstronomy-nautilus-qphi (external, Uniform(q))"],
    colors=["C1", "C2"],
    truths_dict={"all": TRUTHS},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUT_DIR, "compare_deriv_approx_vs_lenstronomy_nautilus_qphi_v2.png"),
)
print(f"\nSaved: {os.path.join(OUT_DIR, 'compare_deriv_approx_vs_lenstronomy_nautilus_qphi_v2.png')}")
