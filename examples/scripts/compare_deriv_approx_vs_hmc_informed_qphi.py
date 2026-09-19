"""
Quick comparison corner: deriv-approx-source vs hmc-informed-source (both q_phi,
same system/priors as gw_only_deriv_approx_source_vs_nautilus_source_qphi.py),
using the already-saved samples.npz from each (no rerun). nautilus-source is
still running separately and is added to the comparison once it finishes.
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi")

DERIV_NPZ = os.path.join(OUTPUT_DIR, "deriv_approx_source", "samples.npz")
DERIV_JSON = os.path.join(OUTPUT_DIR, "deriv_approx_source", "pipeline_outputs_deriv_approx_source.json")
HMC_NPZ = os.path.join(OUTPUT_DIR, "hmc_informed_source", "samples.npz")
FISHER_NPZ = os.path.join(OUTPUT_DIR, "fisher_source", "samples.npz")


def load_samples(npz_path, json_path=None, json_key="samples_image_plane"):
    if os.path.isfile(npz_path):
        return {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    with open(json_path) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d[json_key].items()}


# deriv-approx-source ran before the samples.npz safety-save was added to the
# comparison script -- fall back to the pipeline JSON it did write.
samples_deriv = load_samples(DERIV_NPZ, DERIV_JSON)
samples_hmc = load_samples(HMC_NPZ)
samples_fisher = load_samples(FISHER_NPZ)

# Same truth system as gw_only_deriv_approx_source_vs_nautilus_source_qphi.py.
TRUTHS = {
    "lens0_gamma": 2.0,
    "lens0_q": 0.6,
    "lens0_e1": 0.25,
    "T_star": 14792489.407213762,
    "dL": 11213.719423532619,
    "y0gw": 0.02,
    "y1gw": 0.001,
}

all_sample_dicts = {
    "deriv-approx-source": samples_deriv,
    "hmc-informed-source (smoke)": samples_hmc,
    "fisher-source": samples_fisher,
}
shared_keys = sorted(set.intersection(*(set(d.keys()) for d in all_sample_dicts.values())))
for label, d in all_sample_dicts.items():
    print(f"{label} keys:", sorted(d.keys()))
print("shared (plotted) keys:", shared_keys)

print("\n" + "=" * 100)
print(f"{'param':<14}{'deriv-approx-source':>22}{'hmc-informed-source':>24}{'fisher-source':>22}")
print("-" * 100)
for k in shared_keys:
    d, h, f = samples_deriv[k], samples_hmc[k], samples_fisher[k]
    truth_str = f"  (truth={TRUTHS[k]:.6g})" if k in TRUTHS else ""
    print(f"{k:<14}{d.mean():>13.6g} +/- {d.std():<6.3g}"
          f"{h.mean():>15.6g} +/- {h.std():<6.3g}"
          f"{f.mean():>13.6g} +/- {f.std():<6.3g}{truth_str}")
print("=" * 100)

labels = list(all_sample_dicts.keys())
plot_multi_comparison_corner(
    [all_sample_dicts[l] for l in labels],
    {"all": shared_keys},
    labels=labels,
    colors=["C0", "C1", "C2"],
    truths_dict={"all": {k: TRUTHS[k] for k in shared_keys if k in TRUTHS}},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_deriv_approx_vs_hmc_informed_vs_fisher_qphi.png"),
)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'compare_deriv_approx_vs_hmc_informed_vs_fisher_qphi.png')}")
