"""
Apples-to-apples (with one noted caveat) 3-way comparison for the "centre" EPL
system: gwemfish fisher-source, gwemfish deriv-approx-source (both q_phi), vs
the external lenstronomy+nautilus run in lensing-degeneracies/gw-only-analysis/
geometry-analysis-reparam/runs/centre/full/lenstronomy_nautilus/ (loaded, not
re-run).

Verified matching settings (lenstronomy_nautilus/cfg.json vs our runs): theta_E=1.0,
gamma bounds (0.8,2.8), e1 bounds (0.04,0.48) [-> our q bounds via q=(1-e1)/(1+e1)
at phi=0], T_star/dL bounds, frac_td=0.002, frac_dL_eff=0.1, truths.

One caveat: lenstronomy_nautilus uses an ASYMMETRIC y0gw=(0.00375,0.045) /
y1gw=(1e-5,0.003) box; our deriv-approx-source run used a SYMMETRIC +/-0.02
truth-centered box (deriv-approx-source's source_box_half_width mechanism cannot
express an asymmetric box). fisher-source is unaffected (its Gaussian draw ignores
prior bounds entirely). Truth sits well inside both boxes and the posterior is far
narrower than either, so this is not expected to matter in practice -- called out
here rather than silently assumed identical.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from herculens.Util.param_util import phi_q2_ellipticity

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import plot_multi_comparison_corner

OUTPUT_DIR = "/Users/souvikjana/Documents/lens_reconstruction/examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi"
LENSTRONOMY_NAUTILUS_NPZ = (
    "/Users/souvikjana/Documents/lensing-degeneracies/gw-only-analysis/"
    "geometry-analysis-reparam/runs/centre/full/lenstronomy_nautilus/samples.npz"
)

FISHER_NPZ = os.path.join(OUTPUT_DIR, "fisher_source", "samples.npz")
DERIV_NPZ = os.path.join(OUTPUT_DIR, "deriv_approx_source", "samples.npz")
DERIV_JSON = os.path.join(OUTPUT_DIR, "deriv_approx_source", "pipeline_outputs_deriv_approx_source.json")


def load_samples(npz_path, json_path=None, json_key="samples_image_plane"):
    if os.path.isfile(npz_path):
        return {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    with open(json_path) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d[json_key].items()}


samples_fisher_raw = load_samples(FISHER_NPZ)
samples_deriv_raw = load_samples(DERIV_NPZ, DERIV_JSON)

# lenstronomy_nautilus: weighted nested-sampling points -> equal-weight resample.
ln_raw = np.load(LENSTRONOMY_NAUTILUS_NPZ)
w = np.asarray(ln_raw["weights"])
rng = np.random.default_rng(0)
idx = rng.choice(len(w), size=len(w), p=w / w.sum())
samples_lenstronomy_nautilus = {k: np.asarray(ln_raw[k])[idx] for k in ln_raw.files if k != "weights"}

PHI_TRUTH = 0.0


def to_common_naming(samples, q_key="lens0_q", gamma_key="lens0_gamma"):
    """Rename our gwemfish (lens0_*) keys to the lenstronomy_nautilus convention
    (bare gamma/e1/T_star/dL/y0gw/y1gw), computing e1 from q at the known fixed
    phi=0.0 truth (both our runs fix lens0_phi to exactly this value)."""
    e1, _ = phi_q2_ellipticity(PHI_TRUTH, np.asarray(samples[q_key]))
    return {
        "gamma": np.asarray(samples[gamma_key]),
        "e1": np.asarray(e1),
        "T_star": np.asarray(samples["T_star"]),
        "dL": np.asarray(samples["dL"]),
        "y0gw": np.asarray(samples["y0gw"]),
        "y1gw": np.asarray(samples["y1gw"]),
    }


samples_fisher = to_common_naming(samples_fisher_raw)
samples_deriv = to_common_naming(samples_deriv_raw)

TRUTHS = {
    "gamma": 2.0, "e1": 0.25,
    "T_star": 14792489.407213762, "dL": 11213.719423532619,
    "y0gw": 0.02, "y1gw": 0.001,
}

all_dicts = {
    "fisher-source (gwemfish, q_phi)": samples_fisher,
    "deriv-approx-source (gwemfish, q_phi)": samples_deriv,
    "lenstronomy-nautilus (external)": samples_lenstronomy_nautilus,
}
shared_keys = sorted(set.intersection(*(set(d.keys()) for d in all_dicts.values())))
print("shared keys:", shared_keys)

print("\n" + "=" * 110)
print(f"{'param':<10}{'fisher-source':>20}{'deriv-approx-source':>24}{'lenstronomy-nautilus':>24}")
print("-" * 110)
for k in shared_keys:
    f, d, n = samples_fisher[k], samples_deriv[k], samples_lenstronomy_nautilus[k]
    truth_str = f"  (truth={TRUTHS[k]:.6g})" if k in TRUTHS else ""
    print(f"{k:<10}{f.mean():>13.6g} +/- {f.std():<5.3g}"
          f"{d.mean():>17.6g} +/- {d.std():<5.3g}"
          f"{n.mean():>17.6g} +/- {n.std():<5.3g}{truth_str}")
print("=" * 110)

labels = list(all_dicts.keys())
plot_multi_comparison_corner(
    [all_dicts[l] for l in labels],
    {"all": shared_keys},
    labels=labels,
    colors=["C0", "C1", "C2"],
    truths_dict={"all": TRUTHS},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_fisher_deriv_approx_lenstronomy_nautilus_qphi.png"),
)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'compare_fisher_deriv_approx_lenstronomy_nautilus_qphi.png')}")

# --------------------------------------------------------------------------
# Second corner: deriv-approx-source vs lenstronomy-nautilus only (drop
# fisher-source, whose unbounded Gaussian dominates the axis scale and hides
# the finer agreement between the other two).
# --------------------------------------------------------------------------
labels_2 = ["deriv-approx-source (gwemfish, q_phi)", "lenstronomy-nautilus (external)"]
plot_multi_comparison_corner(
    [samples_deriv, samples_lenstronomy_nautilus],
    {"all": shared_keys},
    labels=labels_2,
    colors=["C1", "C2"],
    truths_dict={"all": TRUTHS},
    hist_kwargs={"density": True},
    levels=[0.95],
    plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True,
    save_path=os.path.join(OUTPUT_DIR, "compare_deriv_approx_vs_lenstronomy_nautilus_qphi.png"),
)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'compare_deriv_approx_vs_lenstronomy_nautilus_qphi.png')}")
