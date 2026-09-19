"""
GW-only comparison: nautilus-image (image_x/y sampling) vs deriv-approx vs fisher.

Uses the same flex layout and tight image-position boxes as gw_only_nautilus.py, but
nautilus-image samples image positions directly (full HMC-equivalent GW likelihood).
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_nautilus_image")

NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_image_checkpoint.hdf5")
NAUTILUS_RESUME = True
NAUTILUS_SIGMA_SPAN = 2.0

IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"
COMPARISON_IMAGE_PLANE_PATH = os.path.join(
    OUTPUT_DIR, "comparison_image_plane_{group_name}.png"
)
COMPARISON_KEY_PARAMS_PATH = os.path.join(OUTPUT_DIR, "comparison_key_params.png")

METHODS = ("deriv-approx", "nautilus-image", "fisher")
KEY_PARAMS = ["T_star", "dL","lens0_gamma", "lens0_e2"]

METHOD_COLORS = {"deriv-approx": "C0", "nautilus-image": "C1", "fisher": "C2"}

GW_SOURCE_POS = (0.05, 1e-6)
IMAGE_BOX_HALF = 0.2
T_STAR_HALF_FRAC = 0.40
DL_HALF_FRAC = 0.22

BASE_CFG = {
    "use_parameter_layout": True,
    "em": {"enabled": False},
    "lens": {
        "kwargs_lens": [
            {
                "theta_E": 1.2,
                "e1": 0.0,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
    },
    "gw": {
        "source_pos": GW_SOURCE_POS,
        "error_scales": {
            "sigma_td": 0.005,
            "sigma_dL_eff": 0.02,
        },
        "image_box_half_width": IMAGE_BOX_HALF,
    },
    "nautilus": {
        "n_live":1000,
        "n_eff": 8000,
        "solver_backend": "jaxtronomy",
        "verbose": True,
        "filepath": NAUTILUS_CHECKPOINT,
        "resume": NAUTILUS_RESUME,
    },
    "inference": {
        "num_warmup": 1000,
        "num_samples": 5000,
        "num_chains": 4,
    },
}

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpyro.distributions as dist
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish import setup_gw_observation, run_inference, plot_posterior
from gwemfish.fisher import invert_fisher_matrix


def require_matching_sample_keys(samples_by_method):
    names = list(samples_by_method.keys())
    ref_name = names[0]
    ref_keys = set(samples_by_method[ref_name].keys())
    for name in names[1:]:
        keys = set(samples_by_method[name].keys())
        if keys != ref_keys:
            raise ValueError(
                f"Sample keys mismatch between {ref_name!r} and {name!r}. "
                f"Only in {ref_name}: {sorted(ref_keys - keys)}. "
                f"Only in {name}: {sorted(keys - ref_keys)}."
            )
    return ref_keys

os.makedirs(OUTPUT_DIR, exist_ok=True)

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]

t_star_true = float(tp["T_star"])
dL_true = float(tp["dL"])

ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_e1":       float(tp["lens0_e1"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens1_gamma1":   float(tp["lens1_gamma1"]),
    "lens1_gamma2":   float(tp["lens1_gamma2"]),
    "lens1_ra_0":     float(tp["lens1_ra_0"]),
    "lens1_dec_0":    float(tp["lens1_dec_0"]),
    "T_star":         dist.Uniform(t_star_true * (1 - T_STAR_HALF_FRAC), t_star_true * (1 + T_STAR_HALF_FRAC)),#float(tp["T_star"]),,
    "dL":             dist.Uniform(dL_true * (1 - DL_HALF_FRAC), dL_true * (1 + DL_HALF_FRAC)),#float(tp["dL"]),,
    "lens0_gamma":    float(tp["lens0_gamma"]),#dist.Uniform(1.7, 2.9),#float(tp["lens0_gamma"]),,
    "lens0_e2":       dist.Uniform(0.05, 0.185),#float(tp["lens0_e2"]),,
}

n_images = sum(1 for k in tp if k.startswith("image_x"))
for i in range(1, n_images + 1):
    xt = float(tp[f"image_x{i}"])
    yt = float(tp[f"image_y{i}"])
    ctx["cfg"]["priors"][f"image_x{i}"] = dist.Uniform(xt - IMAGE_BOX_HALF, xt + IMAGE_BOX_HALF)
    ctx["cfg"]["priors"][f"image_y{i}"] = dist.Uniform(yt - IMAGE_BOX_HALF, yt + IMAGE_BOX_HALF)

print("\n--- GW-only inference: deriv-approx ---\n")
samples_deriv, truths_deriv = run_inference(
    ctx,
    mode="GW-only",
    method="deriv-approx",
    cfg={
        "inference": {"informed": True},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "deriv-approx",
        },
    },
)

corner_dir_deriv = os.path.join(OUTPUT_DIR, "deriv_approx")
os.makedirs(corner_dir_deriv, exist_ok=True)
plot_posterior(
    samples_deriv,
    truths_deriv,
    cfg={
        "output": {"output_dir": corner_dir_deriv},
        "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
    },
)

print("\n--- Nautilus-image priors from Fisher H0 (deriv-approx) ---\n")
keys = ctx["likelihood"]["keys_to_include"]
u0 = np.asarray(ctx["likelihood"]["u0"])
H0 = np.asarray(ctx["fisher"]["H0"])
regularize = bool((ctx.get("cfg") or {}).get("inference", {}).get("regularize", False))
cov = np.asarray(invert_fisher_matrix(-H0, regularize=regularize))
sigmas = np.sqrt(np.diag(cov))

for i, key in enumerate(keys):
    sig = float(sigmas[i])
    if not np.isfinite(sig) or sig <= 0:
        print(f"  Nautilus prior {key}: skip (sigma={sig}) — keep existing prior")
        continue
    mu = float(u0[i])
    lo = mu - NAUTILUS_SIGMA_SPAN * sig
    hi = mu + NAUTILUS_SIGMA_SPAN * sig
    ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
    print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")

print("\n--- GW-only inference: nautilus-image ---\n")
samples_image, truths_image = run_inference(
    ctx,
    mode="GW-only",
    method="nautilus-image",
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "nautilus_image",
        },
    },
)

corner_dir_image = os.path.join(OUTPUT_DIR, "nautilus_image")
os.makedirs(corner_dir_image, exist_ok=True)
plot_posterior(
    samples_image,
    truths_image,
    cfg={
        "output": {"output_dir": corner_dir_image},
        "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
    },
)

print("\n--- GW-only inference: fisher ---\n")
samples_fisher, truths_fisher = run_inference(
    ctx,
    mode="GW-only",
    method="fisher",
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "fisher",
        },
    },
)

require_matching_sample_keys({
    m: samples for m, samples in zip(METHODS, [samples_deriv, samples_image, samples_fisher])
})

sample_list = [samples_deriv, samples_image, samples_fisher]
labels = list(METHODS)
colors = [METHOD_COLORS[m] for m in METHODS]
plot_kw = {"hist_kwargs": {"density": True}}

param_groups = create_default_param_groups(samples_deriv)
truths_dict = {
    group: {p: float(truths_deriv[p]) for p in params if p in truths_deriv}
    for group, params in param_groups.items()
}
plot_multi_comparison_corner(
    sample_list,
    param_groups,
    labels=labels,
    colors=colors,
    truths_dict=truths_dict,
    save_path=COMPARISON_IMAGE_PLANE_PATH,
    **plot_kw,
)

key_params = [p for p in KEY_PARAMS if p in samples_deriv]
plot_multi_comparison_corner(
    sample_list,
    {"key_params": key_params},
    labels=labels,
    colors=colors,
    truths_dict={"key_params": {p: float(truths_deriv[p]) for p in key_params if p in truths_deriv}},
    save_path=COMPARISON_KEY_PARAMS_PATH,
    **plot_kw,
)

print("\nDone.")
