"""
Compare deriv-approx, nautilus, and fisher on the same EM-only system.

Outputs:
- Tagged pipeline JSON under OUTPUT_DIR.
- Per-method corner plots under OUTPUT_DIR/<method_tag>/.
- Three-way image-plane overlay comparison corners under OUTPUT_DIR.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_em_nautilus")

NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False
NAUTILUS_SIGMA_SPAN = 5.0
# Changing Nautilus priors → set NAUTILUS_RESUME = False or delete the checkpoint.

IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"
COMPARISON_IMAGE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_image_plane_{group_name}.png")

METHODS = ("deriv-approx", "nautilus-source", "fisher")
METHOD_COLORS = {"deriv-approx": "C0", "nautilus-source": "C1", "fisher": "C2"}


def method_tag_dir(method):
    return method.strip().lower().replace("-", "_").replace(" ", "_")


def inference_cfg(method):
    cfg = {"output": {"json_tag": method}}
    if method == "deriv-approx":
        cfg["inference"] = {"informed": True}
    if method == "nautilus-source":
        cfg["nautilus"] = {
            "filepath": NAUTILUS_CHECKPOINT,
            "resume": NAUTILUS_RESUME,
        }
    return cfg


def run_method_and_plot(method):
    print(f"\n--- EM-only inference: {method} ---\n")
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
        method=method,
        cfg=inference_cfg(method),
    )
    corner_dir = os.path.join(OUTPUT_DIR, method_tag_dir(method))
    os.makedirs(corner_dir, exist_ok=True)
    plot_posterior(
        samples,
        truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
        },
    )
    return samples, truths


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import (
    DEFAULT_KWARGS_NUMERICS,
    make_default_cfg,
    plot_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish.fisher import invert_fisher_matrix

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(
    f"Nautilus: {'resume ' + NAUTILUS_CHECKPOINT if NAUTILUS_RESUME and os.path.isfile(NAUTILUS_CHECKPOINT) else 'fresh run'}"
)

source_pos = (0.05, 0.1)
CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
pix_scl = 0.1
CFG["em"].update(
    {
        "pixel_grid_kwargs": {"npix": 40, "pix_scl": pix_scl},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.067, "pixel_size": pix_scl},
        "noise_simu_kwargs": {"npix": 40, "background_rms": 1e-2, "exposure_time": 2200},
        "noise_inf_kwargs": {"npix": 40, "background_rms": None, "exposure_time": 2200},
        "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
        "exposure_time": 2200,
        "seed": 87651,
        "source_pos": source_pos,
        "kwargs_source": [
            {
                "amp": 250,
                "R_sersic": 0.04,
                "n_sersic": 1.0,
                "e1": -0.1,
                "e2": 0.2,
                "center_x": source_pos[0],
                "center_y": source_pos[1],
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 50.0,
                "R_sersic": 2.0,
                "n_sersic": 4.0,
                "e1": 0.0,
                "e2": 0.1,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
    }
)
CFG["lens"].update(
    {
        "lens_model_list": ["EPL", "SHEAR"],
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
        "zl": 0.7,
        "zs": 1.5,
    }
)
CFG["plot"].update(
    {
        "plot_mode": "groupwise",
        "save_path": None,
        "save_tag": None,
        "hist_kwargs": {"density": True},
    }
)
CFG["output"].update(
    {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": PIPELINE_JSON_BASE,
    }
)
CFG["nautilus"] = {
    "n_live": 1000,
    "n_eff": 5000,
    "n_like_max": 500000,
    "verbose": True,
    "filepath": NAUTILUS_CHECKPOINT,
    "resume": NAUTILUS_RESUME,
}

ctx = setup_em_observation(cfg=CFG)
tp = ctx["truth_params"]

plot_system_observation(ctx, cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}})

ctx["cfg"]["priors"] = {
    "lens1_ra_0": float(tp["lens1_ra_0"]),
    "lens1_dec_0": float(tp["lens1_dec_0"]),
    "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
}
for key in tp:
    if key.startswith("light0_"):
        ctx["cfg"]["priors"][key] = float(tp[key])

samples_by_method = {}

samples_deriv, truths_image = run_method_and_plot("deriv-approx")
samples_by_method["deriv-approx"] = samples_deriv

print("\n--- Nautilus priors from Fisher H0 (deriv-approx) ---\n")
keys = ctx["likelihood"]["keys_to_include"]
u0 = np.asarray(ctx["likelihood"]["u0"])
H0 = np.asarray(ctx["fisher"]["H0"])
regularize = bool((ctx.get("cfg") or {}).get("inference", {}).get("regularize", False))
cov = np.asarray(invert_fisher_matrix(-H0, regularize=regularize))
sigmas = np.sqrt(np.diag(cov))

for i, key in enumerate(keys):
    sig = float(sigmas[i])
    if not np.isfinite(sig) or sig <= 0:
        print(f"  Nautilus prior {key}: skip (sigma={sig}) — using default prior")
        continue
    mu = float(u0[i])
    lo = mu - NAUTILUS_SIGMA_SPAN * sig
    hi = mu + NAUTILUS_SIGMA_SPAN * sig
    ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
    print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")

samples_nautilus, _ = run_method_and_plot("nautilus-source")
samples_by_method["nautilus-source"] = samples_nautilus

samples_fisher, _ = run_method_and_plot("fisher")
samples_by_method["fisher"] = samples_fisher

shared_img_keys = sorted(
    k for k in samples_by_method[METHODS[0]]
    if all(k in samples_by_method[m] for m in METHODS)
)
if not shared_img_keys:
    print("Warning: no shared image-plane params — skipping comparison corner.")
else:
    param_groups_img = create_default_param_groups(samples_by_method[METHODS[0]])
    param_groups_img = {
        g: [p for p in ps if p in shared_img_keys]
        for g, ps in param_groups_img.items()
    }
    param_groups_img = {g: ps for g, ps in param_groups_img.items() if ps}
    truths_dict_img = {
        group: {p: float(truths_image[p]) for p in params if p in truths_image}
        for group, params in param_groups_img.items()
    }
    plot_multi_comparison_corner(
        [samples_by_method[m] for m in METHODS],
        param_groups_img,
        labels=list(METHODS),
        colors=[METHOD_COLORS[m] for m in METHODS],
        truths_dict=truths_dict_img,
        save_path=COMPARISON_IMAGE_CORNER_PATH,
        hist_kwargs={"density": True},
    )
    print("Image-plane comparison corners saved under:", COMPARISON_IMAGE_CORNER_PATH.replace("{group_name}", "*"))

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
