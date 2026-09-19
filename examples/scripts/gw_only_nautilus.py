"""
GW-only multi-method comparison: deriv-approx, deriv-approx-source, hmc-informed-source,
fisher-source, fisher, and nautilus-source.

Flip the RUN_* toggles below to enable/disable each method block.
Per-method groupwise source-plane corners; multi-way overlay via
create_default_param_groups + plot_multi_comparison_corner.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_nautilus")

# Method toggles — only blocks with True are executed and included in comparison plots.
RUN_DERIV_APPROX = False
RUN_DERIV_APPROX_SOURCE = True
RUN_FISHER_SOURCE = True
RUN_NAUTILUS_SOURCE = True
RUN_FISHER = False
RUN_HMC_INFORMED_SOURCE = False

# hmc-informed-source sample checkpoint (like NAUTILUS_CHECKPOINT / NAUTILUS_RESUME).
HMC_INFORMED_SOURCE_SAMPLES = os.path.join(OUTPUT_DIR, "samples_hmc_informed_source.npz")
LOAD_HMC_INFORMED_SOURCE_SAMPLES = False  # True → load npz, skip run_inference

# HMC NUTS smoke settings (hmc-source / hmc-informed-source only; deriv-approx uses BASE_CFG)
HMC_NUM_WARMUP = 15000
HMC_NUM_SAMPLES = 5000
HMC_NUM_CHAINS = 3

# Nautilus checkpoint: resume=True continues a run with the *same* priors.
# First run after changing NAUTILUS_SIGMA_SPAN, free params, or manual priors → False.
# Re-run with unchanged span/setup → True (fisher-source rebuilds the same H0 boxes).
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False
NAUTILUS_SIGMA_SPAN = 2.0  # Fisher H0 span for nautilus-source priors (after fisher-source)

IMAGE_PLANE_CORNER_PATH  = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE       = "pipeline_outputs.json"

COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")
COMPARISON_SOURCE_ALL_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_all.png")

COMPARISON_LABELS = {
    "deriv-approx": "deriv-approx (ray-shot)",
    "deriv-approx-source": "deriv-approx-source (native source-plane)",
    "hmc-informed-source": "hmc-informed-source (full likelihood, native source-plane)",
    "fisher-source": "fisher-source (native source-plane)",
    "nautilus-source": "nautilus-source (source-plane)",
    "fisher": "fisher (ray-shot)",
}
METHOD_COLORS = {
    "deriv-approx": "C0",
    "deriv-approx-source": "C3",
    "hmc-informed-source": "C5",
    "fisher-source": "C4",
    "nautilus-source": "C1",
    "fisher": "C2",
}
SOURCE_PLANE_CFG = {
    "source_plane": {"filter_std": None, "use_filtered": False},
}

# GW source position (arcsec). Drives image solving, time delays, and truth y0gw/y1gw.
GW_SOURCE_POS = (0.05, 1e-6)

# Tight boxes around truth for free image positions (deriv/fisher), source plane (nautilus),
# and cosmology (all methods).
SOURCE_HALF_Y0 = 0.1
SOURCE_HALF_Y1 = 0.08
IMAGE_BOX_HALF = 1.2
T_STAR_HALF_FRAC = 0.70
DL_HALF_FRAC = 0.70

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
            "sigma_td": 0.002,#0.005,
            "epsilon": 0.0001,
            "sigma_dL_eff": 0.002,
        },
        "source_plane_bounds": {
            "y0gw": (GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
            "y1gw": (GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
        },
        "image_box_half_width": IMAGE_BOX_HALF,
    },
    "inference": {
        "num_warmup": 20000,
        "num_samples": 9000,
        "num_chains": 20,
    },
    "nautilus": {
        "n_live": 2000,
        "n_eff": 5000,
        "n_like_max": 500000,
        "solver_backend": "helens",
        "verbose": True,
        "filepath": NAUTILUS_CHECKPOINT,
        "resume": NAUTILUS_RESUME,
    },
}

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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

from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish.fisher import invert_fisher_matrix
from gwemfish import (
    setup_gw_observation,
    run_inference,
    plot_posterior,
    to_source_plane_samples,
    plot_source_posterior,
)


def load_samples_npz(path):
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}


def apply_fisher_h0_priors(ctx, span):
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
        lo = mu - span * sig
        hi = mu + span * sig
        ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
        print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")


os.makedirs(OUTPUT_DIR, exist_ok=True)

active = [k for k, v in [
    ("deriv-approx", RUN_DERIV_APPROX),
    ("deriv-approx-source", RUN_DERIV_APPROX_SOURCE),
    ("hmc-informed-source", RUN_HMC_INFORMED_SOURCE),
    ("fisher-source", RUN_FISHER_SOURCE),
    ("nautilus-source", RUN_NAUTILUS_SOURCE),
    ("fisher", RUN_FISHER),
] if v]
print(f"Active methods: {', '.join(active) if active else '(none)'}")
if RUN_HMC_INFORMED_SOURCE:
    print(f"hmc-informed-source load samples: {LOAD_HMC_INFORMED_SOURCE_SAMPLES} ({HMC_INFORMED_SOURCE_SAMPLES})")

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]

t_star_true = float(tp["T_star"])
dL_true = float(tp["dL"])

ctx["cfg"]["priors"] = {
    "T_star":         dist.Uniform(t_star_true - T_STAR_HALF_FRAC * t_star_true, t_star_true + T_STAR_HALF_FRAC * t_star_true),
    "dL":             dist.Uniform(dL_true - DL_HALF_FRAC * dL_true, dL_true + DL_HALF_FRAC * dL_true),
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_e1":       float(tp["lens0_e1"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens1_gamma1":   float(tp["lens1_gamma1"]),
    "lens1_gamma2":   float(tp["lens1_gamma2"]),
    "lens1_ra_0":     float(tp["lens1_ra_0"]),
    "lens1_dec_0":    float(tp["lens1_dec_0"]),
    "y0gw":           dist.Uniform(GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
    "y1gw":           dist.Uniform(GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
    "lens0_gamma":    dist.Uniform(1.5, 3.0),#float(tp["lens0_gamma"]),
    "lens0_e2":       dist.Uniform(-0.5, 0.5),
}

n_images = sum(1 for k in tp if k.startswith("image_x"))
for i in range(1, n_images + 1):
    xt = float(tp[f"image_x{i}"])
    yt = float(tp[f"image_y{i}"])
    ctx["cfg"]["priors"][f"image_x{i}"] = dist.Uniform(xt - IMAGE_BOX_HALF, xt + IMAGE_BOX_HALF)
    ctx["cfg"]["priors"][f"image_y{i}"] = dist.Uniform(yt - IMAGE_BOX_HALF, yt + IMAGE_BOX_HALF)

truths_source = {k: float(tp[k]) for k in tp
                 if not (k.startswith("image_x") or k.startswith("image_y"))}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

source_by_method = {}

if RUN_DERIV_APPROX:
    print("\n--- GW-only inference: deriv-approx ---\n")

    deriv_cfg = {
        "inference": {"informed": True},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "deriv-approx",
        },
    }
    samples_deriv, truths_deriv = run_inference(
        ctx, mode="GW-only", method="deriv-approx", cfg=deriv_cfg,
    )

    corner_dir_deriv = os.path.join(OUTPUT_DIR, "deriv_approx")
    os.makedirs(corner_dir_deriv, exist_ok=True)

    plot_posterior(
        samples_deriv, truths_deriv,
        cfg={
            "output": {"output_dir": corner_dir_deriv},
            "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
        },
    )

    source_out_deriv = to_source_plane_samples(
        samples_deriv, ctx,
        cfg={
            **SOURCE_PLANE_CFG,
            "output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "deriv-approx"},
        },
    )
    plot_source_posterior(
        source_out_deriv, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_deriv},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["deriv-approx"] = source_out_deriv["source_plane_samples_plot"]

if RUN_DERIV_APPROX_SOURCE:
    print("\n--- GW-only inference: deriv-approx-source ---\n")

    deriv_source_cfg = {
        "inference": {"informed": True},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "deriv-approx-source",
        },
    }
    samples_deriv_source, _ = run_inference(
        ctx, mode="GW-only", method="deriv-approx-source", cfg=deriv_source_cfg,
    )

    corner_dir_deriv_source = os.path.join(OUTPUT_DIR, "deriv_approx_source")
    os.makedirs(corner_dir_deriv_source, exist_ok=True)

    plot_source_posterior(
        samples_deriv_source, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_deriv_source},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["deriv-approx-source"] = samples_deriv_source

if RUN_HMC_INFORMED_SOURCE:
    corner_dir_hmc_informed_source = os.path.join(OUTPUT_DIR, "hmc_informed_source")
    os.makedirs(corner_dir_hmc_informed_source, exist_ok=True)

    if LOAD_HMC_INFORMED_SOURCE_SAMPLES:
        if not os.path.isfile(HMC_INFORMED_SOURCE_SAMPLES):
            raise FileNotFoundError(
                f"LOAD_HMC_INFORMED_SOURCE_SAMPLES=True but file missing: {HMC_INFORMED_SOURCE_SAMPLES}"
            )
        print("\n--- Loading hmc-informed-source samples (skip inference) ---\n")
        print(f"  {HMC_INFORMED_SOURCE_SAMPLES}")
        samples_hmc_informed_source = load_samples_npz(HMC_INFORMED_SOURCE_SAMPLES)
    else:
        print("\n--- GW-only inference: hmc-informed-source ---\n")
        hmc_informed_source_cfg = {
            "inference": {
                "num_warmup": HMC_NUM_WARMUP,
                "num_samples": HMC_NUM_SAMPLES,
                "num_chains": HMC_NUM_CHAINS,
            },
            "output": {
                "output_dir": OUTPUT_DIR,
                "json_path": PIPELINE_JSON_BASE,
                "json_tag": "hmc-informed-source",
            },
        }
        samples_hmc_informed_source, _ = run_inference(
            ctx, mode="GW-only", method="hmc-informed-source", cfg=hmc_informed_source_cfg,
        )
        np.savez(HMC_INFORMED_SOURCE_SAMPLES, **samples_hmc_informed_source)
        print(f"Saved samples: {HMC_INFORMED_SOURCE_SAMPLES}")

    plot_source_posterior(
        samples_hmc_informed_source, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_hmc_informed_source},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["hmc-informed-source"] = samples_hmc_informed_source

if RUN_FISHER_SOURCE:
    print("\n--- GW-only inference: fisher-source ---\n")

    fisher_source_cfg = {
        "inference": {"n_fisher_samples": 10000},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "fisher-source",
        },
    }
    samples_fisher_source, _ = run_inference(
        ctx, mode="GW-only", method="fisher-source", cfg=fisher_source_cfg,
    )

    corner_dir_fisher_source = os.path.join(OUTPUT_DIR, "fisher_source")
    os.makedirs(corner_dir_fisher_source, exist_ok=True)

    plot_source_posterior(
        samples_fisher_source, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_fisher_source},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["fisher-source"] = samples_fisher_source

nautilus_resume = NAUTILUS_RESUME
if RUN_NAUTILUS_SOURCE:
    # BUGFIX (task14): the nautilus priors must NOT depend on the RUN_FISHER_SOURCE
    # toggle. The checkpoint was built under the Fisher-H0 uniform boxes; nautilus
    # stores unit-cube points and maps them through the CURRENT prior on resume, so
    # resuming under the manual wide boxes silently stretches the stored points and
    # inflates the "nautilus reference" sigmas (2.5-7.5x here). Any earlier
    # run_inference call (deriv-approx-source, fisher-source, ...) populates
    # ctx['fisher']['H0'] with the SAME source-plane Hessian, so rebuild the same
    # Fisher-H0 priors whenever it is available.
    if "fisher" in ctx and "likelihood" in ctx:
        print(f"\n--- Nautilus-source priors from Fisher H0 (span={NAUTILUS_SIGMA_SPAN}) ---\n")
        apply_fisher_h0_priors(ctx, NAUTILUS_SIGMA_SPAN)
    else:
        print("Nautilus: manual priors (no Fisher H0 in ctx — no prior method ran).")
        if nautilus_resume and os.path.isfile(NAUTILUS_CHECKPOINT):
            print("WARNING: resuming an existing checkpoint under manual priors — "
                  "gwemfish will refuse if the checkpoint was built under different priors.")

    print(f"Nautilus resume: {nautilus_resume} (set NAUTILUS_RESUME=False after span/prior changes)")
    print("\n--- GW-only inference: nautilus-source (source-plane) ---\n")

    nautilus_cfg = {
        "nautilus": {
            "filepath": NAUTILUS_CHECKPOINT,
            "resume": nautilus_resume,
        },
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "nautilus_source",
        },
    }
    samples_nautilus, _ = run_inference(
        ctx, mode="GW-only", method="nautilus-source", cfg=nautilus_cfg,
    )

    corner_dir_nautilus = os.path.join(OUTPUT_DIR, "nautilus_source")
    os.makedirs(corner_dir_nautilus, exist_ok=True)

    plot_source_posterior(
        samples_nautilus, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_nautilus},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["nautilus-source"] = samples_nautilus

if RUN_FISHER:
    print("\n--- GW-only inference: fisher ---\n")

    fisher_cfg = {
        "inference": {"n_fisher_samples": 10000},
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": "fisher",
        },
    }
    samples_fisher, truths_fisher = run_inference(
        ctx, mode="GW-only", method="fisher", cfg=fisher_cfg,
    )

    corner_dir_fisher = os.path.join(OUTPUT_DIR, "fisher")
    os.makedirs(corner_dir_fisher, exist_ok=True)

    source_out_fisher = to_source_plane_samples(
        samples_fisher, ctx,
        cfg={
            **SOURCE_PLANE_CFG,
            "output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "fisher"},
        },
    )
    plot_source_posterior(
        source_out_fisher, truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir_fisher},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )
    source_by_method["fisher"] = source_out_fisher["source_plane_samples_plot"]

if not source_by_method:
    raise RuntimeError("No methods enabled — set at least one RUN_* toggle to True.")

METHODS = tuple(source_by_method.keys())
ref_samples = source_by_method[METHODS[0]]
all_sp_keys = sorted(
    k for k in ref_samples
    if all(k in source_by_method[m] for m in METHODS)
)
param_groups_sp = create_default_param_groups(ref_samples)
if len(all_sp_keys) >= 2:
    param_groups_sp = {"all": all_sp_keys, **param_groups_sp}
truths_dict_sp = {
    group: {p: float(truths_source[p]) for p in params if p in truths_source}
    for group, params in param_groups_sp.items()
}
comparison_labels = [COMPARISON_LABELS[m] for m in METHODS]
comparison_colors = [METHOD_COLORS[m] for m in METHODS]
# comparison_kw = {"hist_kwargs": {"density": True},"plot_datapoints": False}
comparison_kw = {
    "hist_kwargs": {"density": True},
    "plot_datapoints": False,
}

if len(all_sp_keys) >= 2:
    plot_multi_comparison_corner(
        [source_by_method[m] for m in METHODS],
        {"all": all_sp_keys},
        labels=comparison_labels,
        colors=comparison_colors,
        truths_dict={"all": truths_dict_sp.get("all", {})},
        save_path=COMPARISON_SOURCE_ALL_PATH,
        **comparison_kw,
    )
    print(f"Combined source-plane comparison saved: {COMPARISON_SOURCE_ALL_PATH}")

if len(METHODS) >= 2 and any(len(v) >= 2 for v in param_groups_sp.values() if v):
    plot_multi_comparison_corner(
        [source_by_method[m] for m in METHODS],
        {k: v for k, v in param_groups_sp.items() if k != "all"},
        labels=comparison_labels,
        colors=comparison_colors,
        truths_dict={k: v for k, v in truths_dict_sp.items() if k != "all"},
        save_path=COMPARISON_SOURCE_CORNER_PATH,
        **comparison_kw,
    )
    print("Per-group source-plane comparisons saved under:",
          COMPARISON_SOURCE_CORNER_PATH.replace("{group_name}", "*"))

print(f"\nDone. Outputs: {os.path.abspath(OUTPUT_DIR)}/")
