"""
GW-only tutorial: fisher-source, deriv-approx-source, nautilus-source.

Flip RUN_* toggles to enable/disable methods. Comparison plots only run when
two or more methods produced samples (no error if a method is off).

Nautilus priors: NAUTILUS_PRIOR_MODE = "fisher_h0" | "manual".
Change mode/span → set NAUTILUS_RESUME = False (or delete the .hdf5).

Does not modify tutorial/gwemfish_tutorial.py.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX devices: {jax.devices()}")

import matplotlib
import numpy as np
import numpyro.distributions as dist

matplotlib.use("Agg")

from gwemfish import (
    make_default_cfg,
    plot_psf,
    plot_source_plane_caustic_with_localization_from_setup,
    plot_source_posterior,
    plot_system_observation,
    prune_gw_images,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish.fisher import invert_fisher_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "tutorial", "outputs", "gw_only")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_FISHER_SOURCE = True
RUN_DERIV_APPROX_SOURCE = True
RUN_NAUTILUS_SOURCE = False

NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False
NAUTILUS_PRIOR_MODE = "fisher_h0"  # "fisher_h0" | "manual"
NAUTILUS_SIGMA_SPAN = 3.5

METHOD_COLORS = {
    "fisher-source": "C4",
    "deriv-approx-source": "C3",
    "nautilus-source": "C1",
}


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


CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
CFG["lens_mass_parametrization"] = "e1e2" #"e1e2"#"q_phi"  # "e1e2" to go back
CFG["gw"]["n_images"] = 4#2
CFG["gw"]["source_box_half_width"] = 0.8
# CFG["source_plane"]["n_images"] = 2
CFG["gw"]["source_pos"] = (0.02, 0.00001)
CFG["gw"]["solver_params"]["backend"] = "jaxtronomy"
CFG["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"#"lenstronomy"
CFG["gw"]["error_scales"]["sigma_td"] = 0.001
CFG["gw"]["error_scales"]["sigma_dL_eff"] = 0.05 #0.1 tstar error is large going to negative values
CFG["inference"]["num_chains"] = 12
CFG["inference"]["num_samples"] = 14000
CFG["inference"]["num_warmup"] = 9000
CFG["nautilus"] = {
    "n_live": 2000,
    "n_eff": 5000,
    "n_like_max": 500_000,
    "filepath": NAUTILUS_CHECKPOINT,
    "resume": NAUTILUS_RESUME,
    "prior_check": True,
    "verbose": True,
}
CFG["output"]["output_dir"] = OUTPUT_DIR

active = [
    name
    for name, flag in [
        ("fisher-source", RUN_FISHER_SOURCE),
        ("deriv-approx-source", RUN_DERIV_APPROX_SOURCE),
        ("nautilus-source", RUN_NAUTILUS_SOURCE),
    ]
    if flag
]
print(f"Active methods: {', '.join(active) if active else '(none)'}")
print(f"Nautilus prior mode: {NAUTILUS_PRIOR_MODE}, resume={NAUTILUS_RESUME}")

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
# ctx = prune_gw_images(ctx, n_keep=2)

truth_params = ctx["truth_params"]
src = ctx["cfg"]["gw"]["source_pos"]
print(f"Simulated {ctx['n_images']} GW images.")

plot_system_observation(
    ctx,
    cfg={"output": {"output_dir": OUTPUT_DIR, "save_system_plot_path": "system_observation.png"}},
)
plot_psf(ctx, cfg={"output": {"output_dir": OUTPUT_DIR, "save_psf_plot_path": "psf.png"}})

Y0_LO, Y0_HI = -0.06, 0.06#0.018, 0.022#0.01992, 0.02005
Y1_LO, Y1_HI = -0.06, 0.06#0.004, 0.016#0.0091, 0.0106

PRIORS = {
    "lens1_gamma1": float(truth_params["lens1_gamma1"]),
    "lens1_gamma2": float(truth_params["lens1_gamma2"]),
    "lens1_ra_0": float(truth_params["lens1_ra_0"]),
    "lens1_dec_0": float(truth_params["lens1_dec_0"]),
    "lens0_theta_E": float(truth_params["lens0_theta_E"]),
    "lens0_center_x": float(truth_params["lens0_center_x"]),
    "lens0_center_y": float(truth_params["lens0_center_y"]),
    # "T_star": float(truth_params["T_star"]),  # dist.Uniform(1e-1, 1e12),
    # "dL": float(truth_params["dL"]), #dist.Uniform(1e-5, 50000.0),
    # e1e2 mode (set lens_mass_parametrization="e1e2"):
    # "lens0_e1": dist.Uniform(-0.5, 0.5),#float(truth_params["lens0_e1"]),
    "lens0_e2": float(truth_params["lens0_e2"]),
    # q_phi mode: fix phi to truth
    "lens0_phi": float(truth_params["lens0_phi"]),
    # "lens0_q": dist.Uniform(0.75, 0.85),
    "lens0_gamma": float(truth_params["lens0_gamma"]),#dist.Uniform(1.5, 2.4), #float(truth_params["lens0_gamma"]),
    "y0gw": dist.Uniform(Y0_LO, Y0_HI),
    "y1gw": dist.Uniform(Y1_LO, Y1_HI),
}
ctx["cfg"]["priors"] = dict(PRIORS)

# This is for nautilus-source method
ctx["cfg"]["gw"]["source_plane_bounds"] = {
    "y0gw": (Y0_LO, Y0_HI),
    "y1gw": (Y1_LO, Y1_HI),
}

samples_by_method = {}
truths_by_method = {}

if RUN_FISHER_SOURCE:
    print("\n--- GW-only: fisher-source ---\n")
    samples, truths = run_inference(
        ctx,
        mode="GW-only",
        method="fisher-source",
        cfg={
            "priors": PRIORS,
            "output": {"output_dir": OUTPUT_DIR, "json_tag": "fisher_source"},
        },
    )
    truths.setdefault("y0gw", float(src[0]))
    truths.setdefault("y1gw", float(src[1]))
    corner_dir = os.path.join(OUTPUT_DIR, "fisher_source")
    os.makedirs(corner_dir, exist_ok=True)
    plot_source_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {"plot_mode": "combined", "save_path": "source_posterior_combined.png"},
        },
    )
    samples_by_method["fisher-source"] = samples
    truths_by_method["fisher-source"] = truths

if RUN_DERIV_APPROX_SOURCE:
    print("\n--- GW-only: deriv-approx-source ---\n")
    samples, truths = run_inference(
        ctx,
        mode="GW-only",
        method="deriv-approx-source",
        cfg={
            "priors": PRIORS,
            "output": {"output_dir": OUTPUT_DIR, "json_tag": "deriv_approx_source"},
            "inference": {"informed": True},
        },
    )
    truths.setdefault("y0gw", float(src[0]))
    truths.setdefault("y1gw", float(src[1]))
    corner_dir = os.path.join(OUTPUT_DIR, "deriv_approx_source")
    os.makedirs(corner_dir, exist_ok=True)
    plot_source_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {"plot_mode": "combined", "save_path": "source_posterior_combined.png"},
        },
    )
    plot_source_plane_caustic_with_localization_from_setup(
        source_samples=samples,
        ctx=ctx,
        truths_source=truths,
        level=0.90,
        save_path=os.path.join(corner_dir, "source_localization_90.png"),
    )
    samples_by_method["deriv-approx-source"] = samples
    truths_by_method["deriv-approx-source"] = truths

if RUN_NAUTILUS_SOURCE:
    if NAUTILUS_PRIOR_MODE == "fisher_h0":
        if "fisher" not in ctx or "likelihood" not in ctx:
            print("\n--- Precursor fisher-source for H0 (nautilus only) ---\n")
            run_inference(
                ctx,
                mode="GW-only",
                method="fisher-source",
                cfg={
                    "priors": PRIORS,
                    "output": {"output_dir": OUTPUT_DIR, "json_tag": "fisher_source_h0_precursor"},
                },
            )
        print(f"\n--- Nautilus priors from Fisher H0 (span={NAUTILUS_SIGMA_SPAN}) ---\n")
        apply_fisher_h0_priors(ctx, NAUTILUS_SIGMA_SPAN)
    elif NAUTILUS_PRIOR_MODE == "manual":
        print("\n--- Nautilus manual priors (hand-set Uniforms / source_plane_bounds) ---\n")
        ctx["cfg"]["priors"] = dict(PRIORS)
        ctx["cfg"]["gw"]["source_plane_bounds"] = {
            "y0gw": (Y0_LO, Y0_HI),
            "y1gw": (Y1_LO, Y1_HI),
        }
    else:
        raise ValueError(f"unknown NAUTILUS_PRIOR_MODE={NAUTILUS_PRIOR_MODE!r}")

    print(f"\n--- GW-only: nautilus-source (resume={NAUTILUS_RESUME}) ---\n")
    samples, truths = run_inference(
        ctx,
        mode="GW-only",
        method="nautilus-source",
        cfg={
            "priors": ctx["cfg"]["priors"],
            "nautilus": {
                "filepath": NAUTILUS_CHECKPOINT,
                "resume": NAUTILUS_RESUME,
                "prior_check": True,
            },
            "output": {"output_dir": OUTPUT_DIR, "json_tag": "nautilus_source"},
        },
    )
    # nautilus truths omit y0gw/y1gw unless truth_params was backfilled
    truths.setdefault("y0gw", float(src[0]))
    truths.setdefault("y1gw", float(src[1]))
    corner_dir = os.path.join(OUTPUT_DIR, "nautilus_source")
    os.makedirs(corner_dir, exist_ok=True)
    plot_source_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {"plot_mode": "combined", "save_path": "source_posterior_combined.png"},
        },
    )
    plot_source_plane_caustic_with_localization_from_setup(
        source_samples=samples,
        ctx=ctx,
        truths_source=truths,
        level=0.90,
        save_path=os.path.join(corner_dir, "source_localization_90.png"),
    )
    samples_by_method["nautilus-source"] = samples
    truths_by_method["nautilus-source"] = truths

METHODS = tuple(samples_by_method.keys())
if len(METHODS) < 2:
    print(f"skip comparison: need >=2 methods (have {len(METHODS)})")
else:
    shared = sorted(set.intersection(*(set(samples_by_method[m]) for m in METHODS)))
    if len(shared) < 2:
        print("skip comparison: fewer than 2 shared parameters")
    else:
        flat_truths = {}
        for t in truths_by_method.values():
            flat_truths.update(t)
        flat_truths.setdefault("y0gw", float(src[0]))
        flat_truths.setdefault("y1gw", float(src[1]))
        groups = {"all_params": shared}
        truths_dict = {
            "all_params": {p: float(flat_truths[p]) for p in shared if p in flat_truths}
        }
        plot_multi_comparison_corner(
            [samples_by_method[m] for m in METHODS],
            groups,
            labels=list(METHODS),
            colors=[METHOD_COLORS[m] for m in METHODS],
            truths_dict=truths_dict,
            save_path=os.path.join(OUTPUT_DIR, "comparison_{group_name}.png"),
            hist_kwargs={"density": True},
            plot_datapoints=False,
        )
        print(f"Comparison saved under {OUTPUT_DIR}/comparison_*.png")

print(f"\nDone. Outputs under {OUTPUT_DIR}/")
