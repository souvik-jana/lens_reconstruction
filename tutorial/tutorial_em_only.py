"""
EM-only tutorial: fisher, deriv-approx, nautilus-source.

Flip RUN_* toggles to enable/disable methods. Comparison plots only run when
two or more methods produced samples (no error if a method is off).

fisher-source / deriv-approx-source are not valid for EM-only (no GW source).

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
    plot_posterior,
    plot_psf,
    plot_system_observation,
    prune_gw_images,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish.fisher import invert_fisher_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RUN_FISHER = True
RUN_DERIV_APPROX = True
RUN_NAUTILUS_SOURCE = False

NAUTILUS_RESUME = False
NAUTILUS_PRIOR_MODE = "fisher_h0"  # "fisher_h0" | "manual"
NAUTILUS_SIGMA_SPAN = 5.0

METHOD_COLORS = {
    "fisher": "C2",
    "deriv-approx": "C0",
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
CFG["lens_mass_parametrization"] = "q_phi"#"e1e2"#"q_phi"  # "e1e2" to go back
OUTPUT_DIR = os.path.join(
    REPO_ROOT, "tutorial", "outputs", "em_only", CFG["lens_mass_parametrization"]
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
# CFG["gw"]["n_images"] = 4#2 # no need     for em only
# CFG["gw"]["source_box_half_width"] = 0.8
# CFG["source_plane"]["n_images"] = 2 # no need for em only
# CFG["gw"]["source_pos"] = (0.02,0.01) #(0.2, 0.01) this is for 2 image configuration # no need for em only
# CFG["gw"]["error_scales"]["sigma_td"] = 0.001
# CFG["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
CFG["inference"]["num_chains"] = 10
CFG["inference"]["num_samples"] = 14000
CFG["inference"]["num_warmup"] = 6000
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
        ("fisher", RUN_FISHER),
        ("deriv-approx", RUN_DERIV_APPROX),
        ("nautilus-source", RUN_NAUTILUS_SOURCE),
    ]
    if flag
]
print(f"Active methods: {', '.join(active) if active else '(none)'}")
print(f"Nautilus prior mode: {NAUTILUS_PRIOR_MODE}, resume={NAUTILUS_RESUME}")

ctx = setup_em_observation(cfg=CFG)
# ctx = setup_gw_observation(ctx, cfg=ctx["cfg"]) # no need for em only
# ctx = prune_gw_images(ctx, n_keep=2) # Turn this on for 2 image configuration

truth_params = ctx["truth_params"]
# print(f"Simulated {ctx['n_images']} GW images (EM-only inference ignores GW likelihood).")

plot_system_observation(
    ctx,
    cfg={"output": {"output_dir": OUTPUT_DIR, "save_system_plot_path": "system_observation.png"}},
)
plot_psf(ctx, cfg={"output": {"output_dir": OUTPUT_DIR, "save_psf_plot_path": "psf.png"}})

PRIORS = {
    "lens1_ra_0": float(truth_params["lens1_ra_0"]),
    "lens1_dec_0": float(truth_params["lens1_dec_0"]),
}
ctx["cfg"]["priors"] = dict(PRIORS)

samples_by_method = {}
truths_by_method = {}

if RUN_FISHER:
    print("\n--- EM-only: fisher ---\n")
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
        method="fisher",
        cfg={
            "priors": PRIORS,
            "output": {"output_dir": OUTPUT_DIR, "json_tag": "fisher"},
        },
    )
    corner_dir = os.path.join(OUTPUT_DIR, "fisher")
    os.makedirs(corner_dir, exist_ok=True)
    plot_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": "posterior_{group_name}.png",
            },
        },
    )
    samples_by_method["fisher"] = samples
    truths_by_method["fisher"] = truths

if RUN_DERIV_APPROX:
    print("\n--- EM-only: deriv-approx ---\n")
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
        method="deriv-approx",
        cfg={
            "priors": PRIORS,
            "output": {"output_dir": OUTPUT_DIR, "json_tag": "deriv_approx"},
            "inference": {"informed": True},
        },
    )
    corner_dir = os.path.join(OUTPUT_DIR, "deriv_approx")
    os.makedirs(corner_dir, exist_ok=True)
    plot_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": "posterior_{group_name}.png",
            },
        },
    )
    samples_by_method["deriv-approx"] = samples
    truths_by_method["deriv-approx"] = truths

if RUN_NAUTILUS_SOURCE:
    if NAUTILUS_PRIOR_MODE == "fisher_h0":
        if "fisher" not in ctx or "likelihood" not in ctx:
            print("\n--- Precursor fisher for H0 (nautilus only) ---\n")
            run_inference(
                ctx,
                mode="EM-only",
                method="fisher",
                cfg={
                    "priors": PRIORS,
                    "output": {"output_dir": OUTPUT_DIR, "json_tag": "fisher_h0_precursor"},
                },
            )
        print(f"\n--- Nautilus priors from Fisher H0 (span={NAUTILUS_SIGMA_SPAN}) ---\n")
        apply_fisher_h0_priors(ctx, NAUTILUS_SIGMA_SPAN)
    elif NAUTILUS_PRIOR_MODE == "manual":
        print("\n--- Nautilus manual priors (hand-set PRIORS only) ---\n")
        ctx["cfg"]["priors"] = dict(PRIORS)
    else:
        raise ValueError(f"unknown NAUTILUS_PRIOR_MODE={NAUTILUS_PRIOR_MODE!r}")

    print(f"\n--- EM-only: nautilus-source (resume={NAUTILUS_RESUME}) ---\n")
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
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
    corner_dir = os.path.join(OUTPUT_DIR, "nautilus_source")
    os.makedirs(corner_dir, exist_ok=True)
    plot_posterior(
        samples,
        truths=truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": "posterior_{group_name}.png",
            },
        },
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
        ref = samples_by_method[METHODS[0]]
        groups = create_default_param_groups({k: ref[k] for k in shared})
        groups = {g: [p for p in ps if p in shared] for g, ps in groups.items()}
        groups = {g: ps for g, ps in groups.items() if len(ps) >= 2}
        if not groups:
            groups = {"all_params": shared}
        truths_dict = {
            g: {p: float(flat_truths[p]) for p in ps if p in flat_truths}
            for g, ps in groups.items()
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
