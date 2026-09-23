"""Verification 3b -- the real tutorial GW-only run, timed, three ways.

Config lifted verbatim from ``tutorial/tutorial_gw_only.py``: same q/phi
parametrisation, same 4 images, same jaxtronomy analytical solver, same error
scales, same fisher_h0 priors at 3.5 sigma, same full budget
(n_live=2000, n_eff=5000, n_like_max=500_000). The tutorial file itself is not
touched and its checkpoint is not overwritten -- everything lands under
nautilus-fix-analysis/outputs/t9_<variant>/.

  .venv/bin/python t9_tutorial_gw_only_timed.py eager     # today's path, ~1.9 h
  .venv/bin/python t9_tutorial_gw_only_timed.py jit
  .venv/bin/python t9_tutorial_gw_only_timed.py jit_pool4

The script body is guarded because `spawn` re-imports it in every pool worker.
"""

import json
import os
import sys
import time

import numpy as np
import numpyro.distributions as dist
from common import OUT, save_json

from gwemfish import (
    make_default_cfg,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.fisher import invert_fisher_matrix

VARIANTS = {"eager": {"jit": False}, "jit": {"jit": True},
            "jit_pool4": {"jit": True, "pool": 4}}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "jit"
SEED = 7
# The tutorial uses 3.5. At that span the Fisher boxes for T_star and dL run
# NEGATIVE (T_star [-1.7e7, 3.2e7], dL [-1.1e4, 4.3e4]) and lens0_q exceeds 1, so
# most of the prior volume is unphysical and nautilus explores it forever:
# n_eff 69 after 181,700 calls. Pass a span as the second argument.
SIGMA_SPAN = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
# Third argument caps n_like_max. The eager path costs ~160 ms/call, so measuring
# its wall time over the full 178,500 calls this config needs would take ~8 h;
# capping both paths at the same call count gives the same ratio in one hour.
N_LIKE_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 500_000


def tutorial_cfg(output_dir, checkpoint):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = "q_phi"
    cfg["gw"]["n_images"] = 4
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["source_pos"] = (0.02, 0.00001)
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["nautilus"] = {"n_live": 2000, "n_eff": 5000, "n_like_max": N_LIKE_MAX,
                       "filepath": checkpoint, "resume": False,
                       "prior_check": True, "verbose": True, "seed": SEED,
                       **VARIANTS[VARIANT]}
    cfg["output"]["output_dir"] = output_dir
    return cfg


def apply_fisher_h0_priors(ctx, span):
    """Verbatim from the tutorial: nautilus priors from the Fisher covariance."""
    keys = ctx["likelihood"]["keys_to_include"]
    u0 = np.asarray(ctx["likelihood"]["u0"])
    H0 = np.asarray(ctx["fisher"]["H0"])
    regularize = bool((ctx.get("cfg") or {}).get("inference", {}).get("regularize", False))
    sigmas = np.sqrt(np.diag(np.asarray(invert_fisher_matrix(-H0, regularize=regularize))))
    for i, key in enumerate(keys):
        sig = float(sigmas[i])
        if not np.isfinite(sig) or sig <= 0:
            continue
        mu = float(u0[i])
        ctx["cfg"]["priors"][key] = dist.Uniform(mu - span * sig, mu + span * sig)


if __name__ == "__main__":
    output_dir = os.path.join(OUT, f"t9_{VARIANT}_span{SIGMA_SPAN:g}_cap{N_LIKE_MAX}")
    checkpoint = os.path.join(output_dir, "nautilus_checkpoint.hdf5")
    os.makedirs(output_dir, exist_ok=True)
    print(f"variant {VARIANT}: {VARIANTS[VARIANT]}  ->  {output_dir}", flush=True)

    ctx = setup_em_observation(cfg=tutorial_cfg(output_dir, checkpoint))
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth_params = ctx["truth_params"]

    Y0_LO, Y0_HI = -0.6, 0.6
    Y1_LO, Y1_HI = -0.6, 0.6
    ctx["cfg"]["priors"] = {
        "lens1_gamma1": float(truth_params["lens1_gamma1"]),
        "lens1_gamma2": float(truth_params["lens1_gamma2"]),
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        "lens0_theta_E": float(truth_params["lens0_theta_E"]),
        "lens0_center_x": float(truth_params["lens0_center_x"]),
        "lens0_center_y": float(truth_params["lens0_center_y"]),
        "lens0_e2": float(truth_params["lens0_e2"]),
        "lens0_phi": float(truth_params["lens0_phi"]),
        "lens0_gamma": float(truth_params["lens0_gamma"]),
        "y0gw": dist.Uniform(Y0_LO, Y0_HI),
        "y1gw": dist.Uniform(Y1_LO, Y1_HI),
    }
    ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (Y0_LO, Y0_HI),
                                               "y1gw": (Y1_LO, Y1_HI)}

    timings = {}
    print("\n--- fisher-source (unchanged by this work; also builds the priors) ---\n",
          flush=True)
    t0 = time.perf_counter()
    fisher_samples, _ = run_inference(
        ctx, mode="GW-only", method="fisher-source",
        cfg={"priors": ctx["cfg"]["priors"],
             "output": {"output_dir": output_dir, "json_tag": "fisher_source",
                        "save_samples_path": "samples.npz",
                        "json_path": "pipeline_outputs.json"}})
    timings["fisher-source"] = time.perf_counter() - t0
    print(f"  fisher-source: {timings['fisher-source']:.1f}s", flush=True)

    apply_fisher_h0_priors(ctx, SIGMA_SPAN)

    print(f"\n--- nautilus-source, variant {VARIANT} ---\n", flush=True)
    t0 = time.perf_counter()
    samples, truths = run_inference(
        ctx, mode="GW-only", method="nautilus-source",
        cfg={"priors": ctx["cfg"]["priors"],
             "nautilus": {"filepath": checkpoint, "resume": False,
                          "prior_check": True},
             "output": {"output_dir": output_dir, "json_tag": "nautilus_source",
                        "save_samples_path": "samples.npz",
                        "save_truths_path": "truths.npz",
                        "json_path": "pipeline_outputs.json"}})
    timings["nautilus-source"] = time.perf_counter() - t0
    print(f"  nautilus-source ({VARIANT}): {timings['nautilus-source']:.1f}s", flush=True)

    with open(os.path.join(output_dir, "pipeline_outputs_nautilus_source.json")) as f:
        diagnostics = json.load(f).get("sampler_diagnostics")

    save_json(f"t9_{VARIANT}_span{SIGMA_SPAN:g}_cap{N_LIKE_MAX}.json", {
        "variant": VARIANT, "settings": VARIANTS[VARIANT], "seed": SEED,
        "sigma_span": SIGMA_SPAN, "n_like_max": N_LIKE_MAX,
        "timings": timings, "sampler_diagnostics": diagnostics,
        "posterior_means": {k: float(np.mean(v)) for k, v in samples.items()},
        "posterior_stds": {k: float(np.std(v)) for k, v in samples.items()},
        "n_samples": int(len(next(iter(samples.values())))),
        "fisher_means": {k: float(np.mean(v)) for k, v in fisher_samples.items()},
    })
    print(f"\nt9 {VARIANT} done: nautilus {timings['nautilus-source']:.1f}s, "
          f"diagnostics {diagnostics}")
