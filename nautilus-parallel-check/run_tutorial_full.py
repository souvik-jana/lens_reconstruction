"""GW-only tutorial at full nautilus settings, prototype variants only.

Reproduces ``tutorial/tutorial_gw_only.py`` exactly -- same CFG, same
``NAUTILUS_PRIOR_MODE="fisher_h0"`` with ``NAUTILUS_SIGMA_SPAN=3.5``, same
``n_live=2000, n_eff=5000, n_like_max=500_000`` -- and runs the three prototype
variants on it:

    B_jit       compiled likelihood, single process
    C_pool      today's eager likelihood, 4 workers
    D_jit_pool  both

A_baseline is deliberately absent: at 136 ms/call a full-budget run is roughly a
day, which is the whole reason this prototype exists.

All three share one seed, so the posteriors are comparable point for point and a
difference means the likelihood differs, not that nautilus drew differently.

The fisher precursor also puts ``ctx["fisher"]["approx_logp"]`` and
``ctx["likelihood"][...]`` into the ctx -- both jitted closures, both unpicklable.
That is exactly the case ``picklable.strip_ctx`` exists for, so this run is also
the test of it.

    python run_tutorial_full.py            # all three
    python run_tutorial_full.py D_jit_pool # one
"""

import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import numpyro.distributions as dist

from gwemfish import (
    make_default_cfg,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)

from plot_single import plot_variant, summary_table
from proto.run_proto import run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "tutorial_full")
os.makedirs(OUT_DIR, exist_ok=True)

N_WORKERS = 4
SEED = 1234
SIGMA_SPAN = 3.5

FULL_NAUTILUS = {"n_live": 2000, "n_eff": 5000, "n_like_max": 500_000}

# Fastest first, and each variant's samples are written as soon as it finishes,
# so a usable posterior exists early instead of only after the slowest variant.
VARIANTS = {
    "D_jit_pool": dict(jit=True, pool=N_WORKERS),
    "B_jit": dict(jit=True, pool=None),
    "C_pool": dict(jit=False, pool=N_WORKERS),
}

COLORS = {"D_jit_pool": "C0", "B_jit": "C1", "C_pool": "C2"}


def tutorial_cfg():
    """CFG block copied from tutorial/tutorial_gw_only.py."""
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = "q_phi"
    cfg["gw"]["n_images"] = 4
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["source_pos"] = (0.02, 0.01)
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["nautilus"] = {**FULL_NAUTILUS, "resume": False,
                       "prior_check": True, "verbose": True}
    cfg["output"]["output_dir"] = OUT_DIR
    return cfg


def tutorial_priors(truth):
    """PRIORS block copied from tutorial/tutorial_gw_only.py."""
    return {
        "lens1_gamma1": float(truth["lens1_gamma1"]),
        "lens1_gamma2": float(truth["lens1_gamma2"]),
        "lens1_ra_0": float(truth["lens1_ra_0"]),
        "lens1_dec_0": float(truth["lens1_dec_0"]),
        "lens0_phi": float(truth["lens0_phi"]),
        "lens0_q": dist.Uniform(0.75, 0.85),
        "lens0_theta_E": float(truth["lens0_theta_E"]),
        "lens0_center_x": float(truth["lens0_center_x"]),
        "lens0_center_y": float(truth["lens0_center_y"]),
        "lens0_gamma": dist.Uniform(1.5, 2.4),
        "T_star": float(truth["T_star"]),
        "dL": float(truth["dL"]),
        "y0gw": dist.Uniform(0.01992, 0.02005),
        "y1gw": dist.Uniform(0.0091, 0.0106),
    }


def fisher_h0_priors(ctx, priors, span):
    """The tutorial's NAUTILUS_PRIOR_MODE='fisher_h0': a box of +-span sigma
    around the Fisher peak, from the fisher-source precursor run."""
    keys = ctx["likelihood"]["keys_to_include"]
    u0 = np.asarray(ctx["likelihood"]["u0"])
    cov = np.linalg.pinv(-np.asarray(ctx["fisher"]["H0"]))
    sigmas = np.sqrt(np.diag(cov))

    out = dict(priors)
    for i, key in enumerate(keys):
        sig = float(sigmas[i])
        if not np.isfinite(sig) or sig <= 0:
            print(f"  prior {key}: skip (sigma={sig})")
            continue
        lo, hi = float(u0[i]) - span * sig, float(u0[i]) + span * sig
        out[key] = dist.Uniform(lo, hi)
        print(f"  prior {key}: Uniform({lo:.6g}, {hi:.6g})  [sigma={sig:.4g}]")
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(VARIANTS)

    CFG = tutorial_cfg()
    ctx = setup_em_observation(cfg=CFG)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth = ctx["truth_params"]
    print(f"Simulated {ctx['n_images']} GW images.")

    PRIORS = tutorial_priors(truth)
    ctx["cfg"]["priors"] = dict(PRIORS)
    ctx["cfg"]["gw"]["source_plane_bounds"] = {
        "y0gw": (0.01992, 0.02005), "y1gw": (0.0091, 0.0106)}

    print("\n--- fisher-source precursor (for fisher_h0 nautilus priors) ---")
    run_inference(ctx, mode="GW-only", method="fisher-source",
                  cfg={"priors": PRIORS,
                       "output": {"output_dir": OUT_DIR, "json_tag": "fisher_source"}})
    print(f"\n--- fisher_h0 priors, span={SIGMA_SPAN} sigma ---")
    PRIORS = fisher_h0_priors(ctx, PRIORS, SIGMA_SPAN)

    cfg = {"priors": PRIORS,
           "gw": {"source_plane_bounds": ctx["cfg"]["gw"]["source_plane_bounds"]}}

    # truth_params holds array-valued entries alongside the scalars; only the
    # scalars are plottable truth markers.
    truths = {k: float(v) for k, v in truth.items()
              if np.ndim(v) == 0 and not k.startswith("image_")}
    truths.setdefault("y0gw", float(CFG["gw"]["source_pos"][0]))
    truths.setdefault("y1gw", float(CFG["gw"]["source_pos"][1]))
    json.dump(truths, open(os.path.join(OUT_DIR, "truths.json"), "w"), indent=2)

    results = {}
    for name in wanted:
        print(f"\n{'#' * 70}\n# GW-only tutorial, full budget -- {name}\n{'#' * 70}",
              flush=True)
        t0 = time.perf_counter()
        try:
            samples, diag = run(
                ctx, cfg, "GW-only",
                n_live=FULL_NAUTILUS["n_live"],
                n_eff=FULL_NAUTILUS["n_eff"],
                n_like_max=FULL_NAUTILUS["n_like_max"],
                seed=SEED, verbose=True,
                # Each variant gets its own checkpoint and resumes from it, so a
                # crash hours in costs the remaining calls, not the whole run.
                filepath=os.path.join(OUT_DIR, f"checkpoint_{name}.hdf5"),
                resume=True,
                **VARIANTS[name],
            )
        except Exception as exc:
            # One variant dying must not take the other two with it -- they are
            # independent runs and the comparison is still worth having.
            print(f"\n  {name} FAILED after {time.perf_counter() - t0:.1f}s: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[-1][:160]}",
                  flush=True)
            print(f"  checkpoint kept at checkpoint_{name}.hdf5 -- rerun "
                  f"'python run_tutorial_full.py {name}' to continue.", flush=True)
            continue
        diag["total_seconds"] = time.perf_counter() - t0
        results[name] = diag
        np.savez(os.path.join(OUT_DIR, f"GW-only_{name}.npz"), **samples)
        print(f"\n  {name}: wall {diag['wall_seconds']:.1f}s  "
              f"n_like {diag['n_like']}  n_eff {diag['n_eff']:.1f}  "
              f"log_z {diag['log_z']:.4f}  samples {diag['n_posterior_samples']}",
              flush=True)

        # Plot this variant now rather than at the end: the slowest variant is
        # hours behind the fastest, and there is no reason to wait for it to look
        # at a finished posterior.
        print(summary_table(name, samples, truths), flush=True)
        print(f"  corner plot: "
              f"{plot_variant(name, samples, truths, OUT_DIR, COLORS[name], diag)}",
              flush=True)

        path = os.path.join(OUT_DIR, "timings.json")
        existing = json.load(open(path)) if os.path.exists(path) else {}
        existing.update(results)
        json.dump(existing, open(path, "w"), indent=2)

    print(f"\n{'=' * 70}\nDONE -- {OUT_DIR}\n{'=' * 70}")
    for name, d in results.items():
        print(f"  {name:<12} {d['wall_seconds']:>9.1f}s  n_like {d['n_like']:>7d}  "
              f"n_eff {d['n_eff']:>8.1f}  log_z {d['log_z']:>10.4f}")
