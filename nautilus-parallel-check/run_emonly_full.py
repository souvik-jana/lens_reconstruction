"""EM-only: prototype nautilus variants at full budget, timed against each other.

Priors come from the fisher precursor as a +-3.5 sigma box (fisher_h0), which the
sanity gate confirmed is buildable -- all 23 Fisher sigmas finite and positive.

A_baseline is absent by request. Note EM-only is the mode where jit is expected
to buy almost nothing: there is no lens equation to solve, and herculens already
compiles lens_image.model internally, so the likelihood starts at ~0.1 ms/call
rather than the 136 ms of GW-only. Here the win, if any, has to come from the
pool -- including nautilus' own neural-network training, which pool_s
parallelises. That makes this the honest stress case for the prototype.

    python run_emonly_full.py
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

from gwemfish import run_inference

from emonly_setup import (
    SIGMA_SPAN,
    build_emonly_ctx,
    emonly_priors,
    fisher_h0_priors,
)
from plot_single import plot_variant, summary_table
from proto.run_proto import run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_only")
os.makedirs(OUT_DIR, exist_ok=True)

N_WORKERS = 4
SEED = 1234
FULL_NAUTILUS = {"n_live": 2000, "n_eff": 5000, "n_like_max": 500_000}

VARIANTS = {
    "D_jit_pool": dict(jit=True, pool=N_WORKERS),
    "B_jit": dict(jit=True, pool=None),
    "C_pool": dict(jit=False, pool=N_WORKERS),
}
COLORS = {"D_jit_pool": "C0", "B_jit": "C1", "C_pool": "C2"}

# A 23x23 corner is unreadable and slow to render; these are the parameters the
# lens reconstruction is actually about.
CORNER_SUBSET = ["lens0_gamma", "lens0_theta_E", "lens0_e1", "lens0_e2",
                 "source0_center_x", "source0_center_y", "noise_sigma_bkg"]

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(VARIANTS)

    ctx = build_emonly_ctx(OUT_DIR)
    truth = ctx["truth_params"]
    PRIORS = emonly_priors(truth)

    print("\n--- fisher precursor (for fisher_h0 nautilus priors) ---", flush=True)
    run_inference(ctx, mode="EM-only", method="fisher",
                  cfg={"priors": PRIORS,
                       "output": {"output_dir": OUT_DIR, "json_tag": "fisher"}})

    print(f"\n--- fisher_h0 priors, span={SIGMA_SPAN} sigma ---")
    PRIORS, skipped = fisher_h0_priors(ctx, PRIORS, SIGMA_SPAN)
    if skipped:
        print(f"  skipped (unusable sigma): {skipped}")
    n_boxed = sum(1 for v in PRIORS.values() if hasattr(v, "low"))
    print(f"  {n_boxed} parameters given a +-{SIGMA_SPAN} sigma uniform box; "
          f"{len(PRIORS) - n_boxed} fixed to truth")

    cfg = {"priors": PRIORS}

    truths = {k: float(v) for k, v in truth.items()
              if np.ndim(v) == 0 and not k.startswith("image_")}
    json.dump(truths, open(os.path.join(OUT_DIR, "truths.json"), "w"), indent=2)

    results = {}
    for name in wanted:
        print(f"\n{'#' * 70}\n# EM-only, full budget -- {name}\n{'#' * 70}", flush=True)
        t0 = time.perf_counter()
        try:
            samples, diag = run(
                ctx, cfg, "EM-only",
                n_live=FULL_NAUTILUS["n_live"],
                n_eff=FULL_NAUTILUS["n_eff"],
                n_like_max=FULL_NAUTILUS["n_like_max"],
                seed=SEED, verbose=True,
                filepath=os.path.join(OUT_DIR, f"checkpoint_{name}.hdf5"),
                resume=True,
                **VARIANTS[name],
            )
        except Exception as exc:
            print(f"\n  {name} FAILED after {time.perf_counter() - t0:.1f}s: "
                  f"{type(exc).__name__}: {str(exc).splitlines()[-1][:160]}",
                  flush=True)
            continue
        diag["total_seconds"] = time.perf_counter() - t0
        results[name] = diag
        np.savez(os.path.join(OUT_DIR, f"EM-only_{name}.npz"), **samples)
        print(f"\n  {name}: wall {diag['wall_seconds']:.1f}s  "
              f"n_like {diag['n_like']}  n_eff {diag['n_eff']:.1f}  "
              f"log_z {diag['log_z']:.4f}  samples {diag['n_posterior_samples']}",
              flush=True)
        print(summary_table(name, samples, truths), flush=True)

        subset = {k: samples[k] for k in CORNER_SUBSET if k in samples}
        print(f"  corner plot: "
              f"{plot_variant(name, subset, truths, OUT_DIR, COLORS[name], diag)}",
              flush=True)

        path = os.path.join(OUT_DIR, "timings.json")
        existing = json.load(open(path)) if os.path.exists(path) else {}
        existing.update(results)
        json.dump(existing, open(path, "w"), indent=2)

    print(f"\n{'=' * 78}\nEM-only TIMING\n{'=' * 78}")
    print(f"  {'variant':<12} {'wall':>11} {'n_like':>9} {'n_eff':>10} {'log_z':>12}")
    for name, d in results.items():
        print(f"  {name:<12} {d['wall_seconds']:>10.1f}s {d['n_like']:>9d} "
              f"{d['n_eff']:>10.1f} {d['log_z']:>12.4f}")
    if "B_jit" in results and "D_jit_pool" in results:
        print(f"\n  pool speedup on the jitted likelihood: "
              f"{results['B_jit']['wall_seconds'] / results['D_jit_pool']['wall_seconds']:.2f}x")
    if "C_pool" in results and "D_jit_pool" in results:
        print(f"  jit speedup at equal pooling:           "
              f"{results['C_pool']['wall_seconds'] / results['D_jit_pool']['wall_seconds']:.2f}x")
