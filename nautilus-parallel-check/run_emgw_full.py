"""EM+GW: sanity gate, then the prototype nautilus variants, at full budget.

Order matters and is deliberate:

  1. fisher-source and deriv-approx-source (informed=True), then check every
     sigma for NaN / non-positive values and for divergence against Fisher.
     The fisher_h0 prior box is built from sqrt(diag(inv(Fisher))), so if those
     sigmas are unusable the box is garbage and nautilus would spend hours
     sampling nonsense.
  2. Only if the box is buildable, run nautilus D_jit_pool / B_jit / C_pool.

informed=True on deriv-approx-source because the default
(cfg["inference"]["informed"] = None) is read as False by
simple_pipeline.py:2229 and gives plain NUTS -- which diverged badly on EM-only
(r_hat ~1e15). EM+GW mixes light amplitudes with arcsecond-scale geometry the
same way, so the Fisher mass matrix is wanted here too.

This run is also the outstanding test of the 'Signs are not different' solver
failure, which has only ever occurred once, in a scaled-down unseeded EM+GW run
(bench_proto.log:517, 2026-09-17 21:48). EM+GW is the mode that produced it and
the only one not yet re-run at full budget. Seeded here, so the outcome is
reproducible either way.

    python run_emgw_full.py             # gate + all three variants
    python run_emgw_full.py D_jit_pool  # gate + one variant
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

from gwemfish import run_inference
from gwemfish.diagnostics import check_conditioning

# fisher_sigmas / fisher_h0_priors take only a ctx -- nothing about them is
# EM-only specific.
from emonly_setup import SIGMA_SPAN, fisher_h0_priors, fisher_sigmas
from plot_single import plot_variant, summary_table
from proto.run_proto import run
from tutorial_cfg import build_ctx, source_bounds

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_gw")
os.makedirs(OUT_DIR, exist_ok=True)

MODE = "EM+GW"
N_WORKERS = 4
SEED = 1234
FULL_NAUTILUS = {"n_live": 2000, "n_eff": 5000, "n_like_max": 500_000}
GRADIENT_METHODS = ["fisher-source", "deriv-approx-source"]

# T_star and dL are pinned, as tutorial_em_gw.py does. Freeing them on this
# 2-image system makes the Fisher matrix singular: 2 images give 1 time delay +
# 2 dL_eff = 3 GW observables, and T_star enters the time delay only as the
# product T_star * delta-phi(lens), so T_star and the profile slope lens0_gamma
# are one unknown to the data, not two. Measured with them free:
#
#   UserWarning: Fisher covariance is not positive definite (min eigenvalue
#   -5.461e+22) ... near-degenerate direction (weakest parameter: lens0_gamma)
#
# Pinning them closes that degeneracy. Freeing them needs more images or an
# informative T_star prior, not a code change.
FIXED_TO_TRUTH = ["lens1_ra_0", "lens1_dec_0", "dL", "T_star"]

VARIANTS = {
    "D_jit_pool": dict(jit=True, pool=N_WORKERS),
    "B_jit": dict(jit=True, pool=None),
    "C_pool": dict(jit=False, pool=N_WORKERS),
}
COLORS = {"D_jit_pool": "C0", "B_jit": "C1", "C_pool": "C2"}

CORNER_SUBSET = ["lens0_gamma", "lens0_theta_E", "lens0_e1", "lens0_e2",
                 "source0_center_x", "source0_center_y", "noise_sigma_bkg"]

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(VARIANTS)

    ctx = build_ctx(MODE)
    truth = ctx["truth_params"]
    bounds = source_bounds(ctx, MODE)
    PRIORS = {
        **{k: float(truth[k]) for k in FIXED_TO_TRUTH},
        "y0gw": dist.Uniform(*bounds["y0gw"]),
        "y1gw": dist.Uniform(*bounds["y1gw"]),
    }
    ctx["cfg"]["priors"] = dict(PRIORS)
    ctx["cfg"]["gw"]["source_plane_bounds"] = source_bounds(ctx, MODE)
    print(f"{MODE}: {ctx['n_images']} GW images. Fixed to truth: "
          f"{[k for k, v in PRIORS.items() if not hasattr(v, 'low')]}")

    truths = {k: float(v) for k, v in truth.items()
              if np.ndim(v) == 0 and not k.startswith("image_")}
    json.dump(truths, open(os.path.join(OUT_DIR, "truths.json"), "w"), indent=2)

    # ---- 1. sanity gate -------------------------------------------------
    grad_samples = {}
    for method in GRADIENT_METHODS:
        print(f"\n--- {MODE}: {method} ---", flush=True)
        samples, _ = run_inference(
            ctx, mode=MODE, method=method,
            cfg={"priors": PRIORS,
                 "inference": ({"informed": True}
                               if method == "deriv-approx-source" else {}),
                 "output": {"output_dir": OUT_DIR,
                            "json_tag": method.replace("-", "_")}},
        )
        grad_samples[method] = {k: np.asarray(v) for k, v in samples.items()}
        np.savez(os.path.join(OUT_DIR, f"{MODE.replace('+', '_')}_{method}.npz"),
                 **grad_samples[method])

    keys, u0, sigmas = fisher_sigmas(ctx)
    bad = [(k, float(s)) for k, s in zip(keys, sigmas)
           if not np.isfinite(s) or s <= 0]

    # Why a sigma goes bad, not just that it did. A NaN or huge sigma means the
    # Fisher matrix is near-singular: some combination of parameters the data
    # cannot constrain. The condition number says how bad, and the eigenvector of
    # the smallest scaled eigenvalue says which combination -- which is the
    # actionable part. Scaling by each parameter's own 1-sigma first is essential;
    # T_star is O(1e10) and ellipticities are O(0.01), so the raw condition number
    # measures unit mismatch rather than degeneracy.
    H0 = np.asarray(ctx["fisher"]["H0"])
    cond = check_conditioning(H0, keys)
    print(f"\n{'=' * 90}\nFISHER CONDITIONING\n{'=' * 90}")
    print(f"  scaled condition number : {cond['condition_number']:.3e} "
          f"(limit {cond['condition_limit']:.1e})")
    print(f"  positive eigenvalues    : {cond['positive_eigenvalues']} "
          f"(should be 0 at a maximum)")
    print(f"  near-zero eigenvalues   : {cond['near_zero_eigenvalues']}")

    scale = 1.0 / np.sqrt(np.abs(np.diag(H0)))
    eigval, eigvec = np.linalg.eigh(H0 * scale[:, None] * scale[None, :])
    flattest = int(np.argmin(np.abs(eigval)))
    weights = np.abs(eigvec[:, flattest])
    print(f"\n  flattest direction (scaled eigenvalue {eigval[flattest]:.3e}) "
          f"-- the combination the data cannot pin down:")
    for i in np.argsort(weights)[::-1][:6]:
        print(f"    {keys[i]:<22} weight {eigvec[i, flattest]:+.3f}")

    print(f"\n{'=' * 90}\nFISHER SIGMAS  ({len(keys)} free parameters)\n{'=' * 90}")
    print(f"  {'param':<22} {'u0':>14} {'sigma':>14}  status")
    for k, c, s in zip(keys, u0, sigmas):
        status = ("NaN/inf" if not np.isfinite(s)
                  else "NON-POSITIVE" if s <= 0 else "ok")
        print(f"  {k:<22} {float(c):>14.6g} {float(s):>14.6g}  {status}")

    print(f"\n{'=' * 90}\nPOSTERIOR SIGMAS\n{'=' * 90}")
    shared = [p for p in grad_samples[GRADIENT_METHODS[0]]
              if all(p in s for s in grad_samples.values())]
    sigma_ref = dict(zip(keys, sigmas))
    print(f"  {'param':<22}" + "".join(f"{m:>24}" for m in GRADIENT_METHODS)
          + "   flag")
    nan_hits, diverged = [], []
    for p in shared:
        cells, flag = [], ""
        for m in GRADIENT_METHODS:
            v = grad_samples[m][p]
            sd = float(np.std(v))
            cells.append(sd)
            if not np.isfinite(sd) or not np.all(np.isfinite(v)):
                flag = "  <-- NaN"
                nan_hits.append((m, p))
            ref = sigma_ref.get(p)
            if ref and np.isfinite(ref) and ref > 0 and np.isfinite(sd):
                ratio = sd / ref
                if ratio > 10 or ratio < 0.1:
                    flag = f"  <-- {ratio:.3g}x Fisher"
                    diverged.append((m, p, ratio))
        print(f"  {p:<22}" + "".join(f"{c:>24.6g}" for c in cells) + flag)

    print(f"\n{'=' * 90}\nGATE\n{'=' * 90}")
    print(f"  Fisher sigmas unusable : {len(bad)}  {bad if bad else ''}")
    print(f"  NaN in samples         : {len(nan_hits)}  {nan_hits if nan_hits else ''}")
    print(f"  diverged (>10x Fisher) : {len(diverged)}  "
          f"{diverged[:5] if diverged else ''}")

    # Nautilus runs only on a clean gate. Any of these three means the fisher_h0
    # box or the gradient methods are untrustworthy, and hours of sampling on a
    # bad prior is worse than stopping here.
    if bad or nan_hits or diverged:
        reasons = []
        if bad:
            reasons.append(f"{len(bad)} unusable Fisher sigma(s): {bad}")
        if nan_hits:
            reasons.append(f"NaN in samples: {nan_hits}")
        if diverged:
            reasons.append(f"{len(diverged)} diverged parameter(s): {diverged[:5]}")
        raise SystemExit("\n  STOPPING before nautilus:\n    "
                         + "\n    ".join(reasons)
                         + "\n\n  See the conditioning block above for which "
                           "parameter combination is unconstrained.")
    print(f"\n  gate clean -> proceeding to nautilus")

    # ---- 2. nautilus prototypes ----------------------------------------
    PRIORS, skipped = fisher_h0_priors(ctx, PRIORS, SIGMA_SPAN)
    if skipped:
        print(f"  skipped (unusable sigma): {skipped}")
    n_boxed = sum(1 for v in PRIORS.values() if hasattr(v, "low"))
    print(f"  {n_boxed} parameters given a +-{SIGMA_SPAN} sigma uniform box; "
          f"{len(PRIORS) - n_boxed} fixed to truth")

    cfg = {"priors": PRIORS,
           "gw": {"source_plane_bounds": ctx["cfg"]["gw"]["source_plane_bounds"]}}

    results = {}
    for name in wanted:
        print(f"\n{'#' * 70}\n# {MODE}, full budget -- {name}\n{'#' * 70}", flush=True)
        t0 = time.perf_counter()
        try:
            samples, diag = run(
                ctx, cfg, MODE,
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
                  f"{type(exc).__name__}: {str(exc).splitlines()[-1][:200]}",
                  flush=True)
            continue
        diag["total_seconds"] = time.perf_counter() - t0
        results[name] = diag
        np.savez(os.path.join(OUT_DIR, f"{MODE.replace('+', '_')}_{name}.npz"),
                 **samples)
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

    print(f"\n{'=' * 78}\n{MODE} TIMING\n{'=' * 78}")
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
