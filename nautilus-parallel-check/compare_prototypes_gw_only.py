"""
Apply both speed prototypes (picklable-pool multiprocessing, helens backend)
to the SAME problem as quick_check_gw_only.py (tutorial's current CFG/priors,
analytical jaxtronomy solver), with the SAME fisher-H0-derived nautilus
priors and SAME sampling budget, and time all three back-to-back in one
process run for a fair, apples-to-apples comparison:

  A. current (unmodified) gwemfish nautilus-source, single process
  B. prototype: de-closured/picklable log_likelihood, pool=4 (spawn)
  C. prototype: helens backend, single process

No src/gwemfish/ changes.
"""

import copy
import multiprocessing as mp
import os
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX devices: {jax.devices()}")

import numpy as np
import numpyro.distributions as dist

from gwemfish import (
    make_default_cfg,
    plot_source_posterior,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.fisher import invert_fisher_matrix
from gwemfish.nautilus_source_inference import build_gw_source_plane_problem

N_LIVE = 200
N_EFF = 500
N_LIKE_MAX = 4000
N_WORKERS = 4

Y0_LO, Y0_HI = 0.01992, 0.02005
Y1_LO, Y1_HI = 0.0091, 0.0106

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "nautilus-parallel-check", "outputs", "compare_prototypes")


def base_cfg(backend, jaxtronomy_solver="analytical"):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = "q_phi"
    cfg["gw"]["n_images"] = 4
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["source_pos"] = (0.02, 0.01)
    cfg["gw"]["solver_params"]["backend"] = backend
    if backend == "jaxtronomy":
        cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = jaxtronomy_solver
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["nautilus"] = {
        "n_live": N_LIVE, "n_eff": N_EFF, "n_like_max": N_LIKE_MAX,
        "filepath": None, "resume": False, "prior_check": False, "verbose": True,
    }
    cfg["output"]["output_dir"] = OUTPUT_DIR
    return cfg


def apply_fisher_h0_priors(ctx, span):
    keys = ctx["likelihood"]["keys_to_include"]
    u0 = np.asarray(ctx["likelihood"]["u0"])
    H0 = np.asarray(ctx["fisher"]["H0"])
    regularize = bool((ctx.get("cfg") or {}).get("inference", {}).get("regularize", False))
    cov = np.asarray(invert_fisher_matrix(-H0, regularize=regularize))
    sigmas = np.sqrt(np.diag(cov))
    updated = {}
    for i, key in enumerate(keys):
        sig = float(sigmas[i])
        if not np.isfinite(sig) or sig <= 0:
            print(f"  Nautilus prior {key}: skip (sigma={sig}) — keep existing prior")
            continue
        mu = float(u0[i])
        lo, hi = mu - span * sig, mu + span * sig
        updated[key] = dist.Uniform(lo, hi)
        print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")
    return updated


def build_priors(truth_params):
    return {
        "lens1_gamma1": float(truth_params["lens1_gamma1"]),
        "lens1_gamma2": float(truth_params["lens1_gamma2"]),
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        "lens0_phi": float(truth_params["lens0_phi"]),
        "lens0_q": dist.Uniform(0.75, 0.85),
        "lens0_theta_E": float(truth_params["lens0_theta_E"]),
        "lens0_center_x": float(truth_params["lens0_center_x"]),
        "lens0_center_y": float(truth_params["lens0_center_y"]),
        "lens0_gamma": dist.Uniform(1.5, 2.4),
        "T_star": float(truth_params["T_star"]),
        "dL": float(truth_params["dL"]),
        "y0gw": dist.Uniform(Y0_LO, Y0_HI),
        "y1gw": dist.Uniform(Y1_LO, Y1_HI),
    }


# ---- prototype B: picklable-by-reconstruction likelihood (same trick as
# bench_picklable_pool.py) ----
_WORKER_LOGLIKE = None


def _worker_build(cfg_full, priors_cfg):
    global _WORKER_LOGLIKE
    if _WORKER_LOGLIKE is None:
        t0 = time.perf_counter()
        ctx = setup_em_observation(cfg=cfg_full)
        ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
        ctx["cfg"]["priors"] = priors_cfg
        _, loglike, _ = build_gw_source_plane_problem(ctx, {"priors": priors_cfg})
        print(f"    [pid {os.getpid()}] worker rebuild: {time.perf_counter()-t0:.1f}s", flush=True)
        _WORKER_LOGLIKE = loglike
    return _WORKER_LOGLIKE


class PicklableGWSourceLikelihood:
    def __init__(self, cfg_full, priors_cfg):
        self.cfg_full = cfg_full
        self.priors_cfg = priors_cfg

    def __call__(self, params):
        return _worker_build(self.cfg_full, self.priors_cfg)(params)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULTS = {}
    samples_by_variant = {}

    # ---- shared setup: analytical jaxtronomy solver, same as tutorial ----
    CFG = base_cfg("jaxtronomy", "analytical")
    ctx = setup_em_observation(cfg=CFG)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth_params = ctx["truth_params"]
    src = ctx["cfg"]["gw"]["source_pos"]

    PRIORS = build_priors(truth_params)
    ctx["cfg"]["priors"] = dict(PRIORS)
    ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (Y0_LO, Y0_HI), "y1gw": (Y1_LO, Y1_HI)}

    print("\n--- precursor fisher-source (for Fisher-H0 nautilus priors) ---\n")
    run_inference(
        ctx, mode="GW-only", method="fisher-source",
        cfg={"priors": PRIORS, "output": {"output_dir": OUTPUT_DIR, "json_tag": "fisher_precursor"}},
    )
    print("\n--- Fisher-H0 priors (span=3.5, same as tutorial) ---\n")
    updates = apply_fisher_h0_priors(ctx, 3.5)
    FINAL_PRIORS = dict(PRIORS)
    FINAL_PRIORS.update(updates)
    ctx["cfg"]["priors"] = dict(FINAL_PRIORS)

    print(f"\nBudget for all 3 variants: n_live={N_LIVE}, n_eff={N_EFF}, n_like_max={N_LIKE_MAX}\n")

    # ---- A: current gwemfish nautilus-source, single process ----
    print("\n=== A: current (single-process) nautilus-source ===\n")
    t0 = time.perf_counter()
    samples_a, truths_a = run_inference(
        ctx, mode="GW-only", method="nautilus-source",
        cfg={"priors": FINAL_PRIORS, "output": {"output_dir": OUTPUT_DIR, "json_tag": "A_current"}},
    )
    RESULTS["A_current_single_process"] = time.perf_counter() - t0
    print(f"[TIMING] A current (single-process): {RESULTS['A_current_single_process']:.1f}s")
    samples_by_variant["A_current"] = samples_a

    # ---- B: picklable-pool prototype ----
    print("\n=== B: prototype picklable-pool (pool=4, spawn) ===\n")
    import nautilus

    prior, _discard, param_names = build_gw_source_plane_problem(ctx, {"priors": FINAL_PRIORS})
    picklable_loglike = PicklableGWSourceLikelihood(ctx["cfg"], FINAL_PRIORS)
    t0 = time.perf_counter()
    sampler = nautilus.Sampler(
        prior, picklable_loglike, n_live=N_LIVE, pool=N_WORKERS, filepath=None, resume=False,
    )
    sampler.run(verbose=False, n_eff=N_EFF, n_like_max=N_LIKE_MAX)
    points, _, _ = sampler.posterior(equal_weight=True)
    samples_b = {name: np.array(points[:, j]) for j, name in enumerate(param_names)}
    RESULTS["B_prototype_pool4"] = time.perf_counter() - t0
    print(f"[TIMING] B prototype pool=4: {RESULTS['B_prototype_pool4']:.1f}s")
    samples_by_variant["B_pool4"] = samples_b

    # ---- C: helens backend prototype, single process ----
    print("\n=== C: prototype helens backend (single process) ===\n")
    CFG_HELENS = base_cfg("helens")
    ctx_helens = setup_em_observation(cfg=CFG_HELENS)
    ctx_helens = setup_gw_observation(ctx_helens, cfg=ctx_helens["cfg"])
    tp_helens = ctx_helens["truth_params"]
    priors_helens = build_priors(tp_helens)
    priors_helens.update(updates)  # same Fisher-H0-derived y0gw prior
    ctx_helens["cfg"]["priors"] = dict(priors_helens)
    ctx_helens["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (Y0_LO, Y0_HI), "y1gw": (Y1_LO, Y1_HI)}

    t0 = time.perf_counter()
    samples_c, truths_c = run_inference(
        ctx_helens, mode="GW-only", method="nautilus-source",
        cfg={"priors": priors_helens, "output": {"output_dir": OUTPUT_DIR, "json_tag": "C_helens"}},
    )
    RESULTS["C_prototype_helens"] = time.perf_counter() - t0
    print(f"[TIMING] C prototype helens: {RESULTS['C_prototype_helens']:.1f}s")
    samples_by_variant["C_helens"] = samples_c

    # ---- comparison ----
    print("\n=== TIMING COMPARISON (same priors, same budget) ===")
    baseline = RESULTS["A_current_single_process"]
    for name, dt in RESULTS.items():
        print(f"  {name}: {dt:.1f}s  ({baseline/dt:.2f}x vs A)")

    shared = sorted(set.intersection(*(set(samples_by_variant[m]) for m in samples_by_variant)))
    if len(shared) >= 2:
        flat_truths = dict(truths_a)
        flat_truths.setdefault("y0gw", float(src[0]))
        flat_truths.setdefault("y1gw", float(src[1]))
        plot_multi_comparison_corner(
            [samples_by_variant[m] for m in samples_by_variant],
            {"all": shared},
            labels=list(samples_by_variant.keys()),
            colors=["C1", "C2", "C3"],
            truths_dict={"all": {p: float(flat_truths[p]) for p in shared if p in flat_truths}},
            save_path=os.path.join(OUTPUT_DIR, "comparison_variants_{group_name}.png"),
            hist_kwargs={"density": True},
            plot_datapoints=False,
        )
        print(f"Saved: {OUTPUT_DIR}/comparison_variants_all.png")

    print("\nDone.")
