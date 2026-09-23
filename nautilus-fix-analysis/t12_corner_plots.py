"""Corner plots for the tightened-prior tutorial GW-only runs.

Overlays fisher-source, nautilus-source (jit, serial) and nautilus-source
(jit + pool=4) on the same axes. 5 free parameters, so `combined` mode per the
plotting conventions, plus a 2-parameter source-localization subset.

The serial jit run predates the output-path fix, so its samples are recovered
from its nautilus checkpoint rather than re-run: nautilus stores unit-cube points
and maps them through the prior at posterior() time, so rebuilding the same
problem and reading the checkpoint reproduces exactly the posterior that run
produced.
"""

import os

import numpy as np
from common import OUT, save_json

import numpyro.distributions as dist
from gwemfish import run_inference, setup_em_observation, setup_gw_observation
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.nautilus_common import build_nautilus_problem
from t9_tutorial_gw_only_timed import apply_fisher_h0_priors, tutorial_cfg

SPAN = 1.0
PLOT_DIR = os.path.join(OUT, "t12_corners")
POOL_DIR = os.path.join(OUT, "t9_jit_pool4_span1")
SERIAL_CKPT = os.path.join(OUT, "t9_jit_span1", "nautilus_checkpoint.hdf5")
COLORS = {"fisher-source": "steelblue",
          "nautilus jit (serial)": "seagreen",
          "nautilus jit + pool4": "darkorange"}


def load_npz(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


def rebuild_ctx():
    ctx = setup_em_observation(cfg=tutorial_cfg(PLOT_DIR, SERIAL_CKPT))
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth = ctx["truth_params"]
    ctx["cfg"]["priors"] = {
        "lens1_gamma1": float(truth["lens1_gamma1"]),
        "lens1_gamma2": float(truth["lens1_gamma2"]),
        "lens1_ra_0": float(truth["lens1_ra_0"]),
        "lens1_dec_0": float(truth["lens1_dec_0"]),
        "lens0_theta_E": float(truth["lens0_theta_E"]),
        "lens0_center_x": float(truth["lens0_center_x"]),
        "lens0_center_y": float(truth["lens0_center_y"]),
        "lens0_phi": float(truth["lens0_phi"]),
        "lens0_gamma": float(truth["lens0_gamma"]),
        "y0gw": dist.Uniform(-0.6, 0.6),
        "y1gw": dist.Uniform(-0.6, 0.6),
    }
    ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (-0.6, 0.6), "y1gw": (-0.6, 0.6)}
    run_inference(ctx, mode="GW-only", method="fisher-source",
                  cfg={"priors": ctx["cfg"]["priors"],
                       "output": {"output_dir": PLOT_DIR}})
    apply_fisher_h0_priors(ctx, SPAN)
    return ctx


def samples_from_checkpoint(ctx, checkpoint):
    """Read a finished run's posterior back out of its checkpoint, without sampling."""
    import nautilus

    cfg = {"priors": ctx["cfg"]["priors"],
           "nautilus": {**ctx["cfg"]["nautilus"], "filepath": checkpoint,
                        "resume": True}}
    prior, loglike, names = build_nautilus_problem(ctx, cfg, "GW-only", "nautilus-source")
    sampler = nautilus.Sampler(prior, loglike, n_live=2000, filepath=checkpoint,
                               resume=True)
    points, _, _ = sampler.posterior(equal_weight=True)
    print(f"  recovered {points.shape[0]} samples, log_z {float(sampler.log_z):.4f}, "
          f"n_eff {float(sampler.n_eff):.1f}, n_like {int(sampler.n_like)}")
    return {name: np.asarray(points[:, j]) for j, name in enumerate(names)}


os.makedirs(PLOT_DIR, exist_ok=True)
ctx = rebuild_ctx()

print("\nrecovering the serial jit run from its checkpoint...")
serial = samples_from_checkpoint(ctx, SERIAL_CKPT)
np.savez(os.path.join(POOL_DIR, "..", "t9_jit_span1", "samples_nautilus_source.npz"),
         **serial)

results = [("fisher-source", load_npz(os.path.join(POOL_DIR, "samples_fisher_source.npz"))),
           ("nautilus jit (serial)", serial),
           ("nautilus jit + pool4",
            load_npz(os.path.join(POOL_DIR, "samples_nautilus_source.npz")))]

# y0gw/y1gw are never in truth_params -- the source position is backfilled only
# inside the source-plane builder, so merge it in last.
truths = {k: float(v) for k, v in
          load_npz(os.path.join(POOL_DIR, "truths_nautilus_source.npz")).items()}
src = ctx["cfg"]["gw"]["source_pos"]
truths["y0gw"] = float(src[0])
truths["y1gw"] = float(src[1])

shared = sorted(set.intersection(*(set(s) for s in (s for _, s in results))))
groups = {"all": shared, "source": ["y0gw", "y1gw"]}
print(f"\nshared parameters: {shared}")

plot_multi_comparison_corner(
    [s for _, s in results], groups,
    labels=[label for label, _ in results],
    colors=[COLORS[label] for label, _ in results],
    truths_dict=truths,
    save_path=os.path.join(PLOT_DIR, "comparison_{group_name}.png"),
)

header = "".join(f"{label:>34}" for label, _ in results)
print(f"\n{'param':>10} {'truth':>13}{header}")
table = {}
for k in shared:
    row = "".join(f"{np.mean(s[k]):18.6g}+-{np.std(s[k]):<14.4g}" for _, s in results)
    print(f"{k:>10} {truths.get(k, float('nan')):13.6g}{row}")
    table[k] = {"truth": truths.get(k),
                **{label: {"mean": float(np.mean(s[k])), "std": float(np.std(s[k]))}
                   for label, s in results}}

save_json("t12_corner_summary.json", {"parameters": table, "plot_dir": PLOT_DIR})
print(f"\nplots in {PLOT_DIR}")
