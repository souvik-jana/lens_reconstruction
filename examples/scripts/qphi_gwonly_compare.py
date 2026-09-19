"""GW-only (variants A and B): q_phi vs e1e2, compared in q/phi space.

Per variant:
  1. lens_mass_parametrization="q_phi"  -> save posterior
  2. lens_mass_parametrization="e1e2"   -> save posterior
  3. convert the e1e2 run to (q, phi) and overlay on the direct q/phi run

DEGREE-OF-FREEDOM ASYMMETRY, deliberate and unavoidable: in q_phi mode lens0_phi
is fixed to truth (GW time delays constrain orientation poorly). e1e2 mode has no
phi parameter, so "phi fixed" cannot be expressed -- both e1 and e2 stay free.
The e1e2 run therefore carries one extra dof and its converted phi posterior will
be broad by construction. The q comparison is the meaningful one; phi is reported
for completeness, not as a like-for-like test.

  A: T_star, dL fixed; lens0_gamma free
  B: lens0_gamma fixed; T_star, dL free

    python qphi_gwonly_compare.py            # both
    python qphi_gwonly_compare.py GW-only-A  # one
"""

import json
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from herculens.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import run_inference

from qphi_setup import (
    GW_ONLY_ALWAYS_FIXED,
    build_ctx,
    priors_for,
    source_plane_bounds,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_gwonly")
os.makedirs(OUT_DIR, exist_ok=True)

METHOD = "deriv-approx-source"
BUDGET = {"num_chains": 4, "num_warmup": 500, "num_samples": 1000,
          "max_tree_depth": 10}


def priors_e1e2(ctx, variant):
    """Same fixed set as the q_phi run, with lens0_e2 standing in for lens0_phi.

    lens0_phi does not exist in e1e2 mode, so the q_phi run's "phi fixed" has no
    direct translation. Fixing lens0_e2 instead keeps the free-parameter count
    equal (4 vs 4), which is what makes the two runs comparable at all.

    It is NOT the same model: with phi pinned, e1 and e2 both vary with q at fixed
    ratio e2/e1 = tan(2*phi); with e2 pinned, e1 roams independently. Equal dof,
    different one-parameter family."""
    truth = ctx["truth_params"]
    keys = [k for k in GW_ONLY_ALWAYS_FIXED if k != "lens0_phi"] + ["lens0_e2"]
    keys += ["T_star", "dL"] if variant == "GW-only-A" else ["lens0_gamma"]
    b = source_plane_bounds(ctx)
    import numpyro.distributions as dist
    return {**{k: float(truth[k]) for k in keys},
            "y0gw": dist.Uniform(*b["y0gw"]), "y1gw": dist.Uniform(*b["y1gw"])}


def run(variant, param):
    print(f"\n{'=' * 72}\n{variant} / {METHOD} / {param}\n{'=' * 72}", flush=True)
    ctx = build_ctx(param)
    ctx["cfg"]["inference"].update(BUDGET)
    priors = priors_for(ctx, variant) if param == "q_phi" else priors_e1e2(ctx, variant)
    samples, _ = run_inference(
        ctx, mode="GW-only", method=METHOD,
        cfg={"priors": priors,
             "inference": {**BUDGET, "informed": True},
             "gw": {"source_plane_bounds": source_plane_bounds(ctx)},
             "output": {"output_dir": OUT_DIR, "json_tag": f"{variant}_{param}"}},
    )
    samples = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(os.path.join(OUT_DIR, f"{variant}_{param}.npz"), **samples)
    truth = {k: float(v) for k, v in ctx["truth_params"].items() if np.ndim(v) == 0}
    json.dump(truth, open(os.path.join(OUT_DIR, "truth.json"), "w"), indent=2)
    print(f"  saved {variant}_{param}.npz  free params: {sorted(samples)}")
    return samples, truth


def compare(variant, qp, ee, truth):
    # e1e2 -> q/phi. phi_truth is what the q_phi run held fixed.
    phi_c, q_c = ellipticity2phi_q(ee["lens0_e1"], ee["lens0_e2"])
    ee = dict(ee)
    ee["lens0_phi"], ee["lens0_q"] = np.asarray(phi_c), np.asarray(q_c)
    qp = dict(qp)
    if "lens0_e1" not in qp:
        # q_phi run with phi fixed produces no e1/e2 columns (the backfill needs
        # both q and phi in samples) -- derive them here using the fixed phi.
        e1, e2 = phi_q2_ellipticity(np.full_like(qp["lens0_q"], truth["lens0_phi"]),
                                    qp["lens0_q"])
        qp["lens0_e1"], qp["lens0_e2"] = np.asarray(e1), np.asarray(e2)
        qp["lens0_phi"] = np.full_like(qp["lens0_q"], truth["lens0_phi"])

    # Tolerance, not "> 0": a column filled from a fixed truth carries ~1e-16 of
    # float noise, which passes a bare > 0 test and then trips corner's
    # no-dynamic-range check.
    varies = lambda v: float(np.std(np.asarray(v, float))) > 1e-12
    shared = [k for k in qp if k in ee and varies(ee[k])]
    order = ["lens0_q", "lens0_phi", "lens0_e1", "lens0_e2", "lens0_gamma",
             "T_star", "dL", "y0gw", "y1gw"]
    plot = [k for k in order if k in shared and varies(qp[k])]

    fig = corner.corner(np.column_stack([qp[k] for k in plot]), labels=plot,
                        color="C0", truths=[truth.get(k) for k in plot],
                        truth_color="k", plot_datapoints=False,
                        levels=(0.68, 0.95), hist_kwargs={"density": True})
    corner.corner(np.column_stack([ee[k] for k in plot]), fig=fig, color="C3",
                  plot_datapoints=False, levels=(0.68, 0.95),
                  hist_kwargs={"density": True})
    fig.legend(handles=[plt.Line2D([], [], color="C0", label="q_phi (direct)"),
                        plt.Line2D([], [], color="C3", label="e1e2 -> q/phi")],
               loc="upper right", frameon=False)
    path = os.path.join(OUT_DIR, f"{variant}_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'=' * 92}\n{variant}: direct q/phi vs e1e2-converted\n{'=' * 92}")
    print(f"  {'param':<16} {'truth':>12} {'direct q/phi':>23} {'e1e2 -> q/phi':>23} {'shift':>8}")
    for k in [x for x in order if x in shared]:
        a, b = np.asarray(qp[k], float), np.asarray(ee[k], float)
        sa = np.std(a)
        sh = abs(np.median(a) - np.median(b)) / sa if sa > 1e-12 else float("nan")
        t = truth.get(k)
        note = "  <- fixed in q_phi" if sa <= 1e-12 else ""
        print(f"  {k:<16} {('-' if t is None else f'{t:12.5g}')} "
              f"{np.median(a):>12.5g} +-{sa:<9.4g} "
              f"{np.median(b):>12.5g} +-{np.std(b):<9.4g} {sh:>7.2f}s{note}")
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    from_saved = "--from-saved" in argv
    if from_saved:
        argv.remove("--from-saved")
    for variant in (argv or ["GW-only-A", "GW-only-B"]):
        if from_saved:
            qp = dict(np.load(os.path.join(OUT_DIR, f"{variant}_q_phi.npz")))
            ee = dict(np.load(os.path.join(OUT_DIR, f"{variant}_e1e2.npz")))
            truth = json.load(open(os.path.join(OUT_DIR, "truth.json")))
        else:
            qp, truth = run(variant, "q_phi")
            ee, _ = run(variant, "e1e2")
        compare(variant, qp, ee, truth)
