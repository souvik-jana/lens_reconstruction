"""EM-only deriv-approx (informed): q_phi vs e1e2, and the two compared in q/phi space.

Step 1  run with lens_mass_parametrization="q_phi"   -> save posterior
Step 2  run with lens_mass_parametrization="e1e2"    -> save posterior
Step 3  convert the e1e2 posterior to (q, phi) via herculens' ellipticity2phi_q and
        overlay it on the directly-sampled q/phi posterior, groupwise.

Both runs use the same system, the same seed and the same fixed parameters
(only the shear centre lens1_ra_0/lens1_dec_0 is pinned), so the only difference
is which pair of ellipticity coordinates was sampled.

informed=True on both: cfg["inference"]["informed"] defaults to None, which
simple_pipeline.py:2229 reads as False and gives plain NUTS. On EM-only that
diverged at r_hat ~1e15 with sigmas 5.5e8x off Fisher.

    python qphi_emonly_compare.py
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

import numpy as np
from herculens.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity

from gwemfish import run_inference
from gwemfish.corner_plot_utils import (
    create_default_param_groups,
    plot_multi_comparison_corner,
)

from qphi_setup import build_ctx, priors_for

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_emonly")
os.makedirs(OUT_DIR, exist_ok=True)

MODE = "EM-only"
METHOD = "deriv-approx"
# Bigger than the coverage matrix: this one is about the posterior, not plumbing.
BUDGET = {"num_chains": 4, "num_warmup": 500, "num_samples": 1000}


def run(param):
    print(f"\n{'=' * 72}\n{MODE} / {METHOD} / lens_mass_parametrization={param!r}\n{'=' * 72}",
          flush=True)
    ctx = build_ctx(param)
    ctx["cfg"]["inference"].update(BUDGET)
    samples, truths = run_inference(
        ctx, mode=MODE, method=METHOD,
        cfg={"priors": priors_for(ctx, MODE),
             "inference": {**BUDGET, "informed": True},
             "output": {"output_dir": OUT_DIR, "json_tag": f"emonly_{param}"}},
    )
    samples = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(os.path.join(OUT_DIR, f"EM-only_deriv-approx_{param}.npz"), **samples)
    truth = {k: float(v) for k, v in ctx["truth_params"].items() if np.ndim(v) == 0}
    json.dump(truth, open(os.path.join(OUT_DIR, "truth.json"), "w"), indent=2)
    print(f"  saved EM-only_deriv-approx_{param}.npz  ({len(samples)} params)")
    return samples, truth


if __name__ == "__main__":
    qp, truth = run("q_phi")
    ee, _ = run("e1e2")

    # Put both on the same footing: q/phi for the e1e2 run, e1/e2 for the q_phi run.
    # ellipticity2phi_q is the exact inverse of the map the reparametrization uses,
    # so this conversion introduces nothing of its own.
    phi_c, q_c = ellipticity2phi_q(ee["lens0_e1"], ee["lens0_e2"])
    ee["lens0_phi"], ee["lens0_q"] = np.asarray(phi_c), np.asarray(q_c)
    if "lens0_e1" not in qp:
        e1, e2 = phi_q2_ellipticity(qp["lens0_phi"], qp["lens0_q"])
        qp["lens0_e1"], qp["lens0_e2"] = np.asarray(e1), np.asarray(e2)

    shared = [k for k in qp if k in ee and np.std(qp[k]) > 0 and np.std(ee[k]) > 0]
    groups = create_default_param_groups({k: qp[k] for k in shared})
    labels = ["q_phi (sampled directly)", "e1e2 (converted to q/phi)"]

    print(f"\nshared parameters: {len(shared)}")
    for g, params in groups.items():
        print(f"  {g:<20} {params}")

    plot_multi_comparison_corner(
        samples_dicts=[{k: qp[k] for k in shared}, {k: ee[k] for k in shared}],
        param_groups=groups,
        labels=labels,
        colors=["C0", "C3"],
        # keyed by GROUP name -- corner_plot_utils.py:727 looks up by group, not label
        truths_dict={g: {p: truth[p] for p in ps if p in truth}
                     for g, ps in groups.items()},
        truth_color="k",
        save_path=os.path.join(OUT_DIR, "emonly_{group_name}.png"),
        plot_datapoints=False,
        levels=(0.68, 0.95),
        hist_kwargs={"density": True},   # densities, so different N do not skew heights
    )

    print(f"\n{'=' * 96}\nEM-only deriv-approx: direct q/phi vs e1e2-converted\n{'=' * 96}")
    print(f"  {'param':<22} {'truth':>11} {'direct q/phi':>24} {'e1e2 -> q/phi':>24} {'shift':>8}")
    worst = 0.0
    for p in sorted(shared):
        a, b = qp[p], ee[p]
        sa = np.std(a)
        shift = abs(np.median(a) - np.median(b)) / sa if sa > 0 else np.nan
        worst = max(worst, shift if np.isfinite(shift) else 0)
        t = truth.get(p)
        print(f"  {p:<22} {('-' if t is None else f'{t:11.5g}')} "
              f"{np.median(a):>13.5g} +-{sa:<9.4g} "
              f"{np.median(b):>13.5g} +-{np.std(b):<9.4g} {shift:>7.2f}s")
    print(f"\n  worst shift: {worst:.2f} sigma")
    print(f"\nwrote {OUT_DIR}/emonly_<group>.png")
