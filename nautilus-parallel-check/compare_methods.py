"""Nautilus (prototype) vs fisher-source vs deriv-approx-source, same system.

The three methods answer the same question with very different machinery:
nautilus samples the true likelihood, while fisher-source and
deriv-approx-source build a local approximation around the peak and sample that.
Overlaying them shows how much the approximations miss -- which is the reason to
care that nautilus is now affordable.

Priors follow the tutorial's own convention: the gradient methods use the manual
PRIORS block, and nautilus used the fisher_h0 box derived from the fisher run.
That is not an inconsistency to fix, it is what tutorial_gw_only.py does.

Reuses the ctx construction from run_tutorial_full so the simulated system is
identical to the one the nautilus posteriors came from.

    python compare_methods.py
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

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import run_inference, setup_em_observation, setup_gw_observation

from run_tutorial_full import OUT_DIR, tutorial_cfg, tutorial_priors

NAUTILUS_VARIANT = "D_jit_pool"
GRADIENT_METHODS = ["fisher-source", "deriv-approx-source"]
COLORS = {"nautilus-source": "C0", "fisher-source": "C4",
          "deriv-approx-source": "C3"}

if __name__ == "__main__":
    nautilus_path = os.path.join(OUT_DIR, f"GW-only_{NAUTILUS_VARIANT}.npz")
    if not os.path.exists(nautilus_path):
        raise SystemExit(f"missing {nautilus_path} -- run run_tutorial_full.py first")
    samples_by_method = {"nautilus-source": dict(np.load(nautilus_path))}

    CFG = tutorial_cfg()
    ctx = setup_em_observation(cfg=CFG)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth = ctx["truth_params"]
    PRIORS = tutorial_priors(truth)
    ctx["cfg"]["priors"] = dict(PRIORS)
    ctx["cfg"]["gw"]["source_plane_bounds"] = {
        "y0gw": (0.01992, 0.02005), "y1gw": (0.0091, 0.0106)}

    for method in GRADIENT_METHODS:
        print(f"\n--- GW-only: {method} ---", flush=True)
        # informed=True gives deriv-approx the Fisher Hessian as its NUTS mass
        # matrix. GW-only converged without it (r_hat 1.0001, only 4 comparable
        # parameters), but EM-only did not, so it is set here too rather than
        # left to depend on how well-scaled a given problem happens to be.
        samples, _ = run_inference(
            ctx, mode="GW-only", method=method,
            cfg={"priors": PRIORS,
                 "inference": ({"informed": True}
                               if method == "deriv-approx-source" else {}),
                 "output": {"output_dir": OUT_DIR,
                            "json_tag": method.replace("-", "_")}},
        )
        samples_by_method[method] = {k: np.asarray(v) for k, v in samples.items()}
        np.savez(os.path.join(OUT_DIR, f"GW-only_{method}.npz"),
                 **samples_by_method[method])

    truths = json.load(open(os.path.join(OUT_DIR, "truths.json")))

    # Only parameters every method actually sampled -- the gradient methods carry
    # a different set of keys, and a parameter one of them holds fixed would make
    # the overlay meaningless.
    names = list(samples_by_method)
    shared = [p for p in samples_by_method["nautilus-source"]
              if all(p in s and np.std(s[p]) > 0 for s in samples_by_method.values())]
    print(f"\nmethods: {names}")
    print(f"shared sampled parameters: {shared}")

    print(f"\n{'=' * 78}\nMARGINALS  (median [16%, 84%])\n{'=' * 78}")
    for p in shared:
        t = truths.get(p)
        print(f"\n{p}" + (f"   truth = {t:.6g}" if t is not None else ""))
        for m in names:
            v = samples_by_method[m][p]
            lo, med, hi = np.percentile(v, [16, 50, 84])
            sd = np.std(v)
            line = f"  {m:<22} {med:12.6g}  [{lo:.6g}, {hi:.6g}]  sigma={sd:.3g}"
            if t is not None:
                line += f"  bias={(med - t) / sd:+.2f}sigma"
            print(line)

    print(f"\n{'=' * 78}\nWIDTH vs NAUTILUS  (sigma_method / sigma_nautilus)\n{'=' * 78}")
    print(f"  {'param':<18}" + "".join(f"{m:>22}" for m in names[1:]))
    for p in shared:
        ref = np.std(samples_by_method["nautilus-source"][p])
        print(f"  {p:<18}" + "".join(
            f"{np.std(samples_by_method[m][p]) / ref:>21.3f}x" for m in names[1:]))

    fig = None
    for m in names:
        arr = np.column_stack([samples_by_method[m][p] for p in shared])
        fig = corner.corner(
            arr, labels=shared, color=COLORS[m], fig=fig,
            truths=[truths.get(p) for p in shared], truth_color="k",
            plot_datapoints=False, levels=(0.68, 0.95),
            hist_kwargs={"density": True}, no_fill_contours=True, fill_contours=False,
        )
    fig.legend(handles=[plt.Line2D([], [], color=COLORS[m], label=m) for m in names],
               loc="upper right", frameon=False, fontsize=11)
    path = os.path.join(OUT_DIR, "method_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {path}")
