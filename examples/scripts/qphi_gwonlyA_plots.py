"""GW-only-A: plot the q_phi and e1e2 posteriors separately, as they come out.

No conversion, no overlay, no injected columns -- each corner shows exactly the
free parameters that method sampled.

  q_phi : lens0_q, lens0_gamma, y0gw, y1gw          (lens0_phi fixed to truth)
  e1e2  : lens0_e1, lens0_e2, lens0_gamma, y0gw, y1gw

The e1e2 run carries one extra free parameter: "phi fixed" has no e1/e2
equivalent, so both ellipticity components stay free.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_gwonly")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "GW-only-A"

truth = json.load(open(os.path.join(OUT_DIR, "truth.json")))
# y0gw/y1gw are not in truth_params; they are the simulated GW source position.
truth.setdefault("y0gw", 0.02)
truth.setdefault("y1gw", 0.01)

for param, color in (("q_phi", "C0"), ("e1e2", "C3")):
    s = dict(np.load(os.path.join(OUT_DIR, f"{VARIANT}_{param}.npz")))
    keys = sorted(s)
    arr = np.column_stack([np.asarray(s[k], float) for k in keys])

    fig = corner.corner(
        arr, labels=keys, color=color,
        truths=[truth.get(k) for k in keys], truth_color="k",
        plot_datapoints=False, levels=(0.68, 0.95), show_titles=True,
        title_fmt=".4g", title_kwargs={"fontsize": 9},
        hist_kwargs={"density": True},
    )
    fig.suptitle(f"{VARIANT}  |  lens_mass_parametrization = {param}  "
                 f"|  {len(keys)} free params", fontsize=11, y=1.01)
    path = os.path.join(OUT_DIR, f"{VARIANT}_{param}_posterior.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{VARIANT} / {param}  ({len(keys)} free)")
    print(f"  {'param':<14} {'truth':>12} {'median':>13} {'sigma':>12} {'bias':>8}")
    for k in keys:
        v = np.asarray(s[k], float)
        t = truth.get(k)
        sd = np.std(v)
        bias = f"{(np.median(v) - t) / sd:+.2f}s" if (t is not None and sd > 0) else "-"
        print(f"  {k:<14} {('-' if t is None else f'{t:12.5g}')} "
              f"{np.median(v):>13.5g} {sd:>12.4g} {bias:>8}")
    print(f"  -> {path}")
