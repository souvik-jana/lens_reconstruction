"""Is sampling (q, phi) directly consistent with sampling e1/e2 and converting?

Nothing in the repo compares the two parametrizations on the same system. This
runs EM-only fisher twice -- once with lens_mass_parametrization="q_phi", once
with "e1e2" -- on the same lens, same seed, and overlays them in (q, phi) space,
converting the e1e2 run via herculens' ellipticity2phi_q.

They should agree roughly, NOT exactly: q/phi carries Uniform(0.01,1) x
Uniform(-pi/2,pi/2) while e1e2 carries TruncatedNormal(0, 0.3, -1, 1) per
component. Those are different priors, not a reweighting of one, so a
prior-sized gap is the correct answer. A large gap would mean a real bug.

    python qphi_consistency.py
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
from herculens.Util.param_util import ellipticity2phi_q

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import run_inference

from qphi_setup import build_ctx, priors_for

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_coverage")
os.makedirs(OUT_DIR, exist_ok=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "EM-only"
METHOD = "fisher" if MODE == "EM-only" else "fisher-source"

runs = {}
for param in ("q_phi", "e1e2"):
    print(f"\n--- {MODE} {METHOD}, lens_mass_parametrization={param!r} ---", flush=True)
    ctx = build_ctx(param)
    cfg = {"priors": priors_for(ctx, MODE),
           "output": {"output_dir": OUT_DIR,
                      "json_tag": f"consistency_{MODE}_{param}".replace("+", "_")}}
    if MODE != "EM-only":
        from qphi_setup import source_plane_bounds
        cfg["gw"] = {"source_plane_bounds": source_plane_bounds(ctx)}
    s, truths = run_inference(ctx, mode=MODE, method=METHOD, cfg=cfg)
    runs[param] = {k: np.asarray(v) for k, v in s.items()}

truth = json.load(open(os.path.join(OUT_DIR, "truth.json")))
q_t, phi_t = truth["lens0_q"], truth["lens0_phi"]

# q/phi sampled directly
direct = np.column_stack([runs["q_phi"]["lens0_q"], runs["q_phi"]["lens0_phi"]])

# q/phi derived from the e1e2 run
phi_c, q_c = ellipticity2phi_q(runs["e1e2"]["lens0_e1"], runs["e1e2"]["lens0_e2"])
converted = np.column_stack([np.asarray(q_c), np.asarray(phi_c)])

labels = ["lens0_q", "lens0_phi"]
fig = corner.corner(direct, labels=labels, color="C0", truths=[q_t, phi_t],
                    truth_color="k", plot_datapoints=False, levels=(0.68, 0.95),
                    hist_kwargs={"density": True})
corner.corner(converted, fig=fig, labels=labels, color="C3",
              plot_datapoints=False, levels=(0.68, 0.95),
              hist_kwargs={"density": True})
fig.legend(handles=[plt.Line2D([], [], color="C0", label="sampled q/phi directly"),
                    plt.Line2D([], [], color="C3", label="e1e2 run, converted to q/phi")],
           loc="upper right", frameon=False)
path = os.path.join(OUT_DIR, f"qphi_consistency_{MODE}.png".replace("+", "_"))
fig.savefig(path, dpi=150, bbox_inches="tight")

print(f"\n{'=' * 78}\nCONSISTENCY [{MODE} / {METHOD}]: direct q/phi vs e1e2-converted\n{'=' * 78}")
print(f"  {'param':<12} {'truth':>10} {'direct':>22} {'converted':>22} {'shift':>9}")
for i, name in enumerate(labels):
    t = (q_t, phi_t)[i]
    d, c = direct[:, i], converted[:, i]
    shift = abs(np.median(d) - np.median(c)) / np.std(d)
    print(f"  {name:<12} {t:>10.5f} "
          f"{np.median(d):>12.5f} +- {np.std(d):<7.5f} "
          f"{np.median(c):>12.5f} +- {np.std(c):<7.5f} {shift:>8.2f}s")
print("\n  Shift is in units of the direct run's sigma. A prior-sized gap is")
print("  expected -- the two parametrizations carry different priors.")
print(f"\nwrote {path}")
