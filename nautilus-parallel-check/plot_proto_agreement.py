"""Overlay the baseline and prototype posteriors saved by bench_proto.py.

Timing means nothing unless the posterior is the same one. This plots variant A
(today's code) against variant D (jit + pool) per mode and prints the shift in
each marginal in units of the baseline's own 1-sigma width -- the number that
says whether a difference matters or is just Monte Carlo noise.

    python plot_proto_agreement.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "proto")
COMPARE = ("A_baseline", "D_jit_pool")


def load(mode, variant):
    path = os.path.join(OUT_DIR, f"{mode.replace('+', '_')}_{variant}.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def plot_mode(mode):
    a, d = (load(mode, v) for v in COMPARE)
    if a is None or d is None:
        print(f"{mode}: missing npz, skipping")
        return

    names = [k for k in a if np.std(a[k]) > 0]
    arr_a = np.column_stack([a[k] for k in names])
    arr_d = np.column_stack([d[k] for k in names])

    fig = corner.corner(arr_a, labels=names, color="C0", hist_kwargs={"density": True},
                        plot_datapoints=False, levels=(0.68, 0.95))
    corner.corner(arr_d, fig=fig, color="C1", hist_kwargs={"density": True},
                  plot_datapoints=False, levels=(0.68, 0.95))
    fig.legend(handles=[plt.Line2D([], [], color="C0", label="A baseline"),
                        plt.Line2D([], [], color="C1", label="D jit+pool")],
               loc="upper right", frameon=False)
    path = os.path.join(OUT_DIR, f"agreement_{mode.replace('+', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{mode}  ->  {os.path.basename(path)}")
    print(f"  {'param':<20} {'median shift':>14}")
    for k in names:
        sigma = np.std(a[k])
        shift = abs(np.median(d[k]) - np.median(a[k])) / sigma
        print(f"  {k:<20} {shift:>13.3f}s")


timings_path = os.path.join(OUT_DIR, "timings.json")
timings = json.load(open(timings_path)) if os.path.exists(timings_path) else {}

modes = sorted({os.path.basename(p).rsplit("_A_baseline", 1)[0].replace("_", "+")
                for p in glob.glob(os.path.join(OUT_DIR, "*_A_baseline.npz"))})
for m in modes:
    plot_mode("GW-only" if m == "GW+only" else m)

if timings:
    print(f"\n{'mode':<10} {'variant':<12} {'wall':>9} {'speedup':>9} {'log_z':>11}")
    for mode, variants in timings.items():
        base = variants["A_baseline"]["wall_seconds"]
        for name, d in variants.items():
            print(f"{mode:<10} {name:<12} {d['wall_seconds']:>8.1f}s "
                  f"{base / d['wall_seconds']:>8.2f}x {d['log_z']:>11.4f}")
