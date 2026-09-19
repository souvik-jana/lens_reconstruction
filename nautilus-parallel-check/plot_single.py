"""Corner plot for one variant's posterior, written as soon as that variant ends.

Kept separate from the three-way comparison so a result is inspectable while the
slower variants are still running, instead of only after all of them finish.
"""

import os

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False


def sampled_params(samples):
    """Fixed parameters are delta functions; corner cannot plot them and they
    carry no information about the fit."""
    return [k for k in samples if np.std(samples[k]) > 0]


def plot_variant(name, samples, truths=None, out_dir=".", color="C0", diagnostics=None):
    params = sampled_params(samples)
    # A run that failed to converge can come back with a single sample, so every
    # parameter has zero spread and there is nothing to plot. Say so instead of
    # dying in column_stack and taking the remaining variants down with it.
    if len(params) < 2:
        return (f"(no plot: only {len(params)} varying parameter(s) in "
                f"{len(next(iter(samples.values()), []))} sample(s) -- "
                f"{name} did not converge)")
    arr = np.column_stack([samples[p] for p in params])
    truths = truths or {}

    fig = corner.corner(
        arr, labels=params, color=color,
        truths=[truths.get(p) for p in params] if truths else None,
        truth_color="k", plot_datapoints=False, levels=(0.68, 0.95),
        show_titles=True, title_fmt=".4g", title_kwargs={"fontsize": 8},
        hist_kwargs={"density": True},
    )

    title = name
    if diagnostics:
        title += (f"   wall {diagnostics['wall_seconds']:.0f}s"
                  f"   n_like {diagnostics['n_like']}"
                  f"   n_eff {diagnostics['n_eff']:.0f}"
                  f"   log_z {diagnostics['log_z']:.3f}"
                  f"   N {diagnostics['n_posterior_samples']}")
    fig.suptitle(title, fontsize=10, y=1.01)

    path = os.path.join(out_dir, f"corner_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_table(name, samples, truths=None):
    truths = truths or {}
    lines = [f"  {'param':<18} {'median':>13} {'16%':>13} {'84%':>13} {'bias':>9}"]
    for p in sampled_params(samples):
        v = samples[p]
        lo, med, hi = np.percentile(v, [16, 50, 84])
        t = truths.get(p)
        bias = f"{(med - t) / np.std(v):+.2f}s" if t is not None else "-"
        lines.append(f"  {p:<18} {med:>13.6g} {lo:>13.6g} {hi:>13.6g} {bias:>9}")
    return "\n".join(lines)
