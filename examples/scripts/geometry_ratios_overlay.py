"""Overlay geometry / absolute marginals across ratio and absolute runs.

Per-method switches below. If a switch is False, that method is skipped. If True
but the samples file is missing, that method is skipped with a warning.

When only absolute methods are selected (lenstronomy absolute + gwemfish
absolute), overlay uses keys q/gamma/T_star/dL/y0gw/y1gw. Mixed selections
intersect to shared keys (geometry-only when ratios are included).

    uv run python examples/scripts/geometry_ratios_overlay.py
"""

import json
import os

import matplotlib
from numpy._core.numeric import False_

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from gwemfish.corner_plot_utils import plot_multi_comparison_corner

if "science" in plt.style.available:
    plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RATIO_ROOT = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_lenstronomy"
)
LS_ABS_ROOT = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_lenstronomy_absolute"
)
GW_ROOT = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_gwemfish_absolute"
)
OUT_DIR = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_overlay"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== SWITCHES =====================
# Ratio lenstronomy (geometry-only likelihood) — each under <RATIO_ROOT>/<sampler>/
RUN_LENSTRONOMY_EMCEE = False
RUN_LENSTRONOMY_NAUTILUS = True
RUN_LENSTRONOMY_NESSAI = False

# Absolute lenstronomy (TD + dL_eff, free T_star/dL)
RUN_LENSTRONOMY_ABSOLUTE_EMCEE = False
RUN_LENSTRONOMY_ABSOLUTE_NAUTILUS = True
RUN_LENSTRONOMY_ABSOLUTE_NESSAI = False

# gwemfish absolute TD + dL_eff — each under <GW_ROOT>/<method>/
RUN_GWEMFISH_DERIV_APPROX = False
RUN_GWEMFISH_NAUTILUS = True

# Corner axis limits: None -> corner autoscales to samples;
# "absolute_prior" -> lenstronomy absolute BOUNDS (shared priors).
AXIS_LIMITS = "absolute_prior"  # None | "absolute_prior"

# Match geometry_ratios_lenstronomy_absolute.BOUNDS
ABSOLUTE_PRIOR_BOUNDS = {
    "q": (0.3, 0.75),
    "gamma": (1.2, 2.8),
    "T_star": (4e6, 1.6e7),
    "dL": (10000.0, 28000.0),
    "y0gw": (-0.001, 0.05),
    "y1gw": (-0.008, 0.028),
}

GEOMETRY_KEYS = ["q", "gamma", "y0gw", "y1gw"]
ABSOLUTE_KEYS = ["q", "gamma", "T_star", "dL", "y0gw", "y1gw"]

GW_RENAME_ABS = {
    "lens0_q": "q",
    "lens0_gamma": "gamma",
    "T_star": "T_star",
    "dL": "dL",
    "y0gw": "y0gw",
    "y1gw": "y1gw",
}

# ===================== METHOD TABLE =====================
# (switch, label, path, rename_or_None, color, kind)
# kind: "geometry" | "absolute"
# rename=None means samples already use destination keys.
METHODS = [
    (
        RUN_LENSTRONOMY_EMCEE,
        "lenstronomy_emcee",
        os.path.join(RATIO_ROOT, "emcee", "samples_emcee_full.npz"),
        None,
        "C0",
        "geometry",
    ),
    (
        RUN_LENSTRONOMY_NAUTILUS,
        "lenstronomy_nautilus",
        os.path.join(RATIO_ROOT, "nautilus", "samples_nautilus_full.npz"),
        None,
        "C1",
        "geometry",
    ),
    (
        RUN_LENSTRONOMY_NESSAI,
        "lenstronomy_nessai",
        os.path.join(RATIO_ROOT, "nessai", "samples_nessai_full.npz"),
        None,
        "C2",
        "geometry",
    ),
    (
        RUN_LENSTRONOMY_ABSOLUTE_EMCEE,
        "lenstronomy_absolute_emcee",
        os.path.join(LS_ABS_ROOT, "emcee", "samples_emcee_full.npz"),
        None,
        "C5",
        "absolute",
    ),
    (
        RUN_LENSTRONOMY_ABSOLUTE_NAUTILUS,
        "lenstronomy_absolute_nautilus",
        os.path.join(LS_ABS_ROOT, "nautilus", "samples_nautilus_full.npz"),
        None,
        "C6",
        "absolute",
    ),
    (
        RUN_LENSTRONOMY_ABSOLUTE_NESSAI,
        "lenstronomy_absolute_nessai",
        os.path.join(LS_ABS_ROOT, "nessai", "samples_nessai_full.npz"),
        None,
        "C7",
        "absolute",
    ),
    (
        RUN_GWEMFISH_DERIV_APPROX,
        "gwemfish_deriv_approx",
        os.path.join(GW_ROOT, "deriv_approx_source", "samples.npz"),
        GW_RENAME_ABS,
        "C3",
        "absolute",
    ),
    (
        RUN_GWEMFISH_NAUTILUS,
        "gwemfish_nautilus",
        os.path.join(GW_ROOT, "nautilus_source", "samples.npz"),
        GW_RENAME_ABS,
        "C4",
        "absolute",
    ),
]

TRUTHS = {
    "q": 0.5,
    "gamma": 2.0,
    "T_star": 7472713.403790,
    "dL": 15946.706162,
    "y0gw": 0.02,
    "y1gw": 0.00001,
}
CONTOUR_LEVELS = [0.95]


def load_samples(path):
    data = np.load(path)
    out = {k: np.asarray(data[k]) for k in data.files if k != "weights"}
    weights = np.asarray(data["weights"]) if "weights" in data.files else None
    return out, weights


def available_keys(samp, rename, kind):
    if rename is None:
        keys = ABSOLUTE_KEYS if kind == "absolute" else GEOMETRY_KEYS
        return [k for k in keys if k in samp]
    out = []
    for src, dst in rename.items():
        if src in samp:
            out.append(dst)
    return out


def to_selected(samp, rename, keys):
    if rename is None:
        return {k: np.asarray(samp[k]).reshape(-1) for k in keys}
    geom = {}
    inv = {dst: src for src, dst in rename.items()}
    for k in keys:
        src = inv.get(k, k)
        if src not in samp:
            raise KeyError(f"missing key {src!r} in samples; have {list(samp)}")
        geom[k] = np.asarray(samp[src]).reshape(-1)
    return geom


def equal_weight(samp, weights, rng, n=None):
    if weights is None:
        return samp
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n = len(w) if n is None else n
    idx = rng.choice(len(w), size=n, p=w)
    return {k: v[idx] for k, v in samp.items()}


def quantiles(v, weights=None):
    v = np.asarray(v).reshape(-1)
    if weights is None:
        return [float(np.quantile(v, q)) for q in (0.16, 0.5, 0.84)]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    order = np.argsort(v)
    v_s = v[order]
    cdf = np.cumsum(w[order])
    return [float(np.interp(q, cdf, v_s)) for q in (0.16, 0.5, 0.84)]


print("=" * 72)
print("  geometry_ratios_overlay")
print("=" * 72)

rng = np.random.default_rng(0)
loaded = []

for flag, label, path, rename, color, kind in METHODS:
    if not flag:
        print(f"  skip {label}: switch=False")
        continue
    if not os.path.exists(path):
        print(f"  skip {label}: no samples at {path}")
        continue
    raw, weights = load_samples(path)
    keys_here = available_keys(raw, rename, kind)
    loaded.append((label, raw, weights, rename, color, path, keys_here, kind))
    print(f"  load {label}: {path}  keys={keys_here}")

if len(loaded) < 2:
    raise SystemExit(
        f"need >= 2 methods with samples for overlay (got {len(loaded)}). "
        "Run inference scripts first, or turn switches on."
    )

# Intersect keys across selected methods; prefer absolute order when all share it.
shared = set(loaded[0][6])
for _label, _raw, _w, _ren, _c, _p, keys_here, _kind in loaded[1:]:
    shared &= set(keys_here)
if not shared:
    raise SystemExit("no shared parameter keys across selected methods")

all_absolute = all(kind == "absolute" for *_rest, kind in loaded)
preferred = ABSOLUTE_KEYS if all_absolute else GEOMETRY_KEYS
PLOT_KEYS = [k for k in preferred if k in shared]
if not PLOT_KEYS:
    PLOT_KEYS = [k for k in ABSOLUTE_KEYS + GEOMETRY_KEYS if k in shared]

print(f"  overlay keys: {PLOT_KEYS}  (all_absolute={all_absolute})")

active = []
summary = {}
for label, raw, weights, rename, color, path, _keys, _kind in loaded:
    sel = to_selected(raw, rename, PLOT_KEYS)
    eq = equal_weight(sel, weights, rng)
    active.append((label, eq, sel, weights, color, path))

print(f"\n  16/50/84% ({', '.join(PLOT_KEYS)}):")
for label, _eq, sel, weights, _color, path in active:
    summary[label] = {"path": path, "quantiles": {}}
    parts = []
    for k in PLOT_KEYS:
        q = quantiles(sel[k], weights)
        summary[label]["quantiles"][k] = {
            "q16": q[0],
            "q50": q[1],
            "q84": q[2],
        }
        parts.append(f"{k}=[{q[0]:.4g}, {q[1]:.4g}, {q[2]:.4g}]")
    print(f"    {label}: " + "  ".join(parts))

with open(os.path.join(OUT_DIR, "geometry_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

truths_plot = {k: TRUTHS[k] for k in PLOT_KEYS if k in TRUTHS}
if AXIS_LIMITS is None:
    param_ranges = None
elif AXIS_LIMITS == "absolute_prior":
    param_ranges = {k: ABSOLUTE_PRIOR_BOUNDS[k] for k in PLOT_KEYS}
else:
    raise SystemExit(
        f"AXIS_LIMITS={AXIS_LIMITS!r}; use None or 'absolute_prior'"
    )
print(f"  axis limits: {AXIS_LIMITS}  ranges={param_ranges}")

plot_multi_comparison_corner(
    [eq for _label, eq, _geom, _w, _c, _p in active],
    {"params": PLOT_KEYS},
    labels=[label for label, _eq, _geom, _w, _c, _p in active],
    colors=[color for _label, _eq, _geom, _w, color, _p in active],
    truths_dict={"params": truths_plot},
    param_ranges=param_ranges,
    hist_kwargs={"density": True},
    levels=CONTOUR_LEVELS,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=False,
    no_fill_contours=True,
    save_path=os.path.join(OUT_DIR, "overlay_geometry.png"),
)
plt.close("all")

print(f"\nDone ({len(active)} methods) -> {OUT_DIR}/overlay_geometry.png")
print(f"  keys={PLOT_KEYS}  AXIS_LIMITS={AXIS_LIMITS}")
