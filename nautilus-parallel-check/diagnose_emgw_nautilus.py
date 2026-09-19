"""Why does EM+GW nautilus fail to converge when GW-only and EM-only succeed?

Measured failure: 500,000 likelihood calls (the ceiling), n_eff = 1.0, one
posterior sample, log_z = -275,684. Compare GW-only (log_z -62.8, n_eff 15,459)
and EM-only (log_z +966.8, n_eff 36,436), both converged on the same machinery.

So the sampler is fine and the prototype is fine -- something about this
likelihood surface is different. Three candidate causes, and this measures each
directly on draws from the *same* fisher_h0 prior box nautilus was given:

  1. Solver rejection. Points where the lens equation does not yield exactly
     n_images distinct images return -1e300. If most of the prior box rejects,
     nautilus has almost no volume to build a bound from and stalls.
  2. Dynamic range. sigma_td = 0.001 makes the time-delay term extremely sharp.
     If log-likelihood swings by ~1e5 across the box, nested sampling's shells
     collapse onto a single point -- exactly what n_eff = 1 looks like.
  3. EM vs GW imbalance. If one term dominates the other by orders of magnitude,
     the combined surface is effectively controlled by one of them.

    python diagnose_emgw_nautilus.py
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from gwemfish import run_inference

from emonly_setup import SIGMA_SPAN, fisher_h0_priors
from proto.jit_likelihood import build_problem
from run_emgw_full import FIXED_TO_TRUTH, MODE, OUT_DIR
from tutorial_cfg import build_ctx, source_bounds

import numpyro.distributions as dist

N_DRAWS = 2000
REJECT = -1e300

ctx = build_ctx(MODE)
truth = ctx["truth_params"]
bounds = source_bounds(ctx, MODE)
PRIORS = {
    **{k: float(truth[k]) for k in FIXED_TO_TRUTH},
    "y0gw": dist.Uniform(*bounds["y0gw"]),
    "y1gw": dist.Uniform(*bounds["y1gw"]),
}
ctx["cfg"]["priors"] = dict(PRIORS)
ctx["cfg"]["gw"]["source_plane_bounds"] = bounds

print("running fisher-source to rebuild the same fisher_h0 box...", flush=True)
run_inference(ctx, mode=MODE, method="fisher-source",
              cfg={"priors": PRIORS,
                   "output": {"output_dir": OUT_DIR, "json_tag": "diag_fisher"}})
PRIORS, _ = fisher_h0_priors(ctx, PRIORS, SIGMA_SPAN)

prior, loglike, names = build_problem(ctx, {"priors": PRIORS}, MODE, jit=True)
n_dim = prior.dimensionality()
print(f"  {n_dim} sampled parameters")

rng = np.random.default_rng(0)
points = [prior.unit_to_dictionary(u) for u in rng.uniform(size=(N_DRAWS, n_dim))]

vals = np.array([loglike(p) for p in points])
rejected = vals <= REJECT / 2
good = vals[~rejected]

print(f"\n{'=' * 78}\n1. SOLVER REJECTION\n{'=' * 78}")
print(f"  draws returning -1e300 (wrong image count): "
      f"{rejected.sum()}/{N_DRAWS} = {100 * rejected.mean():.1f}%")
print(f"  usable draws: {len(good)}")

print(f"\n{'=' * 78}\n2. DYNAMIC RANGE over the prior box\n{'=' * 78}")
if len(good):
    print(f"  logL min    : {good.min():.4g}")
    print(f"  logL median : {np.median(good):.4g}")
    print(f"  logL max    : {good.max():.4g}")
    print(f"  spread (max-min): {good.max() - good.min():.4g}")

# The likelihood at truth is the target nautilus has to find.
truth_point = {k: float(truth[k]) for k in names if k in truth}
missing = [k for k in names if k not in truth_point]
if not missing:
    lv = loglike(truth_point)
    print(f"\n  logL at truth: {lv:.4g}")
    if len(good):
        print(f"  gap, truth - best random draw: {lv - good.max():.4g}")
        print("  (a gap of thousands means the peak occupies a vanishing "
              "fraction of the box;\n   nested sampling must climb that with "
              "shells and n_eff collapses on the way)")
else:
    print(f"\n  cannot evaluate truth: missing {missing}")

# Split the two terms. build_problem gives the sum, so rebuild each mode's
# likelihood on the same points to attribute the dynamic range.
print(f"\n{'=' * 78}\n3. EM vs GW CONTRIBUTION\n{'=' * 78}")
_, em_loglike, em_names = build_problem(ctx, {"priors": PRIORS}, "EM-only", jit=True)
em_vals = []
for p in points[:500]:
    sub = {k: v for k, v in p.items() if k in em_names}
    em_vals.append(em_loglike(sub) if len(sub) == len(em_names) else np.nan)
em_vals = np.array(em_vals, dtype=float)
em_ok = np.isfinite(em_vals)
sub_total = vals[:500][em_ok]
sub_em = em_vals[em_ok]
sub_gw = sub_total - sub_em
usable = sub_total > REJECT / 2
if usable.sum():
    print(f"  on {usable.sum()} usable draws:")
    print(f"    EM term  : median {np.median(sub_em[usable]):.4g}  "
          f"spread {sub_em[usable].max() - sub_em[usable].min():.4g}")
    print(f"    GW term  : median {np.median(sub_gw[usable]):.4g}  "
          f"spread {sub_gw[usable].max() - sub_gw[usable].min():.4g}")
    print("\n  The term with the larger spread is the one shaping the surface "
          "nautilus\n  must navigate.")
print(f"\n  sigma_td = {ctx['cfg']['gw']['error_scales']['sigma_td']}, "
      f"sigma_dL_eff = {ctx['cfg']['gw']['error_scales']['sigma_dL_eff']}")
