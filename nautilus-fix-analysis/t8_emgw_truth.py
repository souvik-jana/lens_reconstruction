"""Verification 5 -- the EM+GW source-centre fix.

Before the fix, the builder solved the lens equation at the EM source centre and
compared against time delays generated at the GW source, so truth was not the
maximum: logL(truth) = -472283 while random prior draws scored 1.6e5 log-units
better, and nautilus burned its whole call budget for n_eff = 1. Three checks:
truth is finite and beats random draws, the parameter set now contains y0gw/y1gw,
and it matches what fisher-source counts as free.
"""

import numpy as np
from common import save_json

from gwemfish import run_inference
from gwemfish.nautilus_common import build_nautilus_problem
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "EM+GW"
N_DRAWS = 200

ctx = build_ctx(MODE)
sub = cfg_for_run(ctx, MODE)
prior, loglike, names = build_nautilus_problem(ctx, sub, MODE, "nautilus-source")

truth = dict(ctx["truth_params"])
src = ctx["cfg"]["gw"]["source_pos"]
truth.setdefault("y0gw", float(src[0]))
truth.setdefault("y1gw", float(src[1]))
logl_truth = loglike({k: truth[k] for k in names if k in truth})

rng = np.random.default_rng(0)
draws = np.array([loglike(prior.unit_to_dictionary(u))
                  for u in rng.uniform(size=(N_DRAWS, prior.dimensionality()))])
best_draw = float(np.max(draws))

print(f"nautilus-source EM+GW: {len(names)} sampled params")
print(f"  y0gw/y1gw sampled: {'y0gw' in names and 'y1gw' in names}")
print(f"  EM source centre also sampled: "
      f"{'source0_center_x' in names and 'source0_center_y' in names}")
print(f"  logL(truth)      = {logl_truth:.3f}")
print(f"  best of {N_DRAWS} draws = {best_draw:.3f}")
print(f"  truth is the maximum: {logl_truth > best_draw}")

run_inference(ctx, mode=MODE, method="fisher-source",
              cfg={**sub, "inference": {"n_fisher_samples": 100, "rng_key": 123},
                   "output": {"output_dir": "outputs/t8_fisher"}})
fisher_keys = list(ctx["likelihood"]["keys_to_include"])
print(f"\nfisher-source free params: {len(fisher_keys)}")
missing = sorted(set(fisher_keys) - set(names))
extra = sorted(set(names) - set(fisher_keys))
print(f"  in fisher but not nautilus: {missing}")
print(f"  in nautilus but not fisher: {extra}")

ok = (np.isfinite(logl_truth) and logl_truth > best_draw
      and "y0gw" in names and "y1gw" in names and not missing)
print(f"\nt8 {'PASS' if ok else 'FAIL'}")
save_json("t8_emgw_truth.json", {
    "n_params_nautilus": len(names), "param_names": names,
    "n_params_fisher": len(fisher_keys), "fisher_keys": fisher_keys,
    "logl_truth": float(logl_truth), "best_random_draw": best_draw,
    "missing_vs_fisher": missing, "extra_vs_fisher": extra, "pass": bool(ok),
})
