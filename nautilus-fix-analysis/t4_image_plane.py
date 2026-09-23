"""T4 -- the gap: nautilus-image was never in the prototype.

The prototype only covers the source-plane builders plus the shared EM-only one.
nautilus-image builds its likelihood from a numpyro probmodel
(nautilus_common.probmodel_log_likelihood), which is a different animal. Measure
what it costs per call, how many programs it recompiles, and whether its closure
can cross a process boundary -- i.e. whether cfg["nautilus"]["jit"]/["pool"] can
mean anything at all for this method.
"""

import pickle

import numpy as np
from common import compiles_per_call, ms_per_call, save_json

from gwemfish.nautilus_image_inference import build_image_plane_problem
from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"
N_DRAWS = 20

ctx = build_ctx(MODE)
sub = cfg_for_run(ctx, MODE)

prior, loglike, names = build_image_plane_problem(ctx, MODE, sub)
print(f"\nparams ({prior.dimensionality()}): {names}")

rng = np.random.default_rng(0)
points = [prior.unit_to_dictionary(u)
          for u in rng.uniform(size=(N_DRAWS, prior.dimensionality()))]
vals = np.array([loglike(p) for p in points])
finite = np.isfinite(vals) & (vals > -1e299)
pts = [p for p, f in zip(points, finite) if f][:10] or points[:10]

c = compiles_per_call(loglike, pts)
t = ms_per_call(loglike, pts)
print(f"accepted {int(finite.sum())}/{N_DRAWS}")
print(f"compiles/call {c:.1f}")
print(f"ms/call       {t:.2f}")

try:
    pickle.dumps(loglike)
    pickle_ok, pickle_err = True, None
except Exception as exc:
    pickle_ok, pickle_err = False, f"{type(exc).__name__}: {exc}"
print(f"\nlikelihood picklable: {pickle_ok}  {pickle_err or ''}")

try:
    pickle.dumps(prior)
    prior_ok, prior_err = True, None
except Exception as exc:
    prior_ok, prior_err = False, f"{type(exc).__name__}: {exc}"
print(f"prior picklable:      {prior_ok}  {prior_err or ''}")

save_json("t4_image_plane.json", {
    "mode": MODE, "n_params": prior.dimensionality(), "param_names": names,
    "compiles_per_call": c, "ms_per_call": t,
    "likelihood_picklable": pickle_ok, "likelihood_pickle_error": pickle_err,
    "prior_picklable": prior_ok, "prior_pickle_error": prior_err,
})
