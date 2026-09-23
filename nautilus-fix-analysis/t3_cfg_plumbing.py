"""T3 -- what happens today if cfg["nautilus"] gains "jit" / "pool" keys.

_finish_nautilus_run (simple_pipeline.py:1793-1801) splats every cfg["nautilus"]
key that is not a run_kwarg and not in its _skip set straight into run_nautilus.
A build-time key like "jit" has no home there. This records the exact failure so
the merge plan names the right lines.
"""

import inspect

from common import save_json

from gwemfish.nautilus_common import run_nautilus

sig = inspect.signature(run_nautilus)
accepted = list(sig.parameters)
print(f"run_nautilus signature: {sig}")

SKIP = {"solver_backend", "solver_validation_tol"}
RUN_KWARG_KEYS = {"n_eff", "n_like_max", "discard_exploration", "timeout"}

n_cfg = {"n_live": 200, "n_eff": 500, "n_like_max": 4000, "resume": False,
         "verbose": False, "solver_backend": "jaxtronomy",
         "jit": True, "pool": 4}

run_kwargs = {k: v for k, v in n_cfg.items() if k in RUN_KWARG_KEYS}
nautilus_top = {k: v for k, v in n_cfg.items()
                if k not in SKIP and k not in RUN_KWARG_KEYS}
print(f"\nkeys _finish_nautilus_run would splat into run_nautilus: {sorted(nautilus_top)}")

unknown = [k for k in nautilus_top if k not in accepted]
print(f"keys run_nautilus does NOT accept: {unknown}")

err = None
try:
    run_nautilus(None, None, run_kwargs=run_kwargs, **nautilus_top)
except TypeError as exc:
    err = str(exc)
print(f"\nTypeError raised: {err}")

print("\nforwarded to nautilus.Sampler today:")
src = inspect.getsource(run_nautilus)
for line in src.splitlines():
    if "nautilus.Sampler(" in line or "=" in line and "        " in line[:9]:
        pass
print("  " + ", ".join(sorted(accepted)))
print("  'pool' accepted by run_nautilus: " + str("pool" in accepted))
print("  'seed' accepted by run_nautilus: " + str("seed" in accepted))

returns_diagnostics = "log_z" in src or "n_eff" in src
print(f"  run_nautilus captures nautilus diagnostics (log_z/n_eff/n_like): {returns_diagnostics}")

save_json("t3_cfg_plumbing.json", {
    "run_nautilus_params": accepted,
    "splatted_keys": sorted(nautilus_top),
    "unaccepted_keys": unknown,
    "typeerror": err,
    "pool_forwarded": "pool" in accepted,
    "seed_forwarded": "seed" in accepted,
    "diagnostics_captured": returns_diagnostics,
})
