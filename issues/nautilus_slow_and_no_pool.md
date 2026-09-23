# Nautilus: 6 XLA recompiles per likelihood call, and `pool=N` cannot be used

**Status:** RESOLVED -- merged as `cfg["nautilus"]["jit"]` (default `True`) and
`cfg["nautilus"]["pool"]` (default `None`), plus `["seed"]`
**Where:** `src/gwemfish/nautilus_jit_cores.py` (new), `nautilus_common.py`,
`nautilus_source_inference.py`, `nautilus_image_inference.py`, `simple_pipeline.py`
**Prototype:** `nautilus-parallel-check/` (superseded by the merge)
**Measurements:** `nautilus-parallel-check/REPORT.md`, and post-merge in
`nautilus-fix-analysis/FINDINGS.md`
**Tests:** `tests/test_nautilus_jit_pool.py`

## Resolution

Both causes are fixed and both defaults are safe: `jit=True` is faster with
identical numbers, `pool=None` behaves exactly as before.

- `jit` compiles the three hot cores (`nautilus_jit_cores.py`). Post-merge, 200
  draws per mode: max relative difference 1.07e-12 (GW-only), 2.48e-13 (EM+GW),
  1.93e-15 (EM-only); compiles/call 6 -> 0; 159 -> 9.3 ms (GW-only), 168 -> 3.6 ms
  (EM+GW).
- `pool` ships the ctx and lets each worker rebuild its own closure
  (`PicklableLikelihood`), forcing `spawn` only when a pool is requested.
  Callable priors are converted to their distribution before shipping instead of
  refusing.
- `nautilus-image` gets `pool` but warns and runs eager on `jit`: its likelihood
  is rebuilt by numpyro on every call, so there is no arithmetic core to compile.
- `run_nautilus` now also captures nautilus' own `log_z` / `n_eff` / `n_like`
  into the saved run json, which it previously discarded.
- Refused rather than silently wrong: `use_mst=True` and
  `use_parameter_layout=False` raise under `jit=True`.
- `fisher-source` and `deriv-approx-source` verified bit-identical before and
  after (`nautilus-fix-analysis/baseline_methods.py`).

## Issue

`nautilus-source` costs hours where `fisher-source` costs seconds. Two independent
causes, both measured.

### 1. JAX recompiles 6 programs on every likelihood call, forever

Steady state, same input point repeated:

```text
call 1:  109.9 ms  compiles=6
call 5:  109.3 ms  compiles=6      <- never settles
```

`jax.log_compiles` names them: five `jit(scan)` over `complex128[4..5]` plus one
`jit(pure_callback)`. The `scan`s come from herculens' EPL profile
(`herculens/Util/jax_util.py:41`, `R_omega` → `lax.fori_loop`).

A `lax.scan` invoked **outside** `jax.jit` retraces and recompiles its body every
call (~13 ms each), because the compilation cache key is the body-function object,
which is fresh each time. gwemfish has no `@jax.jit` anywhere on a nautilus
likelihood path, so this fires on all ~1e5 calls of a run.

About 65 ms of each 110 ms call is this. JAX is not the problem — not using it is.
Removing JAX is not an option anyway: herculens' `LensImage`, `LensImageGW`,
`MassModel` and the solver are JAX throughout.

| | eager | jitted | ratio |
|---|---|---|---|
| EPL `potential` + `alpha` + `magnification` | 54.4 ms | 0.019 ms | ~2900x |

### 2. `pool=N` cannot be passed at all

Every builder returns a function defined *inside* another function, and pickle
cannot serialise a local closure by reference:

```text
AttributeError: Can't get local object
                'build_gw_source_plane_problem.<locals>.log_likelihood'
```

`run_nautilus` (`nautilus_common.py:357`) also never forwards `pool` to
`nautilus.Sampler`, and `cfg_reference.py` documents no parallelism key.

Two further traps found while prototyping:

- `nautilus/pool.py:3` does a bare `from multiprocessing import Pool`, i.e. the
  platform default — `fork` on Linux. Forking a process with JAX's thread pool
  running deadlocks (observed). `spawn` must be forced.
- A prior `fisher` / `deriv-approx` run leaves unpicklable jitted closures in `ctx`
  (`simple_pipeline.py:2097-2098`, `:2114`). They must be stripped or pool startup
  fails, which would break every multi-method script.

## Measured effect of fixing both

Full budget, `n_live=2000, n_eff=5000`, three modes:

| mode | jit | pool(4) | combined |
|---|---|---|---|
| GW-only | 17.2x | 2.8x | **41x** (1.9 h → 163 s) |
| EM+GW | not measured | 2.73x | — |
| EM-only | 1.01x | 1.69x | 1.7x |

EM-only gains nothing from jit, and that is expected rather than a failure: it
solves no lens equation and herculens already jits `lens_image.model`, so there is
no recompilation to remove. Over 90% of an EM-only run is nautilus' own
neural-network training, which neither fix touches.

**Correctness:** with the same seed, the eager and jitted likelihoods drove
*identical* nested sampling runs — same `n_like`, same `n_eff`, same `log_z` to
four decimals, and `np.array_equal` true on all 6648 GW-only posterior samples.
Per-call agreement over 200 random draws per mode: max relative difference 1e-12
or better.

## Proposed plan

Promote the prototype into `src/gwemfish` behind two new cfg keys, in this order.

### Step 1 — `cfg["nautilus"]["jit"]` (default `True`)

Wrap the three hot cores in `jax.jit`, cached on the identity of the heavyweight
objects they close over. The blocking change is small: internal `float()` / `int()`
casts must move to the outermost boundary, since a `float()` inside a traced region
raises `ConcretizationTypeError` — this is exactly what blocks jit today at
`nautilus_source_inference.py:73`.

| core | file |
|---|---|
| GW time-delay + dL_eff loglike | `nautilus_source_inference.py:57` |
| solve + `select_images` | `nautilus_source_inference.py:104` + `lens_setup.py:527` |
| EM pixel loglike | `nautilus_common.py:259` |

`solver.solve` is only *called*, never re-implemented — the differentiable solver,
its `custom_root` gradient rule and the Newton polish stay untouched, so fisher /
hmc / deriv-approx are unaffected.

**Guard:** refuse the jit path when `cfg["use_mst"]` is true. `flex_prob_model.py:161`
assigns `k_mst` onto `lens_image.MassModel`, which herculens passes to `jax.jit` as
`static_argnums=0`; the compiled program would then be keyed on an object whose
contents changed. Better to raise than to return a plausible wrong number.

### Step 2 — `cfg["nautilus"]["pool"]` (default `None`)

A picklable likelihood wrapper carrying the ctx (44 KB, pickles instantly) so each
worker rebuilds only its own closure. Must: strip the unpicklable `ctx["fisher"]`
and `ctx["likelihood"]` entries; force `spawn`; pin each worker to one thread via
the existing `jax_config.setup_jax(ncpus=1)`; and validate that `cfg["priors"]`
contains no lambdas (numpyro distributions pickle, callables do not) with a clear
error naming the offending keys rather than failing inside the pool.

Pass `pool` as an **int** so nautilus takes its `initializer=initialize_worker`
path (`nautilus/pool.py:59`) and pickles the likelihood once per worker instead of
once per `map` call.

### Step 3 — verification

1. Per-call: max relative |Δ logL| < 1e-10 over 200 draws per mode.
2. Compile counter: 6 per call → 0 after warm-up.
3. End-to-end: same seed, eager vs jitted must give identical `n_like`, `n_eff`,
   `log_z` and `np.array_equal` samples.
4. Re-run an existing multi-method script to confirm fisher / deriv-approx / hmc
   numbers are unchanged.

## Not covered here

`vectorized=True` is the next lever after jit — nautilus can take a batch of points
and `vmap` the likelihood, one dispatch instead of `n_batch`. It is mutually
exclusive with `pool` (`nautilus/sampler.py:864-870` bypasses the pool when
vectorized), and the solver's Python-level reject branch would need restructuring
first.

## Related

- `issues/nautilus_emgw_source_centre.md` — `nautilus-source` and the gradient
  methods fit different EM+GW models
- `issues/fisher_cov_physical_clip.md`
