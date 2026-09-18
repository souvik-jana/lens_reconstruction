# Nautilus speed-up prototype — results

Everything here was built outside `src/gwemfish/`, so the fisher / hmc /
deriv-approx paths ran byte-identical code throughout. No file in the package was
edited for this work.

## The two problems

**1. JAX recompiled 6 XLA programs on every likelihood call — forever.**
Steady state, same input point repeated:

```
call 1:  109.9 ms  compiles=6
call 5:  109.3 ms  compiles=6      <- never settles
```

`jax.log_compiles` names them: five `jit(scan)` over `complex128[4..5]` plus one
`jit(pure_callback)`. The `scan`s come from herculens' EPL profile
(`herculens/Util/jax_util.py:41`, `R_omega` → `lax.fori_loop`). A `lax.scan`
invoked *outside* `jax.jit` retraces and recompiles its body every call — roughly
13 ms each — because the cache key is the body-function object, which is fresh
each time. gwemfish has no `@jax.jit` anywhere on a nautilus likelihood path, so
this fired on every one of the ~1e5 calls a run makes.

JAX was never the problem. Not using it was. Dropping JAX is not an option
anyway — herculens' `LensImage`, `LensImageGW`, `MassModel` and the solver are
JAX throughout.

| | eager | jitted | ratio |
|---|---|---|---|
| EPL `potential` + `alpha` + `magnification` | 54.4 ms | 0.019 ms | ~2900x |

**2. `pool=N` could not be used at all.** Every builder returns a function defined
*inside* another function, and pickle cannot serialise a local closure by
reference:

```
AttributeError: Can't get local object
                'build_gw_source_plane_problem.<locals>.log_likelihood'
```

The earlier workaround had each worker re-simulate the system (7.2–8.2 s per
worker, and silently different data if any seed drifts). Unnecessary: the whole
`ctx` pickles in **44 KB, instantly**, so a worker can be handed the exact same
simulated system and rebuild only the closure.

Two further traps, both real: `nautilus/pool.py:3` does a bare
`from multiprocessing import Pool`, i.e. the platform default, which is `fork` on
Linux — and forking a process with JAX's thread pool running deadlocks (observed).
And a prior fisher/deriv-approx run leaves unpicklable jitted closures in `ctx`
(`simple_pipeline.py:2097-2098`, `:2114`), which must be stripped or pool startup
fails.

## Measured results — full budget (`n_live=2000, n_eff=5000, n_like_max=500_000`)

Variants: **B** = jit only, **C** = pool(4) only, **D** = jit + pool(4).
A (today's code, unmodified) was only run at reduced budget — at 136 ms/call a
full-budget GW-only run is ~1.9 h.

### GW-only — tutorial config, fisher_h0 priors at 3.5σ

| variant | wall | n_like | n_eff | log_z |
|---|---|---|---|---|
| C_pool | 2806.3 s | 49,200 | 15,459.0 | −62.7685 |
| B_jit | 459.1 s | 48,800 | 15,361.3 | −62.7827 |
| **D_jit_pool** | **162.7 s** | 49,200 | 15,459.0 | −62.7685 |

jit **17.2x**, pool **2.8x**, combined **41x** against the extrapolated baseline
(49,200 × 136.6 ms ≈ 1.9 h).

### EM-only

| variant | wall | n_like | n_eff | log_z |
|---|---|---|---|---|
| B_jit | 403.0 s | 140,000 | 35,881.8 | 966.7549 |
| C_pool | 242.1 s | 140,900 | 36,435.5 | 966.7534 |
| **D_jit_pool** | **239.0 s** | 140,900 | 36,435.5 | 966.7534 |

jit **1.01x**, pool **1.69x**, combined **1.7x**. This is the honest counter-case:
EM-only solves no lens equation and herculens already jits `lens_image.model`, so
there was no recompilation to remove. At ~0.1 ms/call the likelihood is ~6% of
wall time; the rest is nautilus' own 23-dimensional neural-network training, which
neither fix touches.

### EM+GW

| variant | wall | n_like | n_eff | log_z |
|---|---|---|---|---|
| B_jit | 1974.3 s | 154,700 | 36,678.8 | 939.7763 |
| **D_jit_pool** | **723.8 s** | 154,800 | 37,593.2 | 939.7792 |

pool **2.73x**. C_pool had not finished when this was written, so the jit factor
for this mode is not yet measured.

### Summary

| mode | jit | pool(4) | combined |
|---|---|---|---|
| GW-only | 17.2x | 2.8x | **41x** |
| EM+GW | not yet measured | 2.73x | — |
| EM-only | 1.01x | 1.69x | 1.7x |

The gain lands where the likelihood is expensive. Where it is cheap, Amdahl's law
applies to nautilus' own machinery instead.

## Correctness

Two independent checks, both passed.

**Per-call, 200 random prior draws per mode** (`check_proto_likelihood.py`):

| mode | max relative Δ logL | compiles/call | ms/call |
|---|---|---|---|
| GW-only | 1.07e-12 | 6 → **0** | 136.6 → 8.69 |
| EM+GW | 2.19e-13 | 6 → **0** | 155.8 → 4.19 |
| EM-only | 1.93e-15 | 0 → 0 | 0.1 → 0.09 |

Residuals are float64 reassociation noise — XLA fuses the same arithmetic in a
different order — far below anything nested sampling resolves.

**End-to-end.** With the same seed, `C_pool` (today's eager likelihood) and
`D_jit_pool` (compiled) produced *identical* runs in both GW-only and EM-only:
same `n_like`, same `n_eff` to one decimal, same `log_z` to four, same sample
count, and `np.array_equal` true on all 6,648 GW-only samples across every
parameter. The compiled likelihood reproduced a 49,200-call nested sampling run
exactly, 17x faster.

## Findings in `src/gwemfish` (not fixed here)

### 1. nautilus-source and the gradient methods fit different EM+GW models

`build_em_gw_source_plane_problem` (`nautilus_source_inference.py:442-443`) uses
the **EM source centre as the GW source position**:

```python
y0 = float(kwargs_source[0]["center_x"])
y1 = float(kwargs_source[0]["center_y"])
```

But `FlexProbModelSourcePlaneEMGW` (`flex_prob_model.py:582-609`) and the legacy
`ProbModelSourcePlane` (`prob_model.py:365-411`) sample `y0gw`/`y1gw`
**independently** of the EM source. Confirmed by parameter count in a live run:
Fisher reports 25 free parameters including both `source0_center_x/y` *and*
`y0gw/y1gw`; nautilus sampled 23.

So `nautilus-source` and `deriv-approx-source` EM+GW results are not comparable —
different dimensionality, different physics. EM and GW centroids are not generally
co-located (different emission regions; GW sky localisation is far coarser).

**Silent consequence:** `build_em_gw_source_plane_problem` never adds
`y0gw`/`y1gw` to `default_dists` (`:403-407`), and `build_nautilus_prior`
(`nautilus_common.py:173`) iterates only over `default_dists`. A
`cfg["priors"]["y0gw"]` handed to nautilus in EM+GW mode is **dropped with no
error and no warning**.

Observed effect: with the tutorial's EM source at (0.05, 0.1) and GW source at
(0.2, 0.01), nautilus burned the full 500,000-call ceiling for `n_eff = 1.0`,
one posterior sample, `log_z = −275,684`. Truth was not the maximum — random prior
draws scored 1.63e5 log-units better. Co-locating the two sources changed logL at
truth from −472,283 to +976 and gave `n_eff = 37,593` in 154,800 calls.

### 2. `deriv-approx` silently runs plain NUTS

`cfg["inference"]["informed"]` defaults to `None`, and `simple_pipeline.py:2229`
tests `informed_override is True` — so the default is plain NUTS with an identity
mass matrix. On EM-only, whose parameters span three orders of magnitude
(`light0_amp` ≈ 8, `noise_sigma_bkg` ≈ 0.01), that diverged:

| | r_hat | worst σ vs Fisher |
|---|---|---|
| `informed=False` (default) | ~1e15 | 5.54e8x |
| `informed=True` | ≤1.0001 | 1.02x |

This is a silent failure: finite numbers, no NaN, no warning. Only r_hat exposed
it. GW-only `deriv-approx-source` converges without it (4 comparably-scaled
parameters), which is why it went unnoticed.

### 3. Open: `Signs are not different`

`ValueError` from lenstronomy's `brentq_nojit`, reached via jaxtronomy's analytical
solver through `jax.pure_callback`. Seen **once**, in an unseeded scaled-down
EM+GW run (`bench_proto.log:517`). Not reproduced since in ~150,000 GW-only or
~7,000 EM+GW solver calls. 3,000 matched draws showed zero divergence between the
eager and jitted paths. Not attributed — `catch_solver_failure.py` will capture the
parameters if it recurs.

## Reproducing

```
python check_proto_likelihood.py        # per-call correctness + compile counts
python run_tutorial_full.py             # GW-only, full budget, B/C/D
python run_emonly_full.py               # EM-only
python run_emgw_full.py                 # EM+GW (gate, then B/C/D)
python compare_methods.py               # nautilus vs fisher vs deriv-approx
```

`outputs/` is gitignored — 675 MB of resumable nautilus checkpoints plus derived
`.npz`/`.png`, all regenerable by the above.
