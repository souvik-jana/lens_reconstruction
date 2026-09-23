# Step 2 — prototype tests on current HEAD (`nautilus-speed-up`, 4f73486)

Scripts here read `nautilus-parallel-check/` and `src/gwemfish` and write only to
`nautilus-fix-analysis/outputs/`. No file outside this directory was touched.

Run with the project venv:

```
.venv/bin/python t1_agreement_compiles.py
```

## Results

| test | question | verdict |
|---|---|---|
| T1 | does the prototype still agree, and still remove the recompiles? | **PASS** |
| T2 | is the ctx strip list still complete after a real fisher run? | **PASS** |
| T3 | what breaks today if `cfg["nautilus"]` gains `jit`/`pool`? | confirmed broken |
| T4 | what about `nautilus-image`? | **gap — worse than source-plane** |
| T5 | does `pool=N` work on darwin, not just Linux? | works, with two caveats |

### T1 — agreement and compile counts (20 draws/mode)

| mode | max rel ΔlogL | compiles/call | ms/call | speed-up |
|---|---|---|---|---|
| GW-only | 3.37e-13 | 6 → 0 | 134.5 → 8.40 | 16.0x |
| EM+GW | 4.36e-15 | 6 → 0 | 132.9 → 3.06 | 43.5x |
| EM-only | 1.14e-15 | 0 → 0 | 0.2 → 0.10 | 1.9x |

The REPORT's numbers survive the q/phi and inversion merges unchanged. EM+GW
per-call jit was listed as "not measured" in the REPORT; it is measured here.

### T2 — ctx picklability after `fisher-source`

21 keys, 58.7 KB. Unpicklable set after a real fisher run is **exactly**
`{fisher, likelihood}` — the prototype's strip list is still complete. Stripped
ctx pickles in 56.3 KB.

### T3 — cfg plumbing

`run_nautilus(prior, log_likelihood, *, n_live, filepath, verbose, resume,
prior_check, run_kwargs)`.

`_finish_nautilus_run` (`simple_pipeline.py:1793-1801`) would splat
`['jit', 'n_live', 'pool', 'resume', 'verbose']` into it →

```
TypeError: run_nautilus() got an unexpected keyword argument 'jit'
```

Also confirmed: `run_nautilus` forwards neither `pool` nor `seed` to
`nautilus.Sampler`, and captures none of nautilus' own diagnostics
(`log_z` / `n_eff` / `n_like` are computed by the sampler and thrown away).

### T4 — `nautilus-image` is the real gap

GW-only image-plane problem, 10 sampled params:

| metric | value |
|---|---|
| compiles/call | **8.0** |
| ms/call | **194.2** |
| likelihood picklable | **No** — `Can't get local object 'build_image_plane_problem.<locals>.log_likelihood'` |
| prior picklable | Yes |

Worse than the source-plane path on both counts, and the prototype covers none of
it. Its likelihood is `probmodel_log_likelihood` — numpyro `log_density` +
`trace` + `substitute` per call, Python-level tracing, not a jittable arithmetic
core. So `cfg["nautilus"]["jit"]` cannot mean the same thing for this method;
`pool` can, since it only needs a picklable wrapper.

### T5 — pool on darwin (14 cores, default start method already `spawn`)

Workers start, rebuild the likelihood in 2.9 s each, run to completion. No
deadlock (the `fork` trap is Linux-only, but forcing `spawn` is still needed
there).

Same seed, **pool held fixed at 2**:

| variant | wall | n_like | n_eff | log_z |
|---|---|---|---|---|
| eager_pool2 | 115.4 s | 1200 | 6.7234 | −63.987098 |
| jit_pool2 | 12.0 s | 1200 | 6.7234 | −63.987098 |

Identical run, 9.6x faster. Caveats:

1. **Changing `pool` changes the run.** jit_nopool gave `log_z = −63.709151`
   against −63.987098 for the same seed with `pool=2`: nautilus dispatches points
   in different batches, so seed-reproducibility holds only at a fixed `pool`
   value. Worth a line in `cfg_reference.py` — a user who adds `pool` to an
   existing config will not reproduce their old numbers.
2. **`setup_jax(ncpus=1)` does not pin the worker's device count.** The worker
   reported `OMP=1, MKL=1, OPENBLAS=1` but `JAX device count: 8`, because
   `XLA_FLAGS=--xla_force_host_platform_device_count` is inherited from the parent
   environment. Worker init has to set `XLA_FLAGS` too, or oversubscribe.

Budget here was `n_live=50, n_eff=60, n_like_max=1200` — startup-dominated, so
pool(2) vs no-pool wall (11.0 s vs 12.0 s) says nothing about throughput. The
REPORT's 2.8x stands as the throughput number.

## What this changes in the merge plan

1. `jit` is a **build-time** key, `pool` a **run-time** one. `_skip` in
   `_finish_nautilus_run` must grow, and `jit` must be routed into the builders
   via `_run_nautilus_source_inference` / `_run_nautilus_image_inference`.
2. `run_nautilus` needs `pool` and `seed` parameters, and should return nautilus'
   diagnostics instead of discarding them.
3. `nautilus-image` needs its own decision: `jit` unsupported (raise or ignore
   with a warning), `pool` supported via the picklable wrapper.
4. Worker init must set `XLA_FLAGS`, not only thread limits.
5. `cfg_reference.py` must state that changing `pool` changes the sampling path.

---

# Post-merge verification (same machine, merged code)

The tests above measured the isolated prototype. These re-run against
`src/gwemfish` after the merge, where jit/eager are selected by
`cfg["nautilus"]["jit"]` rather than by importing a different module.

| check | script | result |
|---|---|---|
| 1-2. jit vs eager, 200 draws/mode | `t1_agreement_compiles.py` | **PASS** |
| 5. EM+GW source-centre fix | `t8_emgw_truth.py` | **PASS** |
| 6. fisher / deriv-approx regression | `baseline_methods.py before\|after\|compare` | **bit-identical** |
| 7. nautilus-image warn + pool | `t10_image_method.py` | **PASS** |
| pool through run_inference | `t7_pool_merged.py` | **PASS** |
| unit tests | `pytest tests/test_nautilus_jit_pool.py` | 10 passed, 25 s |

## 1-2. Agreement and compile counts, merged code, 200 draws

| mode | max rel ΔlogL | compiles/call | ms/call | speed-up |
|---|---|---|---|---|
| GW-only | 1.07e-12 | 6 → 0 | 159.4 → 9.28 | 17.2x |
| EM+GW | 2.48e-13 | 6 → 0 | 167.9 → 3.57 | 47.0x |
| EM-only | 1.93e-15 | 0 → 0 | 0.2 → 0.11 | 1.8x |

## 5. EM+GW source centre

| | before the fix | after |
|---|---|---|
| sampled parameters | 23 | **25**, identical set to `fisher-source` (no missing, no extra) |
| `y0gw` / `y1gw` | pinned to the EM source centre | sampled |
| `cfg["priors"]["y0gw"]` | silently dropped | honoured |
| `logL(truth)` | −472,283 | **+976.0** |
| best of 200 random draws | 1.6e5 log-units *better* than truth | −28,038, i.e. truth is the maximum |

## 6. Regression

`fisher-source` and `deriv-approx-source`, same seeds, before vs after the merge:
max |Δ| = **0.000e+00** on every parameter's mean, std, first and last sample.
Wall times unchanged (14.4 → 14.5 s, 4.5 → 4.6 s).

## 7. nautilus-image

Warns once and runs eager under `jit=True`, as decided. `pool` does apply:

| | wall (600-call budget) |
|---|---|
| serial | 160.8 s |
| `pool=4` | 47.0 s (**3.42x**) |

Per-worker rebuild 2.2 s. This is the method that gains most from `pool`,
precisely because its likelihood is the one that cannot be compiled.
