# nautilus-source and the gradient methods fit different EM+GW models

**Status:** open
**Where:** `src/gwemfish/nautilus_source_inference.py:442-443`
**Found:** while benchmarking the nautilus prototype (`nautilus-parallel-check/REPORT.md`)

## Issue

In EM+GW source-plane mode, `build_em_gw_source_plane_problem` uses the **EM source
centre as the GW source position**:

```python
y0 = float(kwargs_source[0]["center_x"])
y1 = float(kwargs_source[0]["center_y"])
```

But `FlexProbModelSourcePlaneEMGW` (`flex_prob_model.py:582-609`) and the legacy
`ProbModelSourcePlane` (`prob_model.py:365-411`) sample `y0gw` / `y1gw`
**independently** of the EM source:

```python
flat["y0gw"] = p["y0gw"]()
flat["y1gw"] = p["y1gw"]()
betas = jnp.array([flat["y0gw"], flat["y1gw"]])
```

Confirmed by parameter count in a live run: Fisher reports **25** free parameters
including both `source0_center_x/y` *and* `y0gw/y1gw`; nautilus sampled **23**.

So `nautilus-source` and `fisher-source` / `deriv-approx-source` / `hmc-source` are
not fitting the same model in EM+GW mode, and their posteriors are not comparable.
EM and GW centroids are not generally co-located — different emission regions of the
host, and GW sky localisation is far coarser than EM.

## The silent part

`build_em_gw_source_plane_problem` never adds `y0gw` / `y1gw` to `default_dists`
(`:403-407` adds only `T_star`, `dL`, `noise_sigma_bkg`), and `build_nautilus_prior`
(`nautilus_common.py:173`) iterates only over `default_dists`. A
`cfg["priors"]["y0gw"]` handed to nautilus in EM+GW mode is therefore **dropped with
no error and no warning**.

Compare the GW-only builder, which does add them (`nautilus_source_inference.py:286`):

```python
default_dists.update(_gw_extra_defaults(bounds, ("T_star", "dL", "y0gw", "y1gw")))
```

## Observed effect

With the tutorial's EM source at (0.05, 0.1) and GW source at (0.2, 0.01), nautilus
burned the full 500,000-call ceiling and returned:

```text
n_like 500000   n_eff 1.0   log_z -275683.8   samples 1
```

Diagnosis: `logL at truth = -472283`, while random prior draws scored **1.63e5
log-units better** — truth was not the maximum, because the likelihood compares time
delays computed at the EM source position against data generated at the GW source
position.

Co-locating the two sources changed `logL(truth)` from **-472283 to +976** and gave
`n_eff = 37593` in 154,800 calls. But that is a workaround: it bends the simulated
data to match nautilus' assumption, and silently removes a degree of freedom the
gradient methods are fitting.

## Proposed fix

Make `build_em_gw_source_plane_problem` match the prob models: add `y0gw` / `y1gw`
to `default_dists` and use them for `betas`, instead of reading the EM source centre.
Then all four EM+GW source-plane methods sample the same 25 parameters.

Until then, `cfg["priors"]["y0gw"]` should at minimum raise rather than be ignored —
a dropped prior that changes the model is worse than an error.

## Related

- `issues/nautilus_slow_and_no_pool.md`
