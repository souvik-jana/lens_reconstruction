---
name: gwemfish-cfg
description: Authority on the GWEMFISH cfg dict — which setting controls what, where to set it, what the default is, and what breaks if it is wrong. Use when choosing or debugging any cfg key (solver_params, diagnostics, nautilus, priors, error_scales, source_box_half_width), when a run fails a [diag] check, when a setting appears to have no effect, or before adding a key to a script or YAML.
---

# GWEMFISH cfg

**Canonical source: `src/gwemfish/cfg_reference.py`** (`from gwemfish.cfg_reference import COMPLETE_CFG`). Every key is annotated inline there with mode/method applicability, type and default. `scripts/cfg.py` and `examples/scripts/cfg.py` are symlinks to it. This skill is the fast lookup; that file is the truth.

Defaults come from `make_default_cfg()` in `src/gwemfish/simple_pipeline.py`. Several keys are absent from it and read via `cfg.get(...)` at the call site — those are marked *(no default in make_default_cfg)* below.

`run_inference(ctx, cfg=...)` **deep-merges** its `cfg` onto `ctx["cfg"]`. So a per-call override changes one run without touching the rest:

```python
run_inference(ctx, mode="GW-only", method="nautilus-source",
              cfg={"gw": {"solver_params": {"backend": "helens"}}})   # this run only
```

---

## 1. Lens-equation solver — `cfg["gw"]["solver_params"]`

Nested by backend. Settings for the backend you are *not* using are carried along, not dropped, so `backend` can be flipped without re-editing. Legacy flat keys still work and are migrated with a `DeprecationWarning`.

```python
cfg["gw"]["solver_params"] = {
    "backend":       "auto",     # "auto" | "helens" | "jaxtronomy"
    "nsolutions":    "auto",     # "auto" -> n_images + 1
    "n_newton":      8,          # polish steps; int >= 1
    "duplicate_tol": None,       # None -> from position accuracy
    "helens":     {"niter": 8, "scale_factor": 2,
                   "nsubdivisions": 5, "pixel_scale_factor": 0.8},
    "jaxtronomy": {"solver": "analytical",
                   "magnification_limit": 1e-4, "Nmeas": 400, "Nmeas_extra": 80,
                   "min_distance": 0.01, "search_window": 15,
                   "precision_limit": 1e-10, "num_iter_max": 1200,
                   "arrival_time_sort": True},
}
```

### Shared keys

| key | default | controls | change when |
|---|---|---|---|
| `backend` | `"auto"` | which code finds the images | forcing one; `auto` picks jaxtronomy for `EPL`/`EPL_NUMBA`/`SIE`/`SIS` (± `SHEAR`/`CONVERGENCE`), else helens |
| `nsolutions` | `"auto"` | solution slots | `auto` = `n_images + 1`: doubles 3, triples 4, quads 5. The spare slot holds the central image when the profile has one (γ<2) and stays padding when it does not (γ≥2 is singular at the centre) |
| `n_newton` | `8` | Newton-polish **step count**, not a switch | rarely; **`0` raises** — see §7 |
| `duplicate_tol` | `None` | separation below which two solutions are one image | `None` → `1e-6"` when polished, `0.5 ×` solver pixel scale when not |

### `backend: "helens"` — triangle search, any lens model

| key | default | change when |
|---|---|---|
| `nsubdivisions` | `5` | **an image is missed — raise this first** |
| `niter` | `8` | positions not converging |
| `scale_factor` | `2` | rarely |
| `pixel_scale_factor` | `0.8` | lower = finer search grid, slower |

### `backend: "jaxtronomy"`, `solver: "analytical"` — closed form, EPL-like only

| key | default | change when |
|---|---|---|
| `magnification_limit` | `1e-4` | junk roots appear (raise). **Do not raise to `1e-1`** — see §8 |
| `Nmeas` | `400` | an image is missed |
| `Nmeas_extra` | `80` | an image is missed at low shear |

### `backend: "jaxtronomy"`, `solver: "lenstronomy"` — grid + Newton, any lens model

| key | default | change when |
|---|---|---|
| `search_window` | `15` | must cover the image separation |
| `min_distance` | `0.01` | images closer than this get merged |
| `precision_limit` | `1e-10` | rarely |
| `num_iter_max` | `1200` | rarely |
| `arrival_time_sort` | `True` | rarely (both jaxtronomy solvers) |

### Which methods read this

| method | uses a solver? | notes |
|---|---|---|
| `fisher-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source` | yes | one build site, `_build_inference_probmodel_source_plane` |
| `nautilus-source` | yes | same solver object, same settings |
| `fisher`, `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image` | **no** | sample `image_x{i}`/`image_y{i}` directly; use `cfg["gw"]["image_box_half_width"]` |
| simulation / truth (`setup_lens`) | yes | **always jaxtronomy**, whatever `backend` says — governed by `jaxtronomy.solver`. Keep that key set even with `backend: "helens"` |

---

## 2. Diagnostics — `cfg["inference"]["diagnostics"]`

Six checks at the **truth** point, before sampling. The truth is the only place a solver failure can be told apart from real physics: during sampling, "solver missed an image" and "source moved outside the caustic" both look like a wrong image count.

| value | behaviour |
|---|---|
| `"warn"` (default) | prints `FAIL`, warns, **run continues** |
| `"raise"` | aborts before sampling |
| `"off"` | skipped — **unsafe for `*-source`**: their Fisher expansion is built *at* truth, so a bad solve there corrupts everything with no signal |

| # | check | fails when | affects verdict |
|---|---|---|---|
| 1 | images | `n_found > nsolutions`; `n_distinct != n_images`; position error `> position_tol` | yes |
| 2 | observables | array-length mismatch; time-delay or dL_eff rel error `> observable_rtol` | yes |
| 3 | source box | caustic margin `<` `source_box_half_width` | **no — advisory only** |
| 4 | parameters | `n_free > n_obs` — **GW-only only**; `EM+GW` / `EM-only` print the tally marked `NA` | GW-only only |
| 5 | fisher cond | eigenvalue positive above noise floor (truth is a saddle); `cond > condition_limit` | yes |
| 6 | gradient | any `\|g0/√\|H_ii\|\| > gradient_sigma`; non-finite gradient | yes |

Checks 1–3 need a solver (skipped for image-plane methods and `EM-only`); 4–6 need a Fisher expansion (skipped for the two nautilus methods).

Check 4 counts, check 5 measures. With EM data the image pixels constrain the model too and never enter `n_obs`, so counting 16 free parameters against a quad's 7 GW observables would fail every `EM+GW` run — hence `NA` there, with the verdict left to check 5.

### Thresholds — `cfg["inference"]["diagnostics_thresholds"]`

Give **only what you want to change**; omitted keys keep the default. A misspelled key raises.

| key | default | check | meaning |
|---|---|---|---|
| `position_tol` | `1e-4` arcsec | 1 | solved vs simulated image positions |
| `observable_rtol` | `1e-3` | 2 | time delays, dL_eff |
| `condition_limit` | `1e10` | 5 | scaled Fisher condition number |
| `gradient_sigma` | `0.5` | 6 | how many σ the truth sits from the peak |

There is no per-check on/off switch. To disarm one, raise its threshold rather than setting `diagnostics: "off"`, which disables the checks still doing useful work.

**`gradient_sigma` meaning:** `g0_i/√|H_ii|` *is* the offset from the likelihood peak measured in that parameter's own σ (near a peak the offset is `−g_i/H_ii` and the width is `1/√|H_ii|`; divide and the `H_ii` cancels). Raw gradients cannot be thresholded because parameters span ~12 orders of magnitude. Healthy runs land at `1e-12`–`1e-14`.

**`condition_limit` calibration** (measured, not chosen — the number is the ratio of largest to smallest Fisher eigenvalue after scaling each parameter by its own σ; error-ellipsoid axis ratio is `√cond`):

| system | free vs obs | cond | widths |
|---|---|---|---|
| double | 3 vs 3 | 4.1e1 | usable |
| quad + central | 5 vs 9 | 7.9 | usable |
| quad | 5 vs 7 | 2.5e4 | usable |
| 3-image, e2 fixed | 4 vs 5 | 5.2e8 | σ/truth 0.23–1.14 |
| 3-image, e2 free | 5 vs 5 | 9.0e12 | **σ/truth 13.7–32.4** |

`1e10` sits in the gap between the last two rows.

---

## 3. How many parameters the data can carry

GW-only supplies `2·n_images − 1` numbers: `n_images − 1` time delays + `n_images` effective distances.

| images | observables | free parameters it supports |
|---|---|---|
| 2 double | 3 | ~3 |
| 3 naked cusp | 5 | ~4 |
| 4 quad | 7 | ~5 |
| 5 quad+central | 9 | ~5+ |

Free as many as there are observables and the Fisher goes degenerate — it still inverts, but widths come back many times the parameter values. Count from the priors dict, not the image count alone: `y0gw`/`y1gw` are always sampled, `T_star`/`dL` unless pinned.

---

## 4. Nautilus — `cfg["nautilus"]` *(no default in make_default_cfg; read via `cfg.get`)*

Nested sampling reads **this** block. `cfg["inference"]` keys (`num_chains`, `num_warmup`, `num_samples`, `informed`, `regularize`) are NUTS controls and are **silently ignored** here.

| key | default | forwarded to |
|---|---|---|
| `n_live` | `500` | `nautilus.Sampler` |
| `filepath` | `None` | checkpoint; essential at this cost |
| `resume` | `True` | `Sampler` |
| `prior_check` | `True` | writes `<filepath>.priors.json`; raises on a prior mismatch when resuming |
| `verbose` | `True` | `sampler.run` |
| `n_eff`, `n_like_max`, `discard_exploration`, `timeout` | `None` | `sampler.run(**run_kwargs)` |
| `polish` | `"auto"` | **`nautilus-source` only** — see §7 |
| `solver_backend` | — | **deprecated** alias of `solver_params["backend"]`; loses to it when both are set |
| `solver_validation_tol` | `0.05` arcsec | truth-position residual warning |

**Checkpoint trap:** `prior_check` catches prior changes on resume. It does **not** catch changes to `n_live`, `sigma_td`, `epsilon`, or the solver — those silently resume the old problem. Delete the `.hdf5` after changing any of them.

**`n_like_max` trap:** hitting it is silent. Measured: `n_like_max=3000` on a 5-parameter problem returned **1 sample**, all finite, no warning. Check the returned sample count before plotting.

**Comparing nautilus with the NUTS methods:** nautilus samples the whole prior box rather than expanding around the truth, and its default `y0gw`/`y1gw` box is `(-1, 1)` versus the truth-centred `±source_box_half_width` the others use. Set `cfg["gw"]["source_plane_bounds"]` to match, or the posteriors differ because the priors differ.

Per-call cost is inflated ~6x by JAX recompiling every call (no jit on the
nautilus likelihood path), and pool=N cannot be used (likelihood is an
unpicklable local closure). Measured: jit+pool gives 41x GW-only, 1.7x EM-only.
See issues/nautilus_slow_and_no_pool.md.

---

## 5. Computation cost

Per likelihood call, 4-image system, 40×40 grid. Every call solves the lens equation; jaxtronomy runs on the host behind `jax.pure_callback` (one round-trip per call).

| backend | polish | ms/call | per 1e5 calls |
|---|---|---|---|
| jaxtronomy | `False` | **41** | **~69 min** ← what `polish: "auto"` picks |
| jaxtronomy | `True` | 99 | ~165 min |
| helens | `False` | 58 | ~96 min |
| helens | `True` | 120 | ~200 min |

Consequences:

- **`nautilus-source` costs hours, not minutes.** Budget accordingly; use `n_like_max` as a ceiling and a checkpoint.
- Forcing `polish: True` with jaxtronomy costs **2.4× for no accuracy gain** — those positions are already exact.
- `hmc-source`/`hmc-informed-source` call the solver at **every leapfrog step** (up to ~1024 per sample at `max_tree_depth=10`).
- `fisher-source`/`deriv-approx-source` call it **only at `u0`** — `jax.hessian` is forward-mode AD at one point, and sampling then runs on a Gaussian or the Taylor surrogate. These are the cheap ones.

---

## 6. Where the solver actually matters, by method

| method | solver at | in-model image-count rejection |
|---|---|---|
| `fisher-source` | `u0` only | not needed |
| `deriv-approx-source` | `u0` only | not needed |
| `hmc-source`, `hmc-informed-source` | every leapfrog step | `numpyro.factor("image_count", -1e10)` |
| `nautilus-source` | every likelihood call | early `return -1e300` |

The `-1e10` factor is a **safety net, not the control**. Being constant it has zero gradient, so NUTS cannot be pushed back and reports a **divergence**. The real control is keeping the source prior box inside the caustic (`cfg["gw"]["source_box_half_width"]`), which diagnostic check 3 measures.

The factor contributes exactly `0` to `logp0`, `g0` and `H0` at a valid truth, so the Taylor surrogate is unchanged. Consequence: `fisher-source` and `deriv-approx-source` have **no runtime protection** — their only safeguard is the truth-point diagnostic.

---

## 7. The polish, and `n_newton`

Neither finder is differentiable: helens selects with piecewise-constant ops, jaxtronomy runs in numpy behind `pure_callback`. Both are wrapped in `stop_gradient`, and a Newton polish with `jax.lax.custom_root` re-attaches derivatives via the implicit function theorem. **That is where differentiability comes from — not the finder.** Hence the finder is swappable, and both backends give identical derivatives (verified: max relative difference `4.5e-13`).

The polish does two jobs: **accuracy** (helens only — 0.05″ → ~1e-14) and **derivatives** (both, always).

| key | scope | values |
|---|---|---|
| `solver_params["n_newton"]` | every method that polishes | int ≥ 1, default 8. **Step count, not a switch** |
| `cfg["nautilus"]["polish"]` | **`nautilus-source` only** | `"auto"` / `True` / `False` |

`"auto"` = polish only when it changes the answer: `False` for jaxtronomy finders (already exact), `True` for helens.

| | can polishing be turned off? | can the step count change? |
|---|---|---|
| the four gradient methods | **no** | yes — `n_newton` |
| `nautilus-source` | yes — `cfg["nautilus"]["polish"]` | yes — `n_newton` |

**`n_newton: 0` raises** on any method that polishes. Zero steps means `custom_root` never runs, every derivative comes back exactly `0.0` with no error, `H0` is all zeros and the covariance is NaN. Verified as a negative control.

---

## 8. `magnification_limit` — do not raise it casually

Measured for `EPL`+`SHEAR`, θ_E=2:

| γ | 5th root \|μ\| | `1e-4` (default) | `1e-1` (jaxtronomy's own suggestion) |
|---|---|---|---|
| 2.0 | no 5th root | — | — |
| 1.9, 1.7 | **0.0** (junk) | drops it ✓ | drops it |
| 1.5 | **1.39e-3** (real central image) | **keeps it** ✓ | **discards it** ✗ |

`1e-1` silently removes a genuine central image at γ=1.5. It also sets what counts as an image for the **simulation** — both sides go through the same solver — so lowering it can raise `n_images`. That is intentional: the two stay consistent by construction.

---

## 9. Other keys that bite

| key | default | note |
|---|---|---|
| `cfg["gw"]["source_box_half_width"]` | `0.05` *(no default in make_default_cfg)* | `y0gw`/`y1gw` prior half-width. Naked-cusp systems sit close to the caustic (catalog 555 has 0.042″ of margin) — a box past it produces NUTS divergences that look like solver failure. Check 3 prints the margin. Measured: box past the margin cost 2133s vs 172s on a corrected box. |
| `cfg["gw"]["image_box_half_width"]` | `0.6` | half-width of the truth-centred `image_x{i}`/`image_y{i}` prior box; image-plane methods only |
| `cfg["gw"]["n_images"]` | `4` | a **hint**. `_resolve_gw_n_images` prefers `len(ctx["x_img_gw"])` and warns on mismatch; raises if `truth_params`/`gw_obs` disagree |
| `cfg["gw"]["error_scales"]["sigma_td"]` | `0.05` | σ = fraction × observed time delays |
| `cfg["gw"]["error_scales"]["sigma_dL_eff"]` | `0.2` | σ = fraction × observed dL_eff |
| `cfg["gw"]["error_scales"]["sigma_td_floor"]` | `1.0` s *(no default)* | stops tiny delays collapsing the likelihood width |
| `cfg["gw"]["error_scales"]["epsilon"]` | — | width on the ray-shooting self-consistency terms |
| `cfg["inference"]["fisher_order"]` | `2` | 2 = Hessian; 3 adds `F0`; 4 adds `Q0` |
| `cfg["inference"]["H0"]` | `None` | optional Hessian override for informed NUTS |
| `cfg["inference"]["regularize"]` | `False` *(no default)* | clips small/negative eigenvalues in the informed-NUTS mass matrix |
| `cfg["use_parameter_layout"]` | `False` | `True` gives `lens{i}_*`/`source{j}_*`/`light{k}_*` names; required for EM-only nautilus |

---

## 10. Priors — `cfg["priors"]`

Flat `{param_name: value}`. Three accepted forms:

| form | effect |
|---|---|
| plain float | **fixed** — held at that value and dropped from `keys_to_include`, so Fisher does not differentiate w.r.t. it |
| numpyro `Distribution` | sampled from it |
| callable returning `numpyro.sample(...)` | sampled |

`parse_cfg_priors` converts numpyro distributions to scipy for the nautilus paths automatically, so the same dict works for every method.

Names depend on `use_parameter_layout`: `lens0_theta_E`, `lens0_e2`, `source0_n_sersic`, `light0_amp` with it; legacy flat `lens_theta_E`, `lens_e1` without.

---

## 11. What is saved

`run_inference` writes, all tagged by method:

| artifact | contents |
|---|---|
| `save_samples_path` `.npz` | one array per sampled parameter |
| `save_truths_path` `.npz` | truths for `keys_all` (includes fixed params) |
| `json_path` `.json` | `injection_parameters`, `setup_parameters` (cfg, kwargs_lens, lens_model_list, n_images), samples, truths, **`fisher`** (`keys`, `u0`, `logp0`, `g0`, `g0_scaled`, `H0`), **`diagnostics`** (the full report) |

`F0`/`Q0` are not saved — rank-3/4 tensors, large, not diagnostics. The two nautilus methods produce no `fisher` block (they never build one).

---

## 12. Debugging table

| symptom | cause | fix |
|---|---|---|
| a setting appears to have no effect | wrong nest, or a method that does not read it | check §1's "which methods read this"; flat keys warn on migration |
| `[diag] images ... n_distinct != n_images` | solver missed/duplicated an image | raise `helens.nsubdivisions` or `jaxtronomy.Nmeas`; or switch `backend` |
| `[diag] fisher cond ... FAIL`, huge widths | degenerate Fisher: some directions unconstrained | fix a parameter via `cfg["priors"]`, or add EM data (§3) |
| `[diag] parameters ... FAIL` (GW-only) | more free parameters than GW observables | fix one via `cfg["priors"]`. EM+GW / EM-only print `NA` here: the pixels constrain the model and never enter the count |
| `[diag] gradient` large | truth is not the likelihood peak | usually truth/model disagreement; check `[diag] images` first |
| NUTS divergences | source box outside the caustic | lower `source_box_half_width` to the margin check 3 prints |
| NaN samples from `fisher*` | non-PSD covariance | now whitened + clipped with a warning; check `[diag] fisher cond` |
| nautilus returns ~1 sample | `n_like_max` hit during exploration | raise it, delete the checkpoint |
| nautilus ignores your settings | put in `cfg["inference"]` instead of `cfg["nautilus"]` | move them (§4) |
| nautilus posterior differs from NUTS | different source prior box | set `source_plane_bounds` to match (§4) |
| `n_newton: 0` raises | it is a step count | use `cfg["nautilus"]["polish"]` (§7) |

---

## 13. Lens mass ellipticity — `cfg["lens_mass_parametrization"]`

`"e1e2"` (default) or `"q_phi"`. Verified across fisher, deriv-approx, hmc-informed,
nautilus-source, nautilus-image, all three modes — see `issues/` below.

e1/e2 land in the output either way, **except**: fisher/deriv-approx/nautilus with
`lens0_phi` fixed produce no e1/e2 columns at all (backfill needs phi in samples,
not just q). hmc-informed always has them (numpyro.deterministic sites).

Two known limits, both benign on tested systems:
- fisher's Gaussian draw is unbounded and ignores the `q∈[0.01,1]` prior — near
  round lenses (q→1) some draws land at q>1, which folds onto the 90°-rotated
  ellipse rather than erroring. 17σ clear on tutorial systems.
- phi is π-periodic against a hard `Uniform(-π/2,π/2)` box, no wrap. Matters only
  for a near-vertical or near-circular lens. 44σ clear here.

Worked examples: `examples/scripts/qphi_*.py`.
