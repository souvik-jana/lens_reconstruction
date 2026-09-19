---
name: gwemfish-infer
description: Runs GWEMFISH run_inference with mode/method selection, priors from ctx, nautilus H0 priors, multi-method RUN toggles/checkpoints, and sample extraction. Use when inferring lens parameters, setting priors, choosing fisher/deriv-approx/hmc/nautilus (image-plane or source-plane, e.g. fisher-source/deriv-approx-source/hmc-informed-source), building nautilus-source priors from a precursor run, adding methods to comparison scripts, or comparing methods.
---

# GWEMFISH infer

**Do not call `run_inference` until the user has chosen options below.** Use `AskQuestion` when available; skip only if the user already specified everything in the same message.

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for repo paths if set; else use the open repo root. Start from `src/gwemfish/cfg_reference.py` (canonical complete config, every key documented inline; `from gwemfish.cfg_reference import COMPLETE_CFG, get_cfg`; `scripts/cfg.py` and `examples/scripts/cfg.py` are symlinks to it); copy narrower priors patterns from the closest `examples/scripts/` file when useful.

## Question 1 — Mode (ask if unclear)

| Choice | `mode=` | Setup needed |
|--------|---------|--------------|
| EM only | `EM-only` | `setup_em_observation` |
| GW only | `GW-only` | `setup_gw_observation`, `em.enabled: false` |
| Joint | `EM+GW` | both |

## Question 2 — Method (ask if unclear)

| Choice | `method=` |
|--------|-----------|
| Fisher | `fisher` |
| Derivative approx | `deriv-approx` |
| Full HMC | `hmc` |
| HMC informed | `hmc-informed` |
| Fisher (source-plane) | `fisher-source` |
| Derivative approx (source-plane) | `deriv-approx-source` |
| Full HMC (source-plane) | `hmc-source` |
| HMC informed (source-plane) | `hmc-informed-source` |
| Nautilus source-plane | `nautilus-source` |
| Nautilus image-plane | `nautilus-image` |

`fisher-source`/`deriv-approx-source`/`hmc-source`/`hmc-informed-source` sample the GW source position (`y0gw`/`y1gw`) directly and solve the lens equation inside the model. Valid only for `mode` in (`GW-only`, `EM+GW`) — **not** `EM-only` (unlike every other method here, which does support `EM-only`).

`fisher-source` mirrors `fisher` exactly (Taylor-Gaussian `N(u0, inv(-H0))`, no MCMC) but with `H0`/`keys_to_include`/`u0` built on the source-plane probmodel — cheapest way to get a source-plane covariance, and typically much better conditioned than image-plane `fisher` (no image-multiplicity redundancy in `y0gw`/`y1gw` vs `image_x*`/`image_y*`).

Multi-method comparison allowed (e.g. deriv-approx + nautilus-source + fisher in `em_nautilus.py`).

For GW modes, ask which Nautilus variant when unclear: **source-plane** (`y0gw`/`y1gw`) vs **image-plane** (`image_x*`/`image_y*`). EM-only: either name works; prefer `nautilus-source`.

`cfg["lens_mass_parametrization"]="q_phi"` verified across fisher, deriv-approx,
hmc-informed, nautilus-source, nautilus-image, all 3 modes. deriv-approx needs
`informed=True` on EM-only or it diverges (r_hat ~1e15 measured).

### Cost — decides which methods are practical

Per likelihood call, 4-image system, 40×40 grid. Every call solves the lens equation; jaxtronomy runs on the host behind `jax.pure_callback` (one round-trip per call).

| backend | polish | ms/call | per 1e5 calls |
|---|---|---|---|
| jaxtronomy | `False` | **41** | **~69 min** ← `polish: "auto"` picks this |
| jaxtronomy | `True` | 99 | ~165 min |
| helens | `False` | 58 | ~96 min |
| helens | `True` | 120 | ~200 min |

How often each method calls the solver:

| method | solver evaluated at | practical cost |
|---|---|---|
| `fisher-source`, `deriv-approx-source` | **`u0` only** — `jax.hessian` is forward-mode AD at one point; sampling then runs on a Gaussian or the Taylor surrogate | cheap |
| `hmc-source`, `hmc-informed-source` | every leapfrog step — up to ~1024 per sample at `max_tree_depth=10` | expensive |
| `nautilus-source` | every likelihood call, ~1e5–1e6 of them | **hours** |
| `fisher`, `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image` | never — image positions are sampled directly | n/a |

Practical workflow: iterate with `fisher-source`/`deriv-approx-source`, run `nautilus-source` once at the end with a checkpoint.

### How many parameters the data can carry

GW-only supplies `2*n_images - 1` numbers (`n_images - 1` time delays + `n_images` effective distances):

| images | observables | free parameters supported |
|---|---|---|
| 2 double | 3 | ~3 |
| 3 naked cusp | 5 | ~4 |
| 4 quad | 7 | ~5 |
| 5 quad+central | 9 | ~5+ |

Free as many as there are observables and the Fisher goes degenerate: it still inverts, but widths come back many times the parameter values. Measured on catalog 555 (3-image): with 5 free, σ/truth was 13.7–32.4; fixing one parameter gave 0.23–1.14. Count the budget from the priors dict — `y0gw`/`y1gw` are always sampled, `T_star`/`dL` unless pinned — not from the image count alone. Diagnostic check 4 reports the count (GW-only only) and check 5 the Fisher conditioning, both before sampling.

## Question 2.5 — Parameter layout (ask if unclear, `GW-only`/`EM+GW`, any method)

Ask: does the lens have more than one independently-parametrized mass component (e.g. main lens + a second galaxy — not just a fixed external shear), or does the user want auto-generated per-profile priors?

- **Yes / unclear-but-useful** → `cfg["use_parameter_layout"] = True`. Flat names become `lens0_*`, `lens1_*`, ... — one block per entry in `lens_mass_model.func_list` (e.g. EPL + Shear = `lens0_*` + `lens1_*`). Priors auto-derived per profile.
- **Default / legacy** → `False`. Old hardcoded single-lens flat names (`lens_theta_E`, `lens_e1`, ...). Still the default, still works for every method below, no breaking change.

Applies uniformly to **all** of: `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image`, `fisher-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`, `nautilus-source`. Only `EM-only`'s nautilus path *requires* it (see reference.md); everywhere else it's opt-in.

## Question 3 — Informed NUTS (deriv-approx/hmc and their source-plane counterparts)

Ask: use Hessian-informed NUTS?

- **Default yes** → `cfg["inference"] = {"informed": True}`
- No → `{"informed": False}`
- `hmc-source` behaves like `hmc` (optional `informed`); `hmc-informed-source` behaves like `hmc-informed` (always informed, flag ignored); `deriv-approx-source` behaves like `deriv-approx` (optional `informed`, default recommendation still yes).
- N/A for `fisher`, `fisher-source`, `hmc-informed`, `hmc-informed-source`, `nautilus-source`, `nautilus-image` (the two `fisher*` methods never run NUTS at all)

## Question 3.5 — epsilon and image-plane/source-plane comparability

**Applies when comparing any image-plane sampler to any source-plane sampler.** Image-plane methods share `ProbModel` and `cfg["gw"]["error_scales"]["epsilon"]` (default **0.005**):

| Image-plane (`ProbModel`) | Source-plane (no `ProbModel`) |
|---------------------------|-------------------------------|
| `fisher`, `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image` | `fisher-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`, `nautilus-source` |

Note on `fisher`/`deriv-approx` specifically: they sample the Taylor/banana model (`ProbModelFisher*`), which has no live `epsilon` site of its own — but the `H0` that shapes that model is built by `compute_fisher` on the *full* `ProbModel` (which does have `epsilon`), evaluated once at the expansion point. So epsilon's curvature is baked into `H0`, and therefore into the resulting width, even with no per-sample `epsilon` penalty visible in a `deriv-approx` trace. Net effect on the table above is unchanged; just the mechanism.

Inside `ProbModel.model()` (`flex_prob_model.py` / `prob_model.py`), `epsilon` sets a soft `Normal(0, epsilon)` on `betx_x_diff`/`bety_y_diff` (images must ray-shoot to one source) plus `log_jacobian = -sum(log|mu_i|)`. Every source-plane method forward-solves from sampled `y0gw`/`y1gw` (`ProbModelSourcePlane*` / `FlexProbModelSourcePlane*`) — no equivalent term, no betx/bety scatter floor.

**Consequence:** `to_source_plane_samples` from any image-plane method has a scatter floor from `epsilon`, not just the GW likelihood. Loose default (`0.005`) can make image-plane methods look ~2–3× wider than any source-plane method (`nautilus-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`) on the same system even when both are correct.

**Before comparing to a source-plane method:** tighten epsilon (start **`1e-4`**):

```python
ctx["cfg"]["gw"]["error_scales"]["epsilon"] = 1e-4
```

- **NUTS** (`deriv-approx`, `hmc`): if divergences/stiffness, back off (e.g. `1e-3`) and/or raise `num_warmup`.
- **Nautilus-image**: sharper surface — raise `n_live` / `n_eff` if acceptance slows.
- **Diagnostic:** per-sample `y0gw_std` / `y1gw_std` from `image_to_source.image_samples_to_source_samples` — if large vs posterior width, `epsilon` dominates scatter; comparison to a source-plane method is not apples-to-apples yet.

**Does NOT apply:** two image-plane methods vs each other (e.g. `deriv-approx` vs `nautilus-image`), or two source-plane methods vs each other (e.g. `hmc-source` vs `nautilus-source`) — same model family, no epsilon asymmetry; they should agree regardless of epsilon. If they disagree, treat as a real bug (prior mismatch, non-convergence, solver backend, wiring) — not an epsilon caveat.

See `gwemfish-plot` skill for overlay interpretation.

## Question 4 — Nautilus variant (when method includes nautilus-source or nautilus-image)

### Three traps to check before anything else

**1. Nautilus reads `cfg["nautilus"]`, not `cfg["inference"]`.** `num_chains`, `num_warmup`, `num_samples`, `informed`, `regularize` are NUTS controls and are **silently ignored** — the run uses defaults throughout and says nothing. Correct block:

```python
cfg={
    "nautilus": {
        "n_live": 500,
        "n_eff": 2000,
        "n_like_max": 200_000,          # stop valve — see trap 2
        "filepath": os.path.join(out, "nautilus_source.hdf5"),
        "resume": True, "prior_check": True, "verbose": True,
        # "polish": "auto" is the default and the fast path
    },
}
```

**2. `n_like_max` hits silently.** Measured: `n_like_max=3000` on a 5-parameter problem returned **1 sample**, all finite, no warning — it would have gone straight into a corner plot. Always check the count:

```python
n = min(len(np.asarray(v)) for v in samples.values())
if n < 100:
    raise RuntimeError(f"nautilus returned {n} samples — hit n_like_max during exploration")
```

**3. Comparing nautilus with the NUTS methods needs matching priors.** Nautilus samples the whole prior box rather than expanding around the truth, and its default `y0gw`/`y1gw` box is `(-1, 1)` against the truth-centred `±source_box_half_width` the others use. Without this the posteriors differ because the *priors* differ:

```python
SRC = ctx["cfg"]["gw"]["source_pos"]; HW = float(ctx["cfg"]["gw"]["source_box_half_width"])
ctx["cfg"]["gw"]["source_plane_bounds"] = {
    "y0gw": (SRC[0] - HW, SRC[0] + HW),
    "y1gw": (SRC[1] - HW, SRC[1] + HW),
}
```

Also: `prior_check` catches prior changes on resume, but **not** changes to `n_live`, `sigma_td`, `epsilon`, or the solver — those silently resume the old problem. Delete the `.hdf5` after any of them.

`truths_nautilus` will not contain `y0gw`/`y1gw` (they are never in `truth_params`), so merge the source position in explicitly when plotting truths.

**4. EM+GW `nautilus-source` ties the GW source to the EM source centre** -- gradient
methods sample `y0gw`/`y1gw` independently (23 vs 25 free params in a live run).
`cfg["priors"]["y0gw"]` given to nautilus in this mode is silently dropped (never
added to `default_dists`). See issues/nautilus_emgw_source_centre.md.


1. **Precursor** — for `nautilus-image`, use `deriv-approx` (default, `informed: True`) or `fisher`, same as always. For `nautilus-source`, four precursors are valid — ask which:
   - **`fisher-source` / `deriv-approx-source`** (recommended default when targeting `nautilus-source`) — direct. `H0`/`keys_to_include`/`u0` are already in `y0gw`/`y1gw` + shared-parameter form, matching `nautilus-source`'s sampling space exactly. `fisher-source` is the cheap no-NUTS option; `deriv-approx-source` gives a real posterior if you want it.
   - **`fisher` / `deriv-approx` (image-plane)** — also valid now, but needs conversion: their `keys_to_include` has `image_x*`/`image_y*` instead of `y0gw`/`y1gw`. Shared lens-mass keys carry over directly; position keys must be ray-shot to the source plane first (`image_samples_to_source_samples`/`to_source_plane_samples` in `image_to_source.py`) before taking their sigma. Use `nautilus_source_priors_from_precursor(ctx, samples, method, span=...)` from `Diagnosis/scripts/task13_nautilus_source_prior_from_precursors/nautilus_source_priors.py` rather than hand-rolling this — it dispatches correctly on all four precursor methods and was validated 2026-07-12 (all four gave `nautilus-source` posteriors agreeing within ~7% on `y0gw`/`y1gw` std on the test system).
2. **Tight Fisher H₀ priors** — after precursor, set Uniform priors from H₀ before nautilus:

| Context | Default `NAUTILUS_SIGMA_SPAN` | Example script |
|---------|------------------------------|----------------|
| EM-only nautilus | **5.0** | `em_nautilus.py` |
| GW-only nautilus (source or image) | **2.0** | `gw_only_nautilus.py`, `gw_only_nautilus_image.py` |

Workflow:

1. Run precursor (`nautilus-image` target: `deriv-approx` with `informed: True`, or `fisher`. `nautilus-source` target: `fisher-source`/`deriv-approx-source` direct, or `fisher`/`deriv-approx` via the conversion helper above)
2. Read `ctx["likelihood"]["keys_to_include"]`, `u0`, `ctx["fisher"]["H0"]`
3. `sigmas = sqrt(diag(inv(-H0)))`
4. For each key: `ctx["cfg"]["priors"][key] = dist.Uniform(mu - span*sig, mu + span*sig)` (skip invalid σ)
5. Set **`cfg["nautilus"]["resume"] = False`** when priors or `NAUTILUS_SIGMA_SPAN` change (or delete the `.hdf5`); `True` for same-prior re-run. A default-on guard (`prior_check`, see item 3 below) hard-errors if you resume under mismatched priors. Comparison scripts may expose this as top-level `NAUTILUS_RESUME` forwarded into `nautilus_cfg["nautilus"]["resume"]` (Question 5).
6. Run `nautilus-source` or `nautilus-image`

```python
NAUTILUS_SIGMA_SPAN = 2.0  # 5.0 for em_nautilus.py

print("\n--- Nautilus priors from Fisher H0 (deriv-approx) ---\n")
keys = ctx["likelihood"]["keys_to_include"]
u0 = np.asarray(ctx["likelihood"]["u0"])
H0 = np.asarray(ctx["fisher"]["H0"])
FM = -H0
try:
    cov = np.linalg.inv(FM)
except np.linalg.LinAlgError:
    cov = np.linalg.pinv(FM)
sigmas = np.sqrt(np.diag(cov))

for i, key in enumerate(keys):
    sig = float(sigmas[i])
    if not np.isfinite(sig) or sig <= 0:
        print(f"  Nautilus prior {key}: skip (sigma={sig}) — keep existing prior")
        continue
    mu = float(u0[i])
    lo = mu - NAUTILUS_SIGMA_SPAN * sig
    hi = mu + NAUTILUS_SIGMA_SPAN * sig
    ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
    print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")
```

**Source-plane caveat (any source-plane method — `nautilus-source`, `fisher-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`):** H₀ from image-plane `fisher`/`deriv-approx` covers `lens0_*`, `T_star`, `dL`, `image_x*`/`image_y*` — **not** `y0gw`/`y1gw` directly, and image keys set in `ctx["cfg"]["priors"]` are ignored by source-plane methods. Two options: (a) use a source-plane precursor (`fisher-source`/`deriv-approx-source`) so `y0gw`/`y1gw` come out natively, or (b) convert an image-plane precursor's position keys via `nautilus_source_priors_from_precursor` (see step 1 above) rather than leaving `y0gw`/`y1gw` at manual/default boxes. Manual `y0gw`/`y1gw` boxes via `ctx["cfg"]["priors"]` and/or `cfg["gw"]["source_plane_bounds"]` (nautilus) / `cfg["gw"]["source_box_half_width"]` (NUTS-based source-plane methods) remain a fine fallback when no precursor is run at all. For `nautilus-image`, all H₀ keys from an image-plane precursor apply directly, no conversion needed.

3. **Checkpoint** — set resume via **`cfg["nautilus"]["resume"]`** (read by `run_inference`). When priors or `NAUTILUS_SIGMA_SPAN` change, use `False`; same-prior re-run → `True`. Comparison scripts forward script-top `NAUTILUS_RESUME` into that key:

```python
# Direct cfg override (canonical)
run_inference(ctx, mode="GW-only", method="nautilus-source", cfg={
    "nautilus": {
        "filepath": "outputs/nautilus_checkpoint.hdf5",
        "resume": False,
    },
})

# Comparison script pattern (gw_only_nautilus.py)
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False
nautilus_cfg = {"nautilus": {"filepath": NAUTILUS_CHECKPOINT, "resume": NAUTILUS_RESUME}}
```

`BASE_CFG["nautilus"]["resume"]` in `ctx` sets a default; per-call `cfg` override wins via deep-merge.

**Prior-mismatch guard (`prior_check`, default `True`):** nautilus stores unit-cube points and maps them through the *current* prior at `posterior()` time, so resuming a checkpoint under different priors used to silently rescale the stored points onto the new boxes (observed 2.5–7.5× σ inflation — see `examples/scripts/NAUTILUS_CHECKPOINT_NOTE.md` and `Diagnosis/scripts/task14_deriv_approx_source_order_dependence/report.md`). `run_nautilus(..., prior_check=True)` now writes a prior-fingerprint sidecar **`<filepath>.priors.json`** on every checkpointed run (per-parameter ppf quantiles; for Uniform priors the outer two ≈ the box edges — `cat` it to see which priors a checkpoint was built under). On `resume=True` it compares current priors to the sidecar and **raises `ValueError` on mismatch** — no silent corruption when a sidecar exists. Legacy checkpoints without a sidecar warn once and get one written. Opt out with `cfg["nautilus"]["prior_check"] = False` (the key flows through `cfg["nautilus"]` to `run_nautilus` like `filepath`/`resume`; applies to both `nautilus-source` and `nautilus-image`). **Limits:** it fingerprints priors only — changed likelihood settings (`sigma_td`, `epsilon`, `solver_backend`) or `n_live` are NOT detected; after changing free parameters, spans, priors, or error scales, set `resume=False` or delete the checkpoint.
4. **GW-only prior choice** — default: Fisher H₀ with `NAUTILUS_SIGMA_SPAN = 2.0` after deriv-approx. Alternative: manual tight truth boxes (`SOURCE_HALF_*`, `IMAGE_BOX_HALF`) when no precursor or for `y0gw`/`y1gw` on source-plane runs

## Question 5 — Multi-method script wiring (ask when adding a method to a comparison script)

**When to ask:** user is adding or modifying a method in a multi-method comparison script (`gw_only_nautilus.py`, `em_nautilus.py`, diagnosis scripts, etc.). Skip if user already specified bare block vs toggle vs checkpoint in the same message.

**AskQuestion prompt:** "How should this method be wired in the script?"

| Option | Meaning |
|--------|---------|
| **Bare block** | Always-run `run_inference` block; method runs every time the script runs |
| **RUN toggle** (recommended for comparison scripts) | Top-level `RUN_<METHOD> = True/False`; only enabled methods run and appear in comparison plots |
| **RUN toggle + checkpoint** | Same as above, plus save/load path so user can skip re-inference on re-runs |

Comparison scripts should default to **RUN toggles**. Canonical reference: `examples/scripts/gw_only_nautilus.py`.

### Canonical toggle pattern

```python
# Method toggles — only blocks with True are executed and included in comparison plots.
RUN_DERIV_APPROX = True
RUN_DERIV_APPROX_SOURCE = True
RUN_FISHER_SOURCE = True
RUN_NAUTILUS_SOURCE = True
RUN_FISHER = False
RUN_HMC_SOURCE = False
RUN_HMC_INFORMED_SOURCE = False

# hmc-source / hmc-informed-source sample checkpoints (same pattern for both)
HMC_SOURCE_SAMPLES = os.path.join(OUTPUT_DIR, "samples_hmc_source.npz")
LOAD_HMC_SOURCE_SAMPLES = False  # True → load npz, skip run_inference

HMC_INFORMED_SOURCE_SAMPLES = os.path.join(OUTPUT_DIR, "samples_hmc_informed_source.npz")
LOAD_HMC_INFORMED_SOURCE_SAMPLES = False

# HMC NUTS smoke settings (hmc-source / hmc-informed-source only)
HMC_NUM_WARMUP = 500
HMC_NUM_SAMPLES = 500
HMC_NUM_CHAINS = 2

# Image-plane NUTS: same LOAD_* + npz pattern when comparing hmc / hmc-informed
# HMC_SAMPLES = os.path.join(OUTPUT_DIR, "samples_hmc.npz")
# LOAD_HMC_SAMPLES = False
# HMC_INFORMED_SAMPLES = os.path.join(OUTPUT_DIR, "samples_hmc_informed.npz")
# LOAD_HMC_INFORMED_SAMPLES = False

# Nautilus checkpoint
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = True
NAUTILUS_SIGMA_SPAN = 2.0
```

Companion metadata when using toggles:
- `COMPARISON_LABELS` / `METHOD_COLORS` entry per method
- `active` list print at startup
- `source_by_method[method_key] = samples` drives comparison plots

### Checkpoint/resume by method type

| Method type | Checkpoint mechanism | When to set resume/load |
|-------------|---------------------|-------------------------|
| `nautilus-source` / `nautilus-image` | `cfg["nautilus"]["resume"]` + `filepath` (script-top `NAUTILUS_RESUME` → same key) | `False` after prior/span change; `True` for same-prior re-run (default-on `prior_check` errors on prior mismatch) |
| NUTS full likelihood (`hmc`, `hmc-informed`, `hmc-source`, `hmc-informed-source`) | Fixed `.npz` path + `LOAD_* = True` per method | Load skips `run_inference`; still plots + compares. **Always wire for informed and uninformed** — both are slow |
| Fast (`fisher-source`, `fisher`) | Optional `n_fisher_samples` only | Usually no checkpoint needed |
| `deriv-approx*` | Usually no checkpoint | Re-run is cheaper than MCMC/nautilus |

**Naming convention:** `<METHOD>_SAMPLES` + `LOAD_<METHOD>_SAMPLES` (e.g. `HMC_SOURCE_SAMPLES` / `LOAD_HMC_SOURCE_SAMPLES` for `method="hmc-source"` with `informed=False`).

**Generic load-or-run branch** (same for `hmc-source` and `hmc-informed-source`; swap method name and constants):

```python
def load_samples_npz(path):
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}

if RUN_HMC_SOURCE:
    if LOAD_HMC_SOURCE_SAMPLES:
        samples = load_samples_npz(HMC_SOURCE_SAMPLES)
    else:
        samples, _ = run_inference(
            ctx, mode="GW-only", method="hmc-source",
            cfg={"inference": {"informed": False}, "output": {...}},
        )
        np.savez(HMC_SOURCE_SAMPLES, **samples)
    source_by_method["hmc-source"] = samples

if RUN_HMC_INFORMED_SOURCE:
    if LOAD_HMC_INFORMED_SOURCE_SAMPLES:
        samples = load_samples_npz(HMC_INFORMED_SOURCE_SAMPLES)
    else:
        samples, _ = run_inference(ctx, mode="GW-only", method="hmc-informed-source", cfg=...)
        np.savez(HMC_INFORMED_SOURCE_SAMPLES, **samples)
    source_by_method["hmc-informed-source"] = samples
```

Use manual `np.savez` for predictable paths — pipeline `save_samples_path` auto-appends `_{method}`.

### Per-method inference overrides (NUTS runtime)

Slow full-likelihood NUTS (`hmc`, `hmc-informed`, `hmc-source`, `hmc-informed-source`) should use **per-call cfg overrides** when smoke-testing — do not inherit publication-scale `BASE_CFG` if you want fast iteration:

| Tier | `num_warmup` | `num_samples` | `num_chains` | Use when |
|------|-------------|---------------|--------------|----------|
| Publication | 20000 | 9000 | 20 | Final comparison (`gw_only_nautilus.py` `BASE_CFG`) |
| Smoke | 500 | 500 | 2 | Dev / correctness (`task2_hmc_informed_source_comparison.py`, `gw_only_nautilus.py` HMC constants) |

```python
# hmc-informed-source smoke override (deep-merged over ctx["cfg"])
run_inference(ctx, mode="GW-only", method="hmc-informed-source", cfg={
    "inference": {"num_warmup": 500, "num_samples": 500, "num_chains": 2},
})

# hmc-source (uninformed) — same keys, plus informed=False
run_inference(ctx, mode="GW-only", method="hmc-source", cfg={
    "inference": {"informed": False, "num_warmup": 500, "num_samples": 500, "num_chains": 2},
})
```

`deriv-approx` / `deriv-approx-source` can keep `BASE_CFG` defaults or use their own override block.

### Agent checklist when adding a new method

1. Ask Question 5 (bare vs toggle vs toggle+checkpoint) unless user specified
2. If toggle: add `RUN_*`, label/color, active list entry
3. If **any** NUTS full-likelihood method (`hmc`, `hmc-informed`, `hmc-source`, `hmc-informed-source`): wire **RUN toggle + npz checkpoint** (informed and uninformed alike); default `LOAD_* = False` for first run
4. Wire block; native source-plane methods use `plot_source_posterior` directly (no `to_source_plane_samples`)
5. Point to `examples/scripts/gw_only_nautilus.py` as live reference

## Solver and diagnostics — read before running any `*-source` method

Full detail in **`gwemfish-cfg`**. The parts that change how you run inference:

**One solver for every method.** `cfg["gw"]["solver_params"]["backend"]` (`"auto"` | `"helens"` | `"jaxtronomy"`) governs all five source-plane methods including `nautilus-source`. `cfg["nautilus"]["solver_backend"]` is a deprecated alias that **loses** to it — set both and the nautilus one is ignored. For one method only, use a per-call override (`run_inference` deep-merges its cfg):

```python
run_inference(ctx, mode="GW-only", method="nautilus-source",
              cfg={"gw": {"solver_params": {"backend": "helens"}}})   # this run only
```

**Differentiability does not come from the finder.** Both finders are wrapped in `stop_gradient`; a Newton polish with `jax.lax.custom_root` re-attaches derivatives via the implicit function theorem. Both backends give identical derivatives (verified to `4.5e-13`). So the finder is a reliability choice, not a gradient one.

`n_newton` (default 8) is a **step count, not a switch**, and `0` raises — zero steps means every derivative comes back exactly `0.0` with no error and a NaN covariance. The only on/off switch is `cfg["nautilus"]["polish"]`, and only `nautilus-source` reads it.

**Diagnostics — `cfg["inference"]["diagnostics"]`:** `"warn"` (default, prints and continues), `"raise"` (aborts before sampling), `"off"`. Six checks at truth:

| # | check | fails when |
|---|---|---|
| 1 | images | solver misses/duplicates an image, or positions differ from the simulation |
| 2 | observables | time delays / dL_eff do not reproduce the simulation |
| 3 | source box | prior box reaches past the caustic — **advisory only, never aborts** |
| 4 | parameters | more free parameters than GW observables — **GW-only only**; `EM+GW` / `EM-only` print the tally marked `NA` |
| 5 | fisher cond | `cond > 1e10`, or a Hessian eigenvalue positive above the noise floor (truth is a saddle) |
| 6 | gradient | truth is not the likelihood peak (`\|g0/√\|H_ii\|\| > 0.5`) |

Thresholds are per-key overridable via `cfg["inference"]["diagnostics_thresholds"]`; give only what you change. Prefer raising one threshold over `diagnostics: "off"`, which disables the checks still working.

`"off"` is **unsafe for the `*-source` family**: their Fisher expansion is built *at* truth, so a bad solve there corrupts everything downstream with nothing to signal it.

**Check 3 and NUTS divergences.** Wrong image counts during sampling are rejected by `numpyro.factor("image_count", -1e10)`. Being constant it has zero gradient, so NUTS cannot be pushed back and reports a **divergence** rather than rejecting cleanly. The real control is keeping the source box inside the caustic (`cfg["gw"]["source_box_half_width"]`, default 0.05); check 3 prints the measured margin. Naked-cusp systems sit close to it — catalog 555 has 0.042".

`fisher-source` and `deriv-approx-source` get **no runtime protection**: the factor contributes exactly 0 to `logp0`/`g0`/`H0` at truth, so the Taylor surrogate never sees it. Their only safeguard is the truth-point diagnostic — which is why `"off"` matters more for them, not less.

## Execution

1. Start from `src/gwemfish/cfg_reference.py` (canonical complete config with every key documented inline; symlinked from `scripts/cfg.py`); mode-specific example scripts remain useful for narrower copy-paste patterns. Set base `ctx["cfg"]["priors"]` from mode example (`em_gw_new.py`, `gw_only.py`, `em_nautilus.py`).
2. **Multi-method GW-only comparison:** start from `examples/scripts/gw_only_nautilus.py` — per-method `RUN_*` toggles, optional npz/HDF5 checkpoints (Question 5).
3. Nautilus: precursor run → H₀ priors → `run_inference(..., method="nautilus-source")` (EM-only) or choose source vs image for GW.
4. Else: `samples, truths = run_inference(ctx, mode=..., method=..., cfg={overrides})`.
5. Optional: `output.json_tag`, `save_samples_path`, pipeline JSON.

## Minimal override

```python
samples, truths = run_inference(
    ctx,
    mode="EM+GW",
    method="deriv-approx",
    cfg={
        "output": {"json_tag": "deriv_approx"},
        "inference": {"informed": True},
    },
)
```

## Custom PSF (simulation only)

Set before `setup_em_observation`; no infer-specific cfg:

```python
cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
```

PSF is baked into `ctx["lens_image"]` once — all methods above use it automatically. See `cfg_reference.py` → `PSF_EXAMPLES`, `example_pixel_psf_em_only.py`.

**Supersampling policy:** `supersampling_factor` defaults to 1 and must stay there unless the user agrees to change it. If a system looks undersampled, call `recommend_supersampling(cfg)`, report the suggestion, and wait — never raise it on your own. Confirm an agreed setting with `check_supersampling_convergence(cfg)` rather than trusting the heuristic. Gradient methods (`deriv-approx`, `hmc`, `fisher`) run the whole PSF path under `jax.grad`; supersampled convolution is verified differentiable and correct (autodiff vs finite differences to 4e-6), so no method is off-limits, but the cost is ~factor² profile evaluations. See the `gwemfish-simulate` skill for the full rule.

See [reference.md](reference.md) for full method/mode table and nautilus cfg keys.
