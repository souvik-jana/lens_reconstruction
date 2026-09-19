# lens_reconstruction — GWEMFISH

Parametric lens reconstruction for strongly lensed EM+GW systems. The main package is `src/gwemfish/`.

## Environment

- Run from repo root: `uv run python examples/scripts/...`
- Examples use JAX x64 on CPU: set `XLA_FLAGS`, `jax_enable_x64=True`, `jax_platform_name="cpu"` before importing jax (see any example script header).
- Install: `uv sync` (package name `gwemfish` in `pyproject.toml`).

## Canonical pipeline

```
setup_em_observation → setup_gw_observation → run_inference → plot_posterior
→ to_source_plane_samples → plot_source_posterior
```

`cfg` is deep-merged with `gwemfish.simple_pipeline.make_default_cfg()`. See `examples/scripts/SIMPLE_PIPELINE_CONFIG.md`.

## Example script index

Read the closest script before inventing patterns:

| Task | Script |
|------|--------|
| Minimal pipeline | `examples/scripts/example_simple_pipeline.py` |
| Custom / PIXEL PSF | `examples/scripts/example_pixel_psf.py`, `example_pixel_psf_em_only.py` |
| PSF plot + SNR + PAL mirror | `examples/scripts/example_psf_plot_and_pal.py`, `example_pal_mirror.py` |
| EM+GW method comparison | `examples/scripts/em_gw_new.py` |
| EM-only nautilus vs deriv-approx vs fisher | `examples/scripts/em_nautilus.py` |
| GW-only nautilus (source-plane) | `examples/scripts/gw_only_nautilus.py` |
| GW-only nautilus (image-plane) | `examples/scripts/gw_only_nautilus_image.py` |
| Full cfg template (canonical) | `scripts/cfg.py` — every key, every mode/method, `use_parameter_layout` on/off examples |
| Full cfg template (older, narrower) | `examples/scripts/cfg.py` + `SIMPLE_PIPELINE_CONFIG.md` |
| ctx inspection | `examples/notebooks/simple_pipeline_demonstration.ipynb` |

## Inference quick reference

| `mode` | Meaning |
|--------|---------|
| `EM+GW` | Joint EM image + GW likelihood |
| `EM-only` | EM likelihood only |
| `GW-only` | GW likelihood only (`em.enabled: false`) |

| `method` | Notes |
|----------|-------|
| `fisher` | Gaussian from H₀ |
| `deriv-approx` | Taylor model + NUTS; set `inference.informed` |
| `hmc` | Full model NUTS |
| `hmc-informed` | Always informed NUTS |
| `deriv-approx-source` | Source-plane (`y0gw`/`y1gw`) counterpart of `deriv-approx`; not valid for `mode='EM-only'` |
| `hmc-source` | Source-plane counterpart of `hmc`; not valid for `mode='EM-only'` |
| `hmc-informed-source` | Source-plane counterpart of `hmc-informed`; not valid for `mode='EM-only'` |
| `nautilus-source` | Source-plane GW nested sampling |
| `nautilus-image` | Image-plane GW nested sampling (EM-only same as source) |

Set `ctx["cfg"]["priors"]` before `run_inference`. Float = fixed; numpyro `Distribution` = sampled.

`cfg["use_parameter_layout"]` (default `False`): switches flat names from legacy single-lens (`lens_theta_E`, ...) to auto-generated `lens0_*`/`lens1_*`/... (one block per mass profile), `source0_*`, `light0_*`. Works for every method above except it's *required* (not optional) for `EM-only` + `nautilus-source`/`nautilus-image`. See `gwemfish-infer` skill.

## Batch multi-sim studies

Batch YAML, `simulate_batch.py`, and `run_parallel.py` live in **lensing-mock** (sibling repo, default `../lensing-mock`). Use `/gwemfish-batch` agent or skill.

## Do not reimplement

Use Herculens / gwemfish APIs: `MassModel`, `LensImage`, `lensimage_gw`, `lens_setup`, `simple_pipeline` public functions. Opt-in PAL mirror: `simulate_in_pal`, `plot_system_observation_pal`, `save_pal_outputs` (`gwemfish.pal_bridge`). Custom PSF: `cfg["em"]["psf_kwargs"]` with `psf_type="PIXEL"` — see `cfg_reference.py` → `PSF_EXAMPLES`. HCL→PAL mass-profile conversions: `pal_bridge.MASS_PROFILE_BUILDERS`, never re-derive.

## Cursor setup

**In-repo (zero install):** Open this repo in Cursor 2.4+ (Agent mode). Skills in `.cursor/skills/` load automatically — try `/gwemfish-infer`, `/gwemfish-batch`.

**Global `/` commands (optional):** From repo root:

```bash
./scripts/install-cursor-skills.sh
```

Symlinks `.cursor/skills/` (gwemfish, pal, lenstronomy, lensing-mock) and `.cursor/agents/gwemfish*.md` into `~/.cursor/`. Does **not** symlink machine-local skills — copy and edit:

```bash
cp -r .cursor/skills/gwemfish-local.example ~/.cursor/skills/gwemfish-local
cp -r .cursor/skills/pal-local.example ~/.cursor/skills/pal-local
```

**Machine paths (batch + CPU):** One-time per machine:

```bash
cp -r .cursor/skills/gwemfish-local.example ~/.cursor/skills/gwemfish-local
# edit LENS_RECONSTRUCTION_ROOT, LENSING_MOCK_ROOT, XLA_CPU_COUNT, DEFAULT_N_JOBS
```

**Batch YAML repo:** Clone `lensing-mock` as sibling `../lensing-mock` (or set path in `gwemfish-local`).

**Subagents:** `/gwemfish`, `/gwemfish-batch` from `.cursor/agents/` (or after global install).

**Skills in this repo** (real files under `.cursor/skills/`; symlinked globally by install script except `*-local.example`):

| Skill | Role |
|-------|------|
| `gwemfish-simulate`, `gwemfish-infer`, `gwemfish-cfg`, `gwemfish-plot`, `gwemfish-batch`, `gwemfish-pal` | GWEMFISH pipeline |
| `pal-infer`, `pal-plot`, `pal-sim` | PyAutoLens fit/plot/sim |
| `lenstronomy-infer`, `lenstronomy-sim` | lenstronomy cross-checks |
| `lensing-mock` | Batch YAML orchestration (sibling repo scripts) |
| `gwemfish-local.example`, `pal-local.example` | Machine path templates → copy to `~/.cursor/skills/` |
