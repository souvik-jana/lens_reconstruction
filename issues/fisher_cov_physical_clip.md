# Fisher covariance: physical-space eigenvalue clip blows up widths

**Status:** open (fix later)  
**Where:** `src/gwemfish/simple_pipeline.py` → `_fisher_covariance`

## Issue

Fisher inversion uses Jacobi whitening (`scale_i = 1/√|FM_ii|`), inverts in scaled space, then unwhitens to physical `cov`. If any physical eigenvalue is `≤ 0`, the code clips with

```text
floor = max(λ_max, 1) × 1e-12
```

in **physical** space. That floor is huge relative to tiny round-off negative eigs (often `~1e-10`–`1e-16` after unwhitening), so it injects ~0.2σ of fake variance along those directions. Symptom: EM+GW `fisher-source` with `σ_dL_eff=0.5` looked far wider than EM-only / unclipped Fisher (e.g. `lens0_q`, `lens0_phi` σ ≈ 0.23 instead of ~0.005–0.01). Easy to misread as “GW dilutes EM.”

## Observation

- Without clipping (sample from unwhitened `cov`, or sample in scaled space), widths match √`cov_ii` and look correct.
- On the repro case, **scaled** `cov_s` was already PD (`λ_min ≈ 0.2`); the tiny negative eig appears only after unwhitening. So the clip often fires on a numerical artifact, not a truly singular Fisher.
- Jacobi scale / unwhiten itself is fine; the bug is the physical-space floor.

## Proposed fix

1. Clip (if needed) on **scaled** `cov_s` with a tiny floor, e.g. `max(n·ε·10, 1e-14)`, not `λ_max×1e-12` in physical units.
2. Prefer sampling in scaled space: `z ~ N(0, cov_s)`, then `δ = scale ⊙ z` (avoids needing physical SPD after round-off).
3. Keep symmetry: `cov = 0.5*(cov + cov.T)`.

Do **not** apply the current physical clip as a “safety” path — it is worse than leaving the tiny negative eig alone for sampling when scaled space is PD.

## Quick check

| path | `lens0_q` σ (σ_dL_eff=0.5) |
|------|----------------------------|
| physical clip (current) | ~0.23 |
| no clip / proposed | ~0.0055 |
