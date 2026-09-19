# Source-plane `y0gw`/`y1gw` truths missing from nautilus

**Status:** open  
**Where:** `simple_pipeline.py` → `_finish_nautilus_run` vs `_build_inference_probmodel_source_plane`

## Issue

`setup_gw_observation` never puts `y0gw`/`y1gw` in `ctx["truth_params"]` (only in `cfg["gw"]["source_pos"]`).

- **NUTS/Fisher `*-source`:** backfill locally → returned `truths` include them. `ctx` unchanged.
- **`nautilus-source`:** no backfill → returned `truths` omit them. Merge by hand when plotting:

```python
src = ctx["cfg"]["gw"]["source_pos"]
truths["y0gw"], truths["y1gw"] = float(src[0]), float(src[1])
```

## Fix

Backfill once in `setup_gw_observation` (or in `_finish_nautilus_run`) from `source_pos`.
