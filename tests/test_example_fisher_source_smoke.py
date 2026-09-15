"""Smoke: example_multimessenger_fisher.py fisher-source path still works.

Mirrors the example script's setup + ``method='fisher-source'`` call (noiseless
EM image, amp=0.5, fixed lens1_ra_0/dec_0) with no example-side cfg changes for
newton_maxp. Default ``inference.newton_maxp.enabled=True`` must not break it.

Run: pytest tests/test_example_fisher_source_smoke.py -v
"""

import numpy as np
import pytest

import gwemfish

gwemfish.setup_jax(verbose=False)


def _example_ctx():
    """Same observation setup as example_multimessenger_fisher.py (no plots)."""
    cfg = gwemfish.make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["em"]["kwargs_source"][0]["amp"] = 0.5

    ctx = gwemfish.setup_em_observation(cfg=cfg)
    em = ctx["cfg"]["em"]
    ctx["em_obs"]["data"] = ctx["lens_image"].model(
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=em["kwargs_source"],
        kwargs_lens_light=em["kwargs_lens_light"],
    )
    return gwemfish.setup_gw_observation(ctx, cfg=cfg)


@pytest.fixture(scope="module")
def example_fisher_result():
    ctx = _example_ctx()
    # Exact infer cfg the example passes for fisher-source (no newton overrides).
    samples, truths = gwemfish.run_inference(
        ctx,
        mode="EM+GW",
        method="fisher-source",
        cfg={
            "priors": {
                "lens1_ra_0": float(0.0),
                "lens1_dec_0": float(0.0),
            },
            "output": {
                "output_dir": None,  # skip writing results in CI/smoke
                "json_tag": "EM+GW-fisher-source-smoke",
            },
            "inference": {
                "n_fisher_samples": 200,  # speed only; example default is larger
                "diagnostics": "warn",
            },
        },
    )
    return ctx, samples, truths


def test_example_fisher_source_returns_samples(example_fisher_result):
    ctx, samples, truths = example_fisher_result
    assert isinstance(samples, dict) and len(samples) > 0
    assert isinstance(truths, dict) and len(truths) > 0
    for k, arr in samples.items():
        a = np.asarray(arr)
        assert a.ndim == 1 and a.size > 0
        assert np.all(np.isfinite(a))
        assert k in truths


def test_example_fisher_source_newton_maxp_default_on(example_fisher_result):
    """Default newton_maxp runs (≤2 jumps); truths stay at given values."""
    ctx, samples, truths = example_fisher_result
    info = (ctx.get("likelihood") or {}).get("newton_maxp")
    assert info is not None, "newton_maxp should run with default enabled=True"
    assert 0 <= info["n_jumps"] <= 2

    u0_given = np.asarray(ctx["likelihood"]["u0_given"])
    keys = ctx["likelihood"]["keys_to_include"]
    # truths_dict must remain the given/example values, not the jumped MAP
    for i, k in enumerate(keys):
        if k in truths:
            assert truths[k] == pytest.approx(float(u0_given[i]), rel=0, abs=1e-10)
