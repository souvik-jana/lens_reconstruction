"""cfg['nautilus']['jit'] and ['pool'], and the EM+GW source-plane parameter set.

No sampling here -- every test builds a problem and calls the likelihood a handful
of times, so the whole file runs in a couple of minutes. The long checks (full
nested sampling runs, timing) live in nautilus-fix-analysis/.

Run: pytest tests/test_nautilus_jit_pool.py
"""

import copy
import pickle

import numpy as np
import numpyro.distributions as dist
import pytest

from gwemfish import make_default_cfg, prune_gw_images, setup_em_observation, setup_gw_observation
from gwemfish.config import DEFAULT_KWARGS_SOURCE
from gwemfish.nautilus_common import (
    PicklableLikelihood,
    build_nautilus_problem,
    prepare_for_pool,
    sanitize_priors_for_pool,
)

REL_TOL = 1e-10


def base_cfg(mode):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1

    if mode == "GW-only":
        cfg["gw"]["n_images"] = 4
        cfg["gw"]["source_pos"] = (0.02, 0.01)
        cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
        cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    else:
        cfg["gw"]["n_images"] = 2
        cfg["source_plane"]["n_images"] = 2
        cfg["gw"]["source_pos"] = (0.2, 0.01)

    if mode == "EM+GW":
        src = cfg["gw"]["source_pos"]
        kwargs_source = copy.deepcopy(DEFAULT_KWARGS_SOURCE)
        kwargs_source[0]["center_x"] = float(src[0])
        kwargs_source[0]["center_y"] = float(src[1])
        cfg["em"]["kwargs_source"] = kwargs_source

    cfg["nautilus"] = {"n_live": 50, "resume": False, "verbose": False}
    return cfg


def build_ctx(mode):
    ctx = setup_em_observation(cfg=base_cfg(mode))
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    if mode != "GW-only":
        ctx = prune_gw_images(ctx, n_keep=2)
    return ctx


def run_cfg(ctx, mode, **nautilus_overrides):
    truth = ctx["truth_params"]
    priors = {"lens1_ra_0": float(truth["lens1_ra_0"]),
              "lens1_dec_0": float(truth["lens1_dec_0"])}
    cfg = {"priors": priors,
           "nautilus": {**ctx["cfg"]["nautilus"], **nautilus_overrides}}
    if mode != "EM-only":
        src = ctx["cfg"]["gw"]["source_pos"]
        hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
        cfg["gw"] = {"source_plane_bounds": {
            "y0gw": (float(src[0]) - hw, float(src[0]) + hw),
            "y1gw": (float(src[1]) - hw, float(src[1]) + hw)}}
    return cfg


@pytest.fixture(scope="module")
def ctx_gw():
    return build_ctx("GW-only")


@pytest.mark.parametrize("mode", ["GW-only", "EM+GW", "EM-only"])
def test_jit_matches_eager(mode):
    """The compiled likelihood must answer what the eager one answers.

    Relative, not absolute: XLA fuses the same float64 arithmetic in a different
    order, so the last bits differ. 1e-10 relative is orders of magnitude tighter
    than anything nested sampling resolves.
    """
    ctx = build_ctx(mode)
    prior, eager, names_e = build_nautilus_problem(
        ctx, run_cfg(ctx, mode, jit=False), mode, "nautilus-source")
    _, jitted, names_j = build_nautilus_problem(
        ctx, run_cfg(ctx, mode, jit=True), mode, "nautilus-source")
    assert names_e == names_j

    rng = np.random.default_rng(0)
    points = [prior.unit_to_dictionary(u)
              for u in rng.uniform(size=(8, prior.dimensionality()))]
    a = np.array([eager(p) for p in points])
    b = np.array([jitted(p) for p in points])

    accepted = np.isfinite(a) & (a > -1e299)
    assert np.array_equal(a <= -1e299, b <= -1e299), "rejects disagree"
    if accepted.any():
        rel = np.max(np.abs((b[accepted] - a[accepted]) / a[accepted]))
        assert rel < REL_TOL, f"{mode}: max relative difference {rel:.3e}"


def test_em_gw_samples_gw_source_independently():
    """EM+GW must sample y0gw/y1gw, not reuse the EM source centre.

    Pinning them to kwargs_source[0] made nautilus-source fit 23 parameters where
    fisher-source fits 25, and silently dropped cfg['priors']['y0gw'] because
    build_nautilus_prior only iterates over default_dists.
    """
    ctx = build_ctx("EM+GW")
    prior, _, names = build_nautilus_problem(
        ctx, run_cfg(ctx, "EM+GW"), "EM+GW", "nautilus-source")

    assert "y0gw" in names and "y1gw" in names
    assert "source0_center_x" in names and "source0_center_y" in names

    cfg = run_cfg(ctx, "EM+GW")
    cfg["priors"]["y0gw"] = dist.Uniform(0.15, 0.25)
    prior_fixed, _, _ = build_nautilus_problem(ctx, cfg, "EM+GW", "nautilus-source")
    lo, hi = prior_fixed.dists[list(prior_fixed.keys).index("y0gw")].ppf([0.0, 1.0])
    assert (lo, hi) == pytest.approx((0.15, 0.25)), "cfg['priors']['y0gw'] ignored"


def test_jit_refuses_mst(ctx_gw):
    cfg = run_cfg(ctx_gw, "GW-only", jit=True)
    cfg["use_mst"] = True
    with pytest.raises(NotImplementedError, match="use_mst"):
        build_nautilus_problem(ctx_gw, cfg, "GW-only", "nautilus-source")


def test_jit_refuses_legacy_naming(ctx_gw):
    cfg = run_cfg(ctx_gw, "GW-only", jit=True)
    cfg["use_parameter_layout"] = False
    with pytest.raises(NotImplementedError, match="use_parameter_layout"):
        build_nautilus_problem(ctx_gw, cfg, "GW-only", "nautilus-source")


def test_prepare_for_pool_strips_without_mutating(ctx_gw):
    """A later method in the same script still needs ctx['fisher']."""
    ctx = dict(ctx_gw)
    ctx["fisher"] = {"H0": lambda x: x}          # stands in for the jitted closure
    ctx["likelihood"] = {"logp": lambda x: x}

    ctx_clean, _ = prepare_for_pool(ctx, run_cfg(ctx_gw, "GW-only"))

    assert "fisher" not in ctx_clean and "likelihood" not in ctx_clean
    assert "fisher" in ctx and "likelihood" in ctx, "caller's ctx was mutated"
    pickle.dumps(ctx_clean)


def test_sanitize_priors_converts_callables():
    import numpyro

    def prior_fn():
        return numpyro.sample("lens0_gamma", dist.Uniform(1.5, 2.4))

    clean = sanitize_priors_for_pool({"lens0_gamma": prior_fn, "dL": 1000.0})
    pickle.dumps(clean)
    assert clean["dL"] == 1000.0
    assert isinstance(clean["lens0_gamma"], dist.Distribution)


def test_sanitize_priors_names_the_unconvertible_key():
    with pytest.raises(TypeError, match="broken_key"):
        sanitize_priors_for_pool({"broken_key": lambda: 1.0 / 0.0})


def test_picklable_likelihood_round_trip(ctx_gw):
    """What a worker receives must compute what the parent computes."""
    cfg = run_cfg(ctx_gw, "GW-only")
    prior, local, _ = build_nautilus_problem(ctx_gw, cfg, "GW-only", "nautilus-source")
    point = prior.unit_to_dictionary(np.full(prior.dimensionality(), 0.5))

    shipped = pickle.loads(pickle.dumps(
        PicklableLikelihood(ctx_gw, cfg, "GW-only", "nautilus-source")))

    assert shipped(point) == pytest.approx(local(point), rel=REL_TOL)
