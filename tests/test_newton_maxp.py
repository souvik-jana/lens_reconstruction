"""Newton–Raphson maxP polish used before Fisher expansion.

NSD-modified Newton + optional Armijo line search; default keeps jumping until
``||g|| < grad_tol``. Quadratic log-density → one jump lands on the mode.

Run: pytest tests/test_newton_maxp.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gwemfish.fisher import format_map_params, newton_raphson_maxp


def _quadratic_logp(mean, precision):
    """log p(u) = -0.5 (u-m)^T P (u-m)  (up to a constant)."""
    mean = jnp.asarray(mean, dtype=jnp.float64)
    precision = jnp.asarray(precision, dtype=jnp.float64)

    def logp(u):
        du = u - mean
        return -0.5 * du @ precision @ du

    return logp


def test_quadratic_lands_in_one_jump():
    mean = jnp.array([1.5, -0.7, 0.25])
    # SPD precision → negative-definite Hessian = -P
    prec = jnp.array(
        [
            [4.0, 0.5, 0.0],
            [0.5, 3.0, 0.2],
            [0.0, 0.2, 2.0],
        ]
    )
    logp = _quadratic_logp(mean, prec)
    u0 = jnp.array([0.0, 0.0, 0.0])

    u_map, info = newton_raphson_maxp(
        logp, u0, max_jumps=None, grad_tol=1e-12, verbose=False
    )

    assert info["converged"]
    assert info["n_jumps"] == 1  # exact after first jump
    np.testing.assert_allclose(u_map, mean, atol=1e-10)
    assert info["grad_norm"] < 1e-10


def test_max_jumps_cap():
    """Non-quadratic bowl: still improves, never exceeds max_jumps."""
    true = jnp.array([0.3, -0.4])

    def logp(u):
        # Mild quartic perturbation around a quadratic bowl
        du = u - true
        return -0.5 * jnp.dot(du, du) - 0.05 * jnp.sum(du ** 4)

    u0 = jnp.array([2.0, 2.0])
    u_map, info = newton_raphson_maxp(
        logp, u0, max_jumps=2, grad_tol=0.0, verbose=False
    )

    assert info["n_jumps"] == 2
    # Closer to the mode than the start
    assert float(jnp.linalg.norm(u_map - true)) < float(jnp.linalg.norm(u0 - true))
    # Gradient shrunk
    g0 = jax.grad(logp)(u0)
    assert info["grad_norm"] < float(jnp.linalg.norm(g0))


def test_until_map_on_quartic():
    """With max_jumps=None, keep jumping until |g| < tol."""
    true = jnp.array([0.3, -0.4])

    def logp(u):
        du = u - true
        return -0.5 * jnp.dot(du, du) - 0.05 * jnp.sum(du ** 4)

    u0 = jnp.array([2.0, 2.0])
    u_map, info = newton_raphson_maxp(
        logp, u0, max_jumps=None, grad_tol=1e-10, verbose=False
    )
    assert info["converged"]
    assert info["grad_norm"] < 1e-10
    np.testing.assert_allclose(u_map, true, atol=1e-6)


def test_already_at_mode_zero_jumps():
    mean = jnp.array([0.1, 0.2])
    logp = _quadratic_logp(mean, jnp.eye(2) * 5.0)
    u_map, info = newton_raphson_maxp(
        logp, mean, max_jumps=None, grad_tol=1e-10, verbose=False
    )
    assert info["n_jumps"] == 0
    assert info["converged"]
    np.testing.assert_allclose(u_map, mean, atol=1e-12)


def test_rejects_negative_max_jumps():
    logp = _quadratic_logp(jnp.zeros(2), jnp.eye(2))
    with pytest.raises(ValueError, match="max_jumps"):
        newton_raphson_maxp(logp, jnp.ones(2), max_jumps=-1, verbose=False)


def test_format_map_params_table():
    text = format_map_params(
        ["a", "b"],
        [1.5, -0.25],
        [1.0, 0.0],
        grad_norm=1e-9,
        n_jumps=3,
        logp=12.0,
    )
    assert "MAP parameters" in text
    assert "a" in text and "b" in text
    assert "given" in text and "delta" in text


def test_default_floors_pin_sigma():
    from gwemfish.fisher import default_param_floors

    keys = ["theta_E", "noise_sigma_bkg", "amp", "y1gw", "lens0_e1"]
    u0 = np.array([2.0, 0.01, 0.5, 1e-6, 0.05])
    floors = default_param_floors(keys, u0, rel=1e-3)
    assert floors[0] == pytest.approx(2e-3)  # theta_E positive
    assert floors[1] == pytest.approx(0.01)  # sigma pinned at given
    assert floors[2] == pytest.approx(5e-4)  # amp positive
    assert floors[3] == -np.inf  # source coord: not floored
    assert floors[4] == -np.inf  # ellipticity: not floored
