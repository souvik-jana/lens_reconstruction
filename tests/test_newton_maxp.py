"""Newton–Raphson maxP polish used before Fisher expansion.

Evaluates g/H at the given start, then jumps u ← u - H^{-1} g (≤ max_jumps).
Quadratic log-density → one jump lands on the mode; a second jump is a no-op.

Run: pytest tests/test_newton_maxp.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gwemfish.fisher import newton_raphson_maxp


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
        logp, u0, max_jumps=2, grad_tol=1e-12, verbose=False
    )

    assert info["n_jumps"] == 1  # exact after first jump; 2nd skipped by tol
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


def test_already_at_mode_zero_jumps():
    mean = jnp.array([0.1, 0.2])
    logp = _quadratic_logp(mean, jnp.eye(2) * 5.0)
    u_map, info = newton_raphson_maxp(
        logp, mean, max_jumps=2, grad_tol=1e-10, verbose=False
    )
    assert info["n_jumps"] == 0
    np.testing.assert_allclose(u_map, mean, atol=1e-12)


def test_rejects_negative_max_jumps():
    logp = _quadratic_logp(jnp.zeros(2), jnp.eye(2))
    with pytest.raises(ValueError, match="max_jumps"):
        newton_raphson_maxp(logp, jnp.ones(2), max_jumps=-1, verbose=False)
