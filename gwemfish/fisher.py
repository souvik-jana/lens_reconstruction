"""
Fisher matrix computation and Taylor expansion approximation.

This module provides functions to compute Fisher matrix approximations
and create approximate log-probability functions for faster inference.
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed


def compute_fisher(model, input_params, keys_to_include, u0, rng_key=None):
    """Compute Fisher matrix approximation (Hessian) and Taylor expansion.
    
    Args:
        model: Numpyro model function
        input_params: Dictionary of input parameter values
        keys_to_include: List of parameter keys to include in Fisher approximation
        u0: Array of parameter values at expansion point (in order of keys_to_include)
        rng_key: Random key for seeding (default: None, uses PRNGKey(1))
    
    Returns:
        approx_logp: Function that computes approximate log-probability
        logp0: Log-probability at expansion point
        g0: Gradient at expansion point
        H0: Hessian at expansion point
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
    
    # Seed the model
    seeded_model = seed(model, rng_key)
    
    # Create logdensity function
    def logdensity_fn(args):
        log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
        return log_density
    
    # Create vectorized logdensity function
    def logdensity_fn_vec(u):
        input_ = input_params.copy()
        for i, key in enumerate(keys_to_include):
            input_[key] = u[i]
        return logdensity_fn(input_)
    
    # Compute Taylor expansion
    grad_b = jax.jacfwd(logdensity_fn_vec)
    H_b = jax.hessian(logdensity_fn_vec)
    
    logp0 = logdensity_fn_vec(u0)
    g0 = grad_b(u0)
    print('Done with gradient')
    H0 = H_b(u0)
    print('Done with Hessian')
    
    # Create approximate log-probability function
    @jax.jit
    def approx_logp(u):
        dx = u - u0
        taylor1 = logp0 + g0 @ dx
        taylor2 = taylor1 + 0.5 * dx @ H0 @ dx
        return taylor2
    
    return approx_logp, logp0, g0, H0

