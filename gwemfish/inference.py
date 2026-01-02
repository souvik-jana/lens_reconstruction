"""
MCMC inference functions.

This module provides functions to run MCMC inference using NUTS sampler.
"""

import jax
import jax.random as random
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary


def run_mcmc(model, num_warmup=6500, num_samples=14500, max_tree_depth=10, 
             dense_mass=True, num_chains=2, rng_key=None):
    """Run MCMC inference with NUTS sampler.
    
    Args:
        model: Numpyro model function
        num_warmup: Number of warmup steps (default: 6500)
        num_samples: Number of samples to draw (default: 14500)
        max_tree_depth: Maximum tree depth for NUTS (default: 10)
        dense_mass: Whether to use dense mass matrix (default: True)
        num_chains: Number of chains (default: 2)
        rng_key: Random key for reproducibility (default: None, uses PRNGKey(2))
    
    Returns:
        samples: Dictionary of samples
        summary_dict: Summary statistics for each parameter
        extra_fields: Extra fields from MCMC (e.g., divergences)
        mcmc: MCMC object
    """
    if rng_key is None:
        rng_key = random.PRNGKey(2)
    
    kernel = NUTS(model, max_tree_depth=max_tree_depth, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method='parallel',  # Run chains in parallel
        progress_bar=True,
    )
    mcmc.run(rng_key)
    summary_dict = summary(mcmc.get_samples(), group_by_chain=False)
    
    # Print the largest r_hat for each variable
    for k, v in summary_dict.items():
        spaces = " " * max(12 - len(k), 0)
        print('\n')
        print("[{}] {} \t max r_hat: {:.4f}".format(k, spaces, jnp.max(v["r_hat"])))
    
    return mcmc.get_samples(), summary_dict, mcmc.get_extra_fields(), mcmc

