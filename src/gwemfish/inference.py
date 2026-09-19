"""
MCMC inference functions.

This module provides functions to run MCMC inference using NUTS sampler.
"""

import jax
import jax.random as random
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value
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


def run_mcmc_informed(model, u0, keys_to_include, H0,
                      num_warmup=2000, num_samples=5000, max_tree_depth=10,
                      num_chains=2, scale=1.0, perturb_scale=0.1, rng_key=None,
                      regularize=False):
    """Run MCMC informed by Fisher Hessian.

    Computes cov via Jacobi invert of -H0, initialises chains near u0 with
    small perturbations, and sets the NUTS mass matrix from the covariance.

    Args:
        model:            Numpyro model function.
        u0:               Parameter values at expansion point, shape (n,).
        keys_to_include:  Ordered list of parameter names matching u0.
        H0:               Hessian at u0, shape (n, n). From compute_fisher.
        num_warmup:       Warmup steps (default 2000).
        num_samples:      Number of samples (default 5000).
        max_tree_depth:   NUTS max tree depth (default 10).
        num_chains:       Number of chains (default 2).
        scale:            Scales the covariance (default 1.0).
                          > 1 broader proposals, < 1 narrower.
        perturb_scale:    Fraction of sigma to perturb init per chain
                          (default 0.1 = 10% of posterior sigma).
                          Ensures chains differ for reliable r-hat.
        rng_key:          Random key (default: None → PRNGKey(2)).
        regularize:       If True, clip non-positive eigenvalues of the Jacobi
                          inverse in whitened space (see invert_fisher_matrix).
                          Enable via cfg['inference']['regularize']=True (default False).

    Returns:
        samples, summary_dict, extra_fields, mcmc
    """
    if rng_key is None:
        rng_key = random.PRNGKey(2)

    from .fisher import invert_fisher_matrix

    cov = invert_fisher_matrix(-H0, regularize=regularize) * (scale ** 2)
    sigmas = jnp.sqrt(jnp.diag(cov))

    rng_key, subkey = random.split(rng_key)
    noise   = random.normal(subkey, shape=(num_chains, len(u0))) * sigmas * perturb_scale
    u_init  = jnp.asarray(u0)[None, :] + noise
    # init_values = {k: u_init[:, j] for j, k in enumerate(keys_to_include)}

    print('Hessian-informed MCMC:')
    print(f'  u0:            {u0}')
    print(f'  Param sigmas:  {sigmas}')
    print(f'  Scale:         {scale}')
    print(f'  Perturb scale: {perturb_scale}')
    print('cov shape:', cov.shape)   # should be (23, 23) — n_params × n_params


    # Temporarily use flat init (same point for all chains) to test
    init_values = {k: jnp.asarray(float(u0[i]))
                   for i, k in enumerate(keys_to_include)}
    kernel = NUTS(
        model,
        max_tree_depth=max_tree_depth,
        dense_mass=True,
        init_strategy=init_to_value(values=init_values),
        inverse_mass_matrix=cov,
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method='parallel', progress_bar=True)
    mcmc.run(rng_key)

    summary_dict = summary(mcmc.get_samples(), group_by_chain=False)
    for k, v in summary_dict.items():
        spaces = " " * max(12 - len(k), 0)
        print('\n')
        print("[{}] {} \t max r_hat: {:.4f}".format(k, spaces, jnp.max(v["r_hat"])))

    return mcmc.get_samples(), summary_dict, mcmc.get_extra_fields(), mcmc