"""
Fisher matrix computation and Taylor expansion approximation.

This module provides functions to compute Fisher matrix approximations
and create approximate log-probability functions for faster inference.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import seed


def invert_fisher_matrix(FM, regularize=False):
    """Jacobi-scale FM, invert, unwhiten. Clip in scaled space only if regularize."""
    FM_np = np.asarray(FM, dtype=float)
    diag = np.abs(np.diag(FM_np))
    scale = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 1.0)

    FM_scaled = FM_np * scale[:, None] * scale[None, :]
    try:
        cov_scaled = np.linalg.inv(FM_scaled)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Scaled Fisher matrix is singular or near-singular; Jacobi invert failed. "
            "Do not use a pseudoinverse as a posterior covariance (it assigns zero "
            "variance to unconstrained directions). Freeze a parameter via "
            "cfg['priors'] or set cfg['inference']['regularize']=True to clip in "
            "whitened space.",
            UserWarning,
            stacklevel=2,
        )
        raise

    if regularize:
        eig = np.linalg.eigvalsh(cov_scaled)
        if eig.min() <= 0:
            n = cov_scaled.shape[0]
            floor = max(n * np.finfo(float).eps * 10.0, 1e-14)
            warnings.warn(
                f"Scaled Fisher covariance is not positive definite (min eigenvalue "
                f"{eig.min():.3e}); clipping eigenvalues below {floor:.3e} in "
                "whitened space because cfg['inference']['regularize'] is True.",
                UserWarning,
                stacklevel=2,
            )
            vals, vecs = np.linalg.eigh(cov_scaled)
            cov_scaled = (vecs * np.maximum(vals, floor)) @ vecs.T
            cov_scaled = 0.5 * (cov_scaled + cov_scaled.T)

    cov = cov_scaled * scale[:, None] * scale[None, :]
    return jnp.asarray(0.5 * (cov + cov.T))


# def compute_fisher(model, input_params, keys_to_include, u0, rng_key=None):
#     """Compute Fisher matrix approximation (Hessian) and Taylor expansion.
    
#     Args:
#         model: Numpyro model function
#         input_params: Dictionary of input parameter values
#         keys_to_include: List of parameter keys to include in Fisher approximation
#         u0: Array of parameter values at expansion point (in order of keys_to_include)
#         rng_key: Random key for seeding (default: None, uses PRNGKey(1))
    
#     Returns:
#         approx_logp: Function that computes approximate log-probability
#         logp0: Log-probability at expansion point
#         g0: Gradient at expansion point
#         H0: Hessian at expansion point
#     """
#     if rng_key is None:
#         rng_key = jax.random.PRNGKey(1)
    
#     # Seed the model
#     seeded_model = seed(model, rng_key)
    
#     # Create logdensity function
#     def logdensity_fn(args):
#         log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
#         return log_density
    
#     # Create vectorized logdensity function
#     def logdensity_fn_vec(u):
#         input_ = input_params.copy()
#         for i, key in enumerate(keys_to_include):
#             input_[key] = u[i]
#         return logdensity_fn(input_)
    
#     # Compute Taylor expansion
#     grad_b = jax.jacfwd(logdensity_fn_vec)
#     H_b = jax.hessian(logdensity_fn_vec)
#     Flex_b = jax.jacfwd(H_b)
#     Qua_b = jax.jacfwd(Flex_b)

    
#     logp0 = logdensity_fn_vec(u0)
#     g0 = grad_b(u0)
#     print('Done with gradient')
#     H0 = H_b(u0)
#     print('Done with Hessian')
#     # F0 = Flex_b(u0)
#     # print('Done with Flex')
#     # Q0 = Qua_b(u0)
#     # print('Done with Q')
#     # Create approximate log-probability function
#     @jax.jit
#     def approx_logp(u):
#         dx = u - u0
#         taylor1 = logp0 + g0 @ dx
#         taylor2 = taylor1 + 0.5 * dx @ H0 @ dx
#         # taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#         # taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
#         return taylor2
    
#     return approx_logp, logp0, g0, H0#, F0, Q0

def compute_fisher(model, input_params, keys_to_include, u0, rng_key=None, order=2):
    """Compute Fisher matrix approximation (Hessian) and Taylor expansion.
    
    Args:
        model: Numpyro model function
        input_params: Dictionary of input parameter values
        keys_to_include: List of parameter keys to include in Fisher approximation
        u0: Array of parameter values at expansion point (in order of keys_to_include)
        rng_key: Random key for seeding (default: None, uses PRNGKey(1))
        order: Taylor expansion order (default: 2).
               2 = up to Hessian (H0)
               3 = up to 3rd order (F0)
               4 = up to 4th order (Q0)
    
    Returns:
        approx_logp: JIT-compiled function that computes approximate log-probability
        logp0: Log-probability at expansion point
        g0: Gradient at expansion point
        H0: Hessian at expansion point
        F0: 3rd order tensor at expansion point (None if order < 3)
        Q0: 4th order tensor at expansion point (None if order < 4)
    """
    if order not in (2, 3, 4):
        raise ValueError(f"order must be 2, 3, or 4, got {order}")

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
    
    # Always compute up to Hessian (order >= 2)
    grad_fn = jax.jacfwd(logdensity_fn_vec)
    hess_fn = jax.hessian(logdensity_fn_vec)

    logp0 = logdensity_fn_vec(u0)
    g0 = grad_fn(u0)
    print('Done with gradient')
    H0 = hess_fn(u0)
    print('Done with Hessian')

    # Conditionally compute higher-order terms
    F0, Q0 = None, None

    if order >= 3:
        flex_fn = jax.jacfwd(hess_fn)
        F0 = flex_fn(u0)
        print('Done with Flex (3rd order)')

    if order >= 4:
        qua_fn = jax.jacfwd(flex_fn)
        Q0 = qua_fn(u0)
        print('Done with Quartic (4th order)')

    if order == 2:
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            return logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx

    elif order == 3:
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            taylor2 = logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx
            taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
            return taylor3 

    else:  # order == 4
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            taylor2 = logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx
            taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
            taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
            return taylor4

    return approx_logp, logp0, g0, H0, F0, Q0


def compute_fisher_multipoint(model, input_params, keys_to_include, u0, 
    fim_results=None, rng_key=None, order=2, alpha=1.0):
    """Compute three-point Taylor expansion: u0, u0 + alpha*sigma*v, u0 - alpha*sigma*v.

    Walks symmetrically along the most degenerate eigendirection of the
    Hessian at u0. If fim_results is provided, uses the same eigendirection.
    Otherwise, internally computes FIM eigendirections from -H0.
    At each anchor the true log-probability is evaluated. The merged
    approximation uses nearest-anchor selection.

    Args:
        model:            Numpyro model function.
        input_params:     Dict of all parameter values.
        keys_to_include:  Ordered list of parameter names matching u0.
        u0:               Central expansion point, shape (n,).
        fim_results:      Output of compute_fim_eigendirections(-H0, ...).
        rng_key:          Random key (default: None → PRNGKey(1)).
        order:            Taylor order at each point (2, 3, or 4).
        alpha:            Step size in units of sigma along degenerate direction.
                          alpha=1.0 → ±1 posterior sigma from u0.

    Returns:
        approx_logp_merged: JIT-compiled merged log-probability function.
        points:             List of 3 dicts with keys:
                              'u'          : anchor point
                              'logp'       : true log-probability at anchor
                              'approx_logp': Taylor approx function at anchor
        fim_results:        Output of compute_fim_eigendirections(-H0, ...).

    Example:
        >>> approx_logp, points, fim_results = compute_fisher_multipoint(
        ...     model=probmodel.model,
        ...     input_params=input_params,
        ...     keys_to_include=keys_to_include,
        ...     u0=u0,
        ...     order=2,
        ...     alpha=1.0,
        ... )
        >>> fisher_model = ProbModelFisher(
        ...     keys_to_include=keys_to_include,
        ...     approx_logp=approx_logp,
        ...     priors=custom_priors,
        ... )
    """
    from .eigenvec_analysis import compute_fim_eigendirections

    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)

    # --- Seed model for true logp evaluation ---
    seeded_model = numpyro.handlers.seed(model, rng_key)

    def logdensity_fn(args):
        log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
        return log_density

    def logdensity_fn_vec(u):
        input_ = input_params.copy()
        for i, key in enumerate(keys_to_include):
            input_[key] = u[i]
        return logdensity_fn(input_)

    # --- Fisher at u0 ---
    print('Computing Fisher at u0...')
    approx_logp0, logp0, g0, H0, _, _ = compute_fisher(
        model, input_params, keys_to_include, u0,
        rng_key=rng_key, order=order)

    # # --- FIM eigendirections from -H0 (internally computed) ---
    # print('Computing FIM eigendirections from -H0...')
    # fim_results = compute_fim_eigendirections(
    #     FM=-H0,
    #     keys_to_include=keys_to_include,
    #     verbose=True,
    # )

    # v_flat     = fim_results['v_deg']
    # sigma_flat = fim_results['sigma_deg']
    # --- Most degenerate direction ---
    if fim_results is not None:
        # Use externally provided fim_results — same direction as profile scan
        v_flat     = fim_results['v_deg']
        sigma_flat = fim_results['sigma_deg']
        print('Using externally provided fim_results for eigendirection.')
    else:
        # Compute internally from -H0
        print('Computing FIM eigendirections internally from -H0...')
        from .eigenvec_analysis import compute_fim_eigendirections
        fim_results = compute_fim_eigendirections(
            FM=-H0, keys_to_include=keys_to_include, verbose=True)
        v_flat     = fim_results['v_deg']
        sigma_flat = fim_results['sigma_deg']

    print(f'Most degenerate direction sigma: {sigma_flat:.6g}')

    # --- Symmetric anchor points ---
    step  = alpha * float(sigma_flat) * v_flat
    u_pos = u0 + step
    u_neg = u0 - step

    # --- True logp at each anchor ---
    print(f'\nEvaluating true logp at u_pos (u0 + {alpha}σ along v_deg)...')
    logp_pos = logdensity_fn_vec(u_pos)

    print(f'Evaluating true logp at u_neg (u0 - {alpha}σ along v_deg)...')
    logp_neg = logdensity_fn_vec(u_neg)

    # --- Fisher at each anchor ---
    print('\nComputing Fisher at u_pos...')
    approx_logp_pos, _, _, _, _, _ = compute_fisher(
        model, input_params, keys_to_include, u_pos,
        rng_key=rng_key, order=order)

    print('\nComputing Fisher at u_neg...')
    approx_logp_neg, _, _, _, _, _ = compute_fisher(
        model, input_params, keys_to_include, u_neg,
        rng_key=rng_key, order=order)

    points = [
        {'u': u0,    'logp': logp0,    'approx_logp': approx_logp0},
        {'u': u_pos, 'logp': logp_pos, 'approx_logp': approx_logp_pos},
        {'u': u_neg, 'logp': logp_neg, 'approx_logp': approx_logp_neg},
    ]

    print(f'\nAnchor logp values:')
    print(f'  u0    : {float(logp0):.4f}')
    print(f'  u_pos : {float(logp_pos):.4f}')
    print(f'  u_neg : {float(logp_neg):.4f}')

    approx_fns = [p['approx_logp'] for p in points]
    # # Debug check — Sanity check at anchors:
    # print(f'\nSanity check at anchors:')
    # print(f'  approx_logp0(u0)       = {float(approx_logp0(u0)):.4f}  '
    #     f'true logp0 = {float(logp0):.4f}')
    # print(f'  approx_logp_pos(u_pos) = {float(approx_logp_pos(u_pos)):.4f}  '
    #     f'true logp_pos = {float(logp_pos):.4f}')
    # print(f'  approx_logp_neg(u_neg) = {float(approx_logp_neg(u_neg)):.4f}  '
    #     f'true logp_neg = {float(logp_neg):.4f}')
    # --- Nearest anchor selection ---
    u0_arr    = jnp.asarray(u0)
    u_pos_arr = jnp.asarray(u_pos)
    u_neg_arr = jnp.asarray(u_neg)

    @jax.jit
    def approx_logp_merged(u):
        d0    = jnp.sqrt(jnp.sum((u - u0_arr)    ** 2))
        d_pos = jnp.sqrt(jnp.sum((u - u_pos_arr) ** 2))
        d_neg = jnp.sqrt(jnp.sum((u - u_neg_arr) ** 2))
        idx_nearest = jnp.argmin(jnp.array([d0, d_pos, d_neg]))
        return jax.lax.switch(
            idx_nearest,
            [approx_fns[0], approx_fns[1], approx_fns[2]],
            u,
        )

    return approx_logp_merged, points, fim_results

