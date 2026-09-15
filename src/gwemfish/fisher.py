"""
Fisher matrix computation and Taylor expansion approximation.

This module provides functions to compute Fisher matrix approximations
and create approximate log-probability functions for faster inference.
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed


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

def _newton_ascent_direction(g, H, eig_floor=1e-12):
    """Newton ascent direction for maximizing ``L``: solve ``H_nsd d = -g``.

    At a local max the Hessian is negative definite and this reduces to
    ``d = -H^{-1} g``.  Positive eigenvalues (saddle / indefinite ``H``) are
    flipped so the quadratic model is concave and the step seeks a max rather
    than a saddle. Near-zero eigenvalues are floored away from zero.
    """
    evals, evecs = jnp.linalg.eigh(H)
    # Want negative-definite curvature for maximization.
    evals_nsd = jnp.where(evals < 0.0, evals, -jnp.maximum(jnp.abs(evals), eig_floor))
    evals_nsd = jnp.where(
        jnp.abs(evals_nsd) < eig_floor, -eig_floor, evals_nsd
    )
    # H_nsd = Q diag(λ) Q^T ; solve H_nsd d = -g
    # d = Q diag(1/λ) Q^T (-g)
    d = evecs @ ((evecs.T @ (-g)) / evals_nsd)
    return d


def _project_vector(v, u, floors, eps=1e-14):
    """Zero components that would push a lower-bounded coordinate further down."""
    if floors is None:
        return v
    blocked = (u <= floors + eps) & (v < 0.0)
    return jnp.where(blocked, 0.0, v)


def _armijo_backtrack(
    logdensity_fn_vec,
    u,
    d,
    g,
    *,
    floors=None,
    c=1e-4,
    shrink=0.5,
    alpha0=1.0,
    min_alpha=1e-16,
):
    """Armijo line search along ascent direction ``d`` (maximize log-density)."""
    import numpy as np

    d = _project_vector(d, u, floors)
    logp0 = float(logdensity_fn_vec(u))
    gTd = float(jnp.dot(g, d))
    if not np.isfinite(gTd) or gTd <= 0.0:
        # Direction not ascent → fall back to projected steepest ascent.
        d = _project_vector(g, u, floors)
        gTd = float(jnp.dot(g, d))
        if not np.isfinite(gTd) or gTd <= 0.0:
            return u, 0.0, d, False

    d_norm = float(jnp.linalg.norm(d))
    if not np.isfinite(d_norm) or d_norm < 1e-16:
        return u, 0.0, d, False

    alpha = float(alpha0)
    while alpha >= float(min_alpha):
        u_try = u + alpha * d
        if floors is not None:
            u_try = jnp.maximum(u_try, floors)
        logp_try = float(logdensity_fn_vec(u_try))
        if np.isfinite(logp_try) and logp_try >= logp0 + max(
            c * alpha * gTd, 1e-14 * (1.0 + abs(logp0))
        ):
            return u_try, alpha, d, True
        alpha *= float(shrink)
    return u, 0.0, d, False


def _projected_grad_metrics(g, H, u, floors, eps=1e-14):
    """Projected raw ||g|| and max_i |g_i|/sqrt(|H_ii|) (diagnostic scale)."""
    import numpy as np

    g_np = np.asarray(g, dtype=np.float64).ravel()
    H_np = np.asarray(H, dtype=np.float64)
    u_np = np.asarray(u, dtype=np.float64).ravel()
    g_proj = g_np.copy()
    if floors is not None:
        fl = np.asarray(floors, dtype=np.float64).ravel()
        active = (u_np <= fl + eps) & (g_np < 0.0)
        g_proj[active] = 0.0
    diag = np.abs(np.diag(H_np))
    scaled = np.abs(g_proj) / np.sqrt(np.maximum(diag, 1e-30))
    return float(np.linalg.norm(g_proj)), float(np.max(scaled) if scaled.size else 0.0)


def default_param_floors(keys, u0, *, rel=1e-3, abs_floor=1e-12):
    """Lower bounds for positive-at-start parameters (stops σ→0 ridges).

    - Keys containing ``sigma``: floor at the given value (pin noise scale; on
      noiseless data free σ→0 is an unbounded MAP ridge).
    - Other positive starts: floor ``max(abs_floor, rel*|u0_i|)``.
    - Non-positive starts: unbounded below (``-inf``).
    """
    import numpy as np

    u0 = np.asarray(u0, dtype=np.float64).ravel()
    keys = list(keys)
    if len(keys) != u0.size:
        raise ValueError(f"keys length {len(keys)} != u0 size {u0.size}")
    floors = np.full(u0.shape, -np.inf, dtype=np.float64)
    for i, (k, v) in enumerate(zip(keys, u0)):
        if v > 0.0:
            if "sigma" in str(k).lower():
                floors[i] = float(v)
            else:
                floors[i] = max(float(abs_floor), float(rel) * float(v))
    return floors


def format_map_params(keys, u_map, u0_given=None, *, grad_norm=None, n_jumps=None, logp=None):
    """Readable multi-line table of MAP (and optional given) parameter values."""
    import numpy as np

    u_map = np.asarray(u_map, dtype=np.float64).ravel()
    keys = list(keys)
    if len(keys) != u_map.size:
        raise ValueError(
            f"keys length {len(keys)} != u_map size {u_map.size}"
        )
    header_bits = ["[newton_maxp] MAP parameters"]
    if n_jumps is not None:
        header_bits.append(f"n_jumps={n_jumps}")
    if grad_norm is not None:
        header_bits.append(f"scaled|g|={float(grad_norm):.3e}")
    if logp is not None:
        header_bits.append(f"logp={float(logp):.6g}")
    lines = [", ".join(header_bits)]
    if u0_given is None:
        lines.append(f"{'param':24s} {'MAP':>16s}")
        for k, v in zip(keys, u_map):
            lines.append(f"{k:24s} {float(v):16.8g}")
    else:
        u0 = np.asarray(u0_given, dtype=np.float64).ravel()
        lines.append(
            f"{'param':24s} {'given':>16s} {'MAP':>16s} {'delta':>16s}"
        )
        for k, v0, vm in zip(keys, u0, u_map):
            lines.append(
                f"{k:24s} {float(v0):16.8g} {float(vm):16.8g} "
                f"{float(vm - v0):16.8g}"
            )
    return "\n".join(lines)


def newton_raphson_maxp(
    logdensity_fn_vec,
    u0,
    *,
    max_jumps=None,
    grad_tol=5e-2,
    verbose=True,
    line_search=True,
    floors=None,
):
    """Newton–Raphson ascent of ``logdensity_fn_vec`` toward its maxP / MAP.

    Evaluates the gradient ``g`` and Hessian ``H`` of the log-density at the
    current point, builds a negative-definite Newton ascent direction
    (flipping positive Hessian eigenvalues so saddles do not attract the
    iterate), optionally takes an Armijo backtracking step along that
    direction, and repeats until the projected *scaled* gradient is small:
    ``max_i |g_i|/sqrt(|H_ii|) < grad_tol`` (same scale as Fisher diagnostics).

    Optional ``floors`` clamp parameters from below after each step (and inside
    the line search). Keys containing ``sigma`` are pinned at the given value
    by :func:`default_param_floors` (blocks the noiseless σ→0 ridge).

    Tracks the best iterate by scaled gradient and restores it on divergence.

    Args:
        logdensity_fn_vec: Callable ``u -> log p(u)`` (same shape as ``u0``).
        u0: Starting parameter vector (given / truth values).
        max_jumps: Maximum Newton steps. ``None`` (default) means keep jumping
            until scaled |g| < grad_tol (hard safety cap 10_000). ``0`` skips.
        grad_tol: Convergence threshold on max scaled projected |g| (default 5e-2).
        verbose: Print per-jump diagnostics.
        line_search: If True (default), Armijo backtrack on the Newton step.
            If False, take the full undamped step (quadratic toy problems).
        floors: Optional lower bounds, same shape as ``u0``. ``None`` = unbounded.

    Returns:
        u_map: Parameter vector after the jumps.
        info: Dict with ``n_jumps``, ``grad_norm`` (raw projected ||g||),
            ``grad_norm_scaled``, ``converged``, ``history``, final ``g`` / ``H``,
            ``u0_given``, ``u_map``, ``logp``.
    """
    import numpy as np

    hard_cap = 10_000
    if max_jumps is None:
        jump_limit = hard_cap
    else:
        if max_jumps < 0:
            raise ValueError(f"max_jumps must be >= 0 or None, got {max_jumps}")
        jump_limit = int(max_jumps)

    u = jnp.asarray(u0, dtype=jnp.float64)
    floors_j = None if floors is None else jnp.asarray(floors, dtype=jnp.float64)
    if floors_j is not None:
        u = jnp.maximum(u, floors_j)

    grad_fn = jax.jacfwd(logdensity_fn_vec)
    hess_fn = jax.hessian(logdensity_fn_vec)

    history = []
    n_jumps = 0
    converged = False
    g = grad_fn(u)
    H = hess_fn(u)

    def _logp(uu):
        return float(logdensity_fn_vec(uu))

    def _metrics(gg, HH, uu):
        return _projected_grad_metrics(gg, HH, uu, floors_j)

    best_u = np.asarray(u)
    best_g = np.asarray(g)
    best_H = np.asarray(H)
    best_gnorm, best_scaled = _metrics(g, H, u)
    best_logp = _logp(u)
    scaled0 = best_scaled

    for j in range(jump_limit):
        gnorm, scaled = _metrics(g, H, u)
        if verbose:
            print(
                f"[newton_maxp] at start of jump {j + 1}: "
                f"|g|_proj={gnorm:.3e}  max|g|/sqrt|H|={scaled:.3e}"
            )
        if not np.isfinite(gnorm) or not np.isfinite(scaled):
            if verbose:
                print("[newton_maxp] |g| non-finite; restoring best iterate")
            u = jnp.asarray(best_u)
            g = jnp.asarray(best_g)
            H = jnp.asarray(best_H)
            history.append(
                {
                    "u": np.asarray(u),
                    "g": np.asarray(g),
                    "grad_norm": best_gnorm,
                    "grad_norm_scaled": best_scaled,
                    "step_norm": 0.0,
                    "alpha": 0.0,
                    "skipped": True,
                    "restored_best": True,
                }
            )
            break
        if scaled < float(grad_tol):
            converged = True
            if verbose:
                print(
                    f"[newton_maxp] scaled |g| < grad_tol ({grad_tol:g}); "
                    f"MAP reached before jump {j + 1}"
                )
            history.append(
                {
                    "u": np.asarray(u),
                    "g": np.asarray(g),
                    "grad_norm": gnorm,
                    "grad_norm_scaled": scaled,
                    "step_norm": 0.0,
                    "alpha": 0.0,
                    "skipped": True,
                }
            )
            break

        if scaled0 > 0 and scaled > max(1e4, 1e3 * scaled0):
            if verbose:
                print(
                    f"[newton_maxp] scaled |g| blew up ({scaled:.3e}); "
                    "restoring best iterate"
                )
            u = jnp.asarray(best_u)
            g = jnp.asarray(best_g)
            H = jnp.asarray(best_H)
            history.append(
                {
                    "u": np.asarray(u),
                    "g": np.asarray(g),
                    "grad_norm": best_gnorm,
                    "grad_norm_scaled": best_scaled,
                    "step_norm": 0.0,
                    "alpha": 0.0,
                    "skipped": True,
                    "restored_best": True,
                }
            )
            break

        d = _newton_ascent_direction(g, H)
        d = _project_vector(d, u, floors_j)
        if line_search:
            u_new, alpha, d_used, ok = _armijo_backtrack(
                logdensity_fn_vec, u, d, g, floors=floors_j
            )
            if not ok:
                d_ga = _project_vector(g, u, floors_j)
                u_new, alpha, d_used, ok = _armijo_backtrack(
                    logdensity_fn_vec, u, d_ga, g, floors=floors_j
                )
            if not ok:
                if verbose:
                    print(
                        f"[newton_maxp] line search failed at jump {j + 1}; "
                        "stopping"
                    )
                history.append(
                    {
                        "u": np.asarray(u),
                        "g": np.asarray(g),
                        "grad_norm": gnorm,
                        "grad_norm_scaled": scaled,
                        "step_norm": float(jnp.linalg.norm(d)),
                        "alpha": 0.0,
                        "skipped": True,
                    }
                )
                break
            step_norm = float(jnp.linalg.norm(u_new - u))
            alpha_f = float(alpha)
        else:
            alpha_f = 1.0
            u_new = u + d
            if floors_j is not None:
                u_new = jnp.maximum(u_new, floors_j)
            step_norm = float(jnp.linalg.norm(u_new - u))

        u_scale = 1.0 + float(jnp.linalg.norm(u))
        if step_norm < 1e-12 * u_scale or (line_search and alpha_f < 1e-10):
            if verbose:
                print(
                    f"[newton_maxp] ineffective step "
                    f"(|du|={step_norm:.3e}, alpha={alpha_f:.3g}); stopping"
                )
            history.append(
                {
                    "u": np.asarray(u),
                    "g": np.asarray(g),
                    "grad_norm": gnorm,
                    "grad_norm_scaled": scaled,
                    "step_norm": step_norm,
                    "alpha": alpha_f,
                    "skipped": True,
                    "stalled": True,
                }
            )
            break

        u = u_new
        n_jumps += 1
        history.append(
            {
                "u": np.asarray(u),
                "g": np.asarray(g),
                "grad_norm": gnorm,
                "grad_norm_scaled": scaled,
                "step_norm": step_norm,
                "alpha": alpha_f,
                "skipped": False,
            }
        )
        if verbose:
            print(
                f"[newton_maxp] jump {n_jumps}: |du|={step_norm:.3e} "
                f"alpha={alpha_f:.3g} (from scaled|g|={scaled:.3e})"
            )

        g = grad_fn(u)
        H = hess_fn(u)
        gnorm_now, scaled_now = _metrics(g, H, u)
        logp_now = _logp(u)
        if np.isfinite(scaled_now) and scaled_now < best_scaled:
            best_u = np.asarray(u)
            best_g = np.asarray(g)
            best_H = np.asarray(H)
            best_gnorm = gnorm_now
            best_scaled = scaled_now
            best_logp = logp_now
    else:
        gnorm, scaled = _metrics(g, H, u)
        if scaled < float(grad_tol):
            converged = True

    gnorm_final, scaled_final = _metrics(g, H, u)
    if (not np.isfinite(scaled_final)) or (
        np.isfinite(best_scaled) and best_scaled < scaled_final
    ):
        u = jnp.asarray(best_u)
        g = jnp.asarray(best_g)
        H = jnp.asarray(best_H)
        gnorm_final, scaled_final = best_gnorm, best_scaled

    if scaled_final < float(grad_tol):
        converged = True
    # Ineffective-step stop with scaled|g| already near tol ≈ practical MAP.
    if (not converged) and scaled_final < max(5.0 * float(grad_tol), 0.5):
        last = history[-1] if history else {}
        if last.get("stalled") or last.get("skipped"):
            converged = True
    logp_final = _logp(u)
    if verbose:
        status = "converged" if converged else "stopped (not converged)"
        print(
            f"[newton_maxp] done: n_jumps={n_jumps}, "
            f"|g|_proj={gnorm_final:.3e}, scaled|g|={scaled_final:.3e}, "
            f"logp={logp_final:.6g}, {status}"
        )
        if max_jumps is None and n_jumps >= hard_cap and not converged:
            print(
                f"[newton_maxp] warning: hit safety cap ({hard_cap} jumps) "
                "without reaching grad_tol"
            )

    info = {
        "n_jumps": int(n_jumps),
        "grad_norm": gnorm_final,
        "grad_norm_scaled": scaled_final,
        "converged": bool(converged),
        "g": g,
        "H": H,
        "history": history,
        "u0_given": np.asarray(u0),
        "u_map": np.asarray(u),
        "logp": logp_final,
        "best_grad_norm": best_gnorm,
        "best_grad_norm_scaled": best_scaled,
        "best_logp": best_logp,
    }
    return u, info



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

