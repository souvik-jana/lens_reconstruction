"""Jitted copies of the three hot likelihood cores.

Why these exist at all: gwemfish never wraps a nautilus likelihood in
``jax.jit``, so herculens' EPL profile -- which uses ``lax.scan`` internally --
retraces and recompiles its loop body on *every* call. Measured on a GW-only
source-plane likelihood, steady state, same input point repeated:

    call 1:  109.9 ms  compiles=6
    call 5:  109.3 ms  compiles=6

Six XLA compilations per call, at roughly 13 ms each, forever. Wrapping the same
arithmetic in one ``jax.jit`` compiles it once and reuses it.

These are copies, not patches. The originals in ``gwemfish.nautilus_source_inference``
and ``gwemfish.lens_setup`` keep their ``float()``/``int()`` casts and are still
what fisher / hmc / deriv-approx call, so nothing about autodiff or the
differentiable lens-equation solver changes. The only edit made here is moving the
host conversion to the outermost boundary: a ``float()`` *inside* the traced
region raises ConcretizationTypeError, which is precisely what blocks jit today
(see ``nautilus_source_inference.py:73``).

No gradient path is defined here. Nested sampling needs no derivatives, which is
why these copies cannot affect the methods that do.
"""

import jax
import jax.numpy as jnp

from gwemfish.data_sim import compute_gw_from_images
from gwemfish.lens_setup import resolve_duplicate_tol, select_images

# Compiled functions are cached on the identity of the heavyweight objects they
# close over (lens_gw, solver, lens_image) plus every static scalar. The objects
# themselves are kept alive in the value so CPython cannot recycle an id and hand
# back a function compiled against a different lens.
CACHE = {}


def normal_logpdf(x, mu, sigma):
    return -0.5 * jnp.sum(((x - mu) / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma ** 2))


def gw_loglike_fn(lens_gw, gw_obs, error_scales):
    """Compiled GW time-delay + dL_eff log-likelihood.

    Copy of ``nautilus_source_inference._gw_loglike_from_images`` with the outer
    ``float()`` removed so the body is traceable; the caller converts.
    """
    sigma_td_frac = float(error_scales.get("sigma_td", 0.3))
    sigma_dL_frac = float(error_scales.get("sigma_dL_eff", 0.3))
    td_floor = float(error_scales.get("sigma_td_floor", 1.0))

    key = ("gw", id(lens_gw), id(gw_obs), sigma_td_frac, sigma_dL_frac, td_floor)
    if key in CACHE:
        return CACHE[key][0]

    obs_td = jnp.array(gw_obs["time_delays"])
    obs_dL_eff = jnp.array(gw_obs["dL_eff"])
    sigma_td = jnp.maximum(td_floor, sigma_td_frac * obs_td)
    sigma_dL_eff = sigma_dL_frac * obs_dL_eff

    @jax.jit
    def core(x_pos, y_pos, kwargs_lens, T_star, dL):
        _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
            x_pos, y_pos, kwargs_lens, lens_gw, T_star, dL
        )
        return (normal_logpdf(model_td, obs_td, sigma_td)
                + normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff))

    CACHE[key] = (core, lens_gw, gw_obs)
    return core


def solve_fn(solver, solver_params, lens_gw, n_images):
    """Compiled lens-equation solve + image selection.

    Copy of ``nautilus_source_inference._solve_images`` composed with
    ``lens_setup.solve_and_select``. Two differences, both host-side only:
    image positions stay as arrays instead of being unpacked into Python lists
    (which costs one device transfer per image), and the single remaining host
    sync is the ``n_distinct`` reject test, done by the caller.

    ``solver.solve`` is called exactly as the real code calls it -- the
    differentiable solver, its ``custom_root`` gradient rule and the Newton polish
    are untouched and unread.
    """
    key = ("solve", id(solver), id(lens_gw), n_images,
           tuple(sorted((k, repr(v)) for k, v in solver_params.items())))
    if key in CACHE:
        return CACHE[key][0]

    tol = resolve_duplicate_tol(
        getattr(solver, "duplicate_tol", None),
        polished=getattr(solver, "polish", True),
    )

    @jax.jit
    def core(y0, y1, kwargs_lens, cx, cy):
        betas = jnp.array([y0, y1])
        thetas, betas_out = solver.solve(betas, kwargs_lens, **solver_params)
        mu_all = lens_gw.magnification(thetas[:, 0], thetas[:, 1], kwargs_lens)
        x_pos, y_pos, _, _, _, flags = select_images(
            thetas, betas_out, cx, cy, n_images, tol_dup=tol, magnifications=mu_all)
        return x_pos, y_pos, flags["n_distinct"]

    CACHE[key] = (core, solver, lens_gw)
    return core


def em_loglike_fn(lens_image, noise, em_data):
    """Compiled EM pixel Gaussian log-likelihood.

    ``lens_image.model`` and ``noise.C_D_model`` are already jitted inside
    herculens; this collapses the surrounding eager ops into the same program.
    """
    key = ("em", id(lens_image), id(noise), id(em_data))
    if key in CACHE:
        return CACHE[key][0]

    data = jnp.asarray(em_data)

    @jax.jit
    def core(kwargs_lens, kwargs_source, kwargs_lens_light, sigma_bkg):
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        return jnp.sum(-0.5 * ((data - model_image) ** 2 / model_var
                               + jnp.log(2 * jnp.pi * model_var)))

    CACHE[key] = (core, lens_image, noise, em_data)
    return core


def check_jittable(cfg_full):
    """Refuse the jit path where it would be silently wrong.

    ``flex_prob_model.py:161`` assigns ``self.lens_image.MassModel.kappa0 = k_mst``
    -- mutating an object herculens passes to ``jax.jit`` as ``static_argnums=0``.
    The compiled program is then keyed on an object whose contents changed, so a
    stale kappa0 can be reused. Rather than return a plausible wrong number, stop.
    """
    if bool(cfg_full.get("use_mst")):
        raise NotImplementedError(
            "prototype jit path does not support use_mst=True: k_mst is assigned "
            "onto lens_image.MassModel, which herculens treats as a static jit "
            "argument. Run with jit=False for MST systems."
        )
