"""Compiled cores for the nautilus likelihoods.

Nested sampling needs no derivatives, so the nautilus path can compile its
likelihood outright. Without that, herculens' EPL profile recompiles its inner
loop on every call: ``R_omega`` (``herculens/Util/jax_util.py:41``) defines its
``@jit`` loop body *inside* the function, so the cache key is a fresh function
object each time and every lookup misses. Measured steady state, same point
repeated: 6 XLA compilations per call, ~13 ms each, ~65 ms of a 110-135 ms call,
on all ~1e5 calls of a run.

Wrapping the same arithmetic in one ``jax.jit`` moves the cache key to the shapes
and dtypes of the inputs, which never change between calls. Measured: 134 -> 8.4
ms (GW-only), 133 -> 3.1 ms (EM+GW), answers equal to 1e-13 relative.

These cores only *call* ``solver.solve``, ``select_images`` and
``compute_gw_from_images``. The differentiable solver, its ``custom_root``
gradient rule and the Newton polish are untouched, so fisher / hmc / deriv-approx
run exactly the code they ran before.
"""

import jax
import jax.numpy as jnp

from .data_sim import compute_gw_from_images
from .lens_setup import resolve_duplicate_tol, select_images

# Compiled functions are cached on the identity of the heavyweight objects they
# close over plus every static scalar. Those objects are kept alive in the value
# so CPython cannot recycle an id and hand back a function compiled against a
# different lens.
CACHE = {}


def normal_logpdf(x, mu, sigma):
    return -0.5 * jnp.sum(((x - mu) / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma ** 2))


def gw_loglike_core(lens_gw, gw_obs, error_scales):
    """Compiled GW time-delay + dL_eff log-likelihood.

    Same arithmetic as ``nautilus_source_inference._gw_loglike_from_images``,
    minus the trailing ``float()``: a host conversion inside a traced region
    raises ConcretizationTypeError, so the caller converts instead.
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


def solve_core(solver, solver_params, lens_gw, n_images):
    """Compiled lens-equation solve + image selection.

    ``lens_setup.solve_and_select`` inlined so the whole thing lands in one
    compiled program. Image positions stay as arrays rather than being unpacked
    into Python lists, and ``n_distinct`` comes back for the caller's reject test
    -- the one host sync per call.
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


def em_loglike_core(lens_image, noise, em_data):
    """Compiled EM pixel Gaussian log-likelihood.

    ``lens_image.model`` and ``noise.C_D_model`` are already jitted inside
    herculens; this collapses the surrounding eager ops into the same program.
    Gains little on its own (EM-only was already 0 compiles/call) but keeps EM+GW
    in a single program.
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
    """Refuse the compiled path where it would be silently wrong.

    ``flex_prob_model.py:161`` assigns ``k_mst`` onto ``lens_image.MassModel``,
    an object herculens passes to ``jax.jit`` as ``static_argnums=0``. The
    compiled program would then be keyed on an object whose contents changed, so
    a stale kappa0 could be reused. Better to stop than to return a plausible
    wrong number.

    The legacy ``lens_theta_E``/``lens_e1`` naming is refused too: the compiled
    builders read the flex parameter layout, and silently falling back to the
    eager path would hide that no speed-up happened.
    """
    if bool(cfg_full.get("use_mst")):
        raise NotImplementedError(
            "cfg['nautilus']['jit'] does not support use_mst=True: k_mst is "
            "assigned onto lens_image.MassModel, which herculens treats as a "
            "static jit argument, so the compiled program could reuse a stale "
            "kappa0. Set cfg['nautilus']['jit'] = False for MST systems."
        )
    if not bool(cfg_full.get("use_parameter_layout")):
        raise NotImplementedError(
            "cfg['nautilus']['jit'] requires use_parameter_layout=True (the flex "
            "lens0_*/source0_*/light0_* names); the legacy lens_theta_E/lens_e1 "
            "naming has no compiled path. Set cfg['nautilus']['jit'] = False."
        )
