"""
Probabilistic model for EM+GW joint inference.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import herculens as hcl
from .data_sim import compute_gw_from_images
from .config import arcsecond_to_radians, Mpc_to_m, c, SOLVER_PARAMS
from .ellipticity_reparam import (
    compute_qphi_ellipticity,
    validate_parametrization,
    warn_if_ellipticity_prior_keys_unused,
)
from .lens_setup import image_count_penalty, remove_central_image, solve_and_select


# ---------------------------------------------------------------------------
# Default prior registries and default geometry
# ---------------------------------------------------------------------------

from .priors import (
    _make_default_priors_em_gw,
    DEFAULT_PRIORS_EM_GW,
    DEFAULT_PRIORS_EM_ONLY,
    DEFAULT_PRIORS_GW_ONLY,
    DEFAULT_IMAGE_POSITION_PRIORS_EM,
    DEFAULT_IMAGE_POSITION_PRIORS_GW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prior_lens(lens_theta_E, lens_e1, lens_e2, lens_gamma,
                      lens_center_x, lens_center_y, gamma1, gamma2,
                      ra_0=None, dec_0=None):
    if ra_0  is None: ra_0  = jnp.asarray(0.0)
    if dec_0 is None: dec_0 = jnp.asarray(0.0)
    return [
        {
            'theta_E':  lens_theta_E,
            'e1':       lens_e1,
            'e2':       lens_e2,
            'gamma':    lens_gamma,
            'center_x': lens_center_x,
            'center_y': lens_center_y,
        },
        {
            'gamma1': gamma1,
            'gamma2': gamma2,
            'ra_0':   ra_0,
            'dec_0':  dec_0,
        }
    ]


def _sample_lens_ellipticity(p, parametrization):
    if parametrization == "q_phi":
        return compute_qphi_ellipticity('lens', p)
    return p['lens_e1'](), p['lens_e2']()


def _sample_image_positions(n_images, priors, image_position_priors):
    ip = image_position_priors
    x_list, y_list = [], []
    for i in range(n_images):
        xk, yk = f'image_x{i + 1}', f'image_y{i + 1}'
        if xk in priors:
            x = priors[xk]()
        else:
            # Prefer per-image centered boxes if geometry provides them;
            # otherwise fall back to a global (min/max) box.
            if 'x_image_true' in ip and 'delx' in ip:
                mean_x = ip['x_image_true'][i]
                half_dx = ip['delx'][i] / 2
                x = numpyro.sample(xk, dist.Uniform(mean_x - half_dx, mean_x + half_dx))
            else:
                x = numpyro.sample(xk, dist.Uniform(ip['minx'], ip['maxx']))
        if yk in priors:
            y = priors[yk]()
        else:
            if 'y_image_true' in ip and 'dely' in ip:
                mean_y = ip['y_image_true'][i]
                half_dy = ip['dely'][i] / 2
                y = numpyro.sample(yk, dist.Uniform(mean_y - half_dy, mean_y + half_dy))
            else:
                y = numpyro.sample(yk, dist.Uniform(ip['miny'], ip['maxy']))
        x_list.append(x)
        y_list.append(y)
    return jnp.array(x_list), jnp.array(y_list)


# def _sample_image_positions_flat(n_images, priors, image_positions):
#     # Backward-compatible wrapper.
#     # Keep the name, but delegate to the unified implementation.
#     return _sample_image_positions(n_images, priors, image_positions)


# ---------------------------------------------------------------------------
# ProbModel  (EM + GW, image-plane positions)
# ---------------------------------------------------------------------------

class ProbModel(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW parameter estimation."""

    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None,
                 priors=None, image_position_priors=None, image_positions=None,
                 gw_error_scales=None, use_mst: bool = False,
                 lens_parametrization="e1e2"):
        """
        Args:
            n_images:        Number of lensed images.
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            em_observations: Dict with 'data'.
            lens_image:      hcl.LensImage instance.
            lens_gw:         LensImageGW instance.
            noise:           hcl.Noise instance (background_rms=None for inference).
            priors:          Optional dict of zero-arg callables overriding defaults.
                             All parameters including source_center_x/y and
                             lens_center_x/y can be sampled by passing lambdas here.
            image_position_priors: Optional dict overriding image-position prior geometry.
            image_positions: Legacy alias for image_position_priors (kept for compatibility).
            gw_error_scales: Optional dict scaling GW likelihood uncertainties.
                             Keys: 'sigma_td', 'sigma_dL_eff', 'epsilon'.
            use_mst:         If True, include mass-sheet parameter ``k_mst`` in GW forward model.
            lens_parametrization: "e1e2" (default) or "q_phi" -- see ellipticity_reparam.py.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image      = lens_image
        self.lens_gw         = lens_gw
        self.noise           = noise
        self.use_mst         = bool(use_mst)
        self.pix_scl         = 0.4
        self.lens_parametrization = validate_parametrization(lens_parametrization)
        warn_if_ellipticity_prior_keys_unused(priors, self.lens_parametrization, ("lens",))
        self.priors          = {**_make_default_priors_em_gw(self.pix_scl), **(priors or {})}
        if image_positions is not None and image_position_priors is None:
            image_position_priors = image_positions
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_EM,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors

        # --- GW cosmological params ---
        T_star = p['T_star']()
        dL     = p['dL']()

        # --- Source light ---
        prior_source = [{
            'amp':      p['source_amp'](),
            'R_sersic': p['source_R_sersic'](),
            'n_sersic': p['source_n'](),
            'e1':       p['source_e1'](),
            'e2':       p['source_e2'](),
            'center_x': p['source_center_x'](),   # fixed or sampled via priors=
            'center_y': p['source_center_y'](),   # fixed or sampled via priors=
        }]

        # --- Lens light ---
        prior_lens_light = [{
            'amp':      p['light_amp'](),
            'R_sersic': p['light_R_sersic'](),
            'n_sersic': p['light_n'](),
            'e1':       p['light_e1'](),
            'e2':       p['light_e2'](),
            'center_x': p['light_center_x'](),
            'center_y': p['light_center_y'](),
        }]

        # --- Lens mass ---
        # Sample theta_E before e1/e2 so the numpyro RNG-split order matches the
        # original (theta_E, e1, e2, gamma, ...) exactly in default e1e2 mode.
        lens_theta_E = p['lens_theta_E']()
        lens_e1, lens_e2 = _sample_lens_ellipticity(p, self.lens_parametrization)
        prior_lens = _build_prior_lens(
            lens_theta_E  = lens_theta_E,
            lens_e1       = lens_e1,
            lens_e2       = lens_e2,
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = p['lens_center_x'](),  # fixed or sampled via priors=
            lens_center_y = p['lens_center_y'](),  # fixed or sampled via priors=
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        # --- Noise ---
        sigma_bkg = p['noise_sigma_bkg']()

        # --- EM likelihood ---
        model_image = self.lens_image.model(
            kwargs_lens       = prior_lens,
            kwargs_lens_light = prior_lens_light,
            kwargs_source     = prior_source,
        )
        em_data   = self.em_observations['data']
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample('obs',
                       dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
                       obs=em_data)

        # --- Image positions ---
        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors)

        # --- GW likelihood ---
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs       = self.gw_observations
        sigma_td     = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs['time_delays'],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs['dL_eff']
        epsilon      = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))

    def params2kwargs(self, params):
        return {
            'kwargs_lens': [
                {
                    'theta_E':  params['lens_theta_E'],
                    'e1':       params['lens_e1'],
                    'e2':       params['lens_e2'],
                    'gamma':    params['lens_gamma'],
                    'center_x': params.get('lens_center_x', 0.0),
                    'center_y': params.get('lens_center_y', 0.0),
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0':   0.0,
                    'dec_0':  0.0,
                }
            ],
            'kwargs_source': [{
                'amp':      params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1':       params['source_e1'],
                'e2':       params['source_e2'],
                'center_x': params.get('source_center_x', 0.05),
                'center_y': params.get('source_center_y', 0.1),
            }],
            'kwargs_lens_light': [{
                'amp':      params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1':       params['light_e1'],
                'e2':       params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y'],
            }],
            'image_positions': [
                (params.get(f'image_x{i+1}', 0.0),
                 params.get(f'image_y{i+1}', 0.0))
                for i in range(self.n_images)
            ],
        }


# ---------------------------------------------------------------------------
# ProbModelSourcePlane  (EM + GW, source-plane positions)
# ---------------------------------------------------------------------------

class ProbModelSourcePlane(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW inference — source-plane parametrisation."""

    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None,
                 solver=None, solver_params=None, priors=None,
                 gw_error_scales=None, use_mst: bool = False,
                 lens_parametrization="e1e2"):
        """
        Args:
            n_images:        Number of lensed images (excluding central image).
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            em_observations: Dict with 'data'.
            lens_image:      hcl.LensImage instance.
            lens_gw:         LensImageGW instance.
            noise:           hcl.Noise instance.
            solver:          LensEquationSolver_helens instance.
            solver_params:   Dict of solver parameters. Defaults to SOLVER_PARAMS.
            priors:          Optional override dict. Also accepts:
                               'y0gw': lambda returning sampled y0gw scalar
                               'y1gw': lambda returning sampled y1gw scalar
                             Source/lens center keys also accepted.
            gw_error_scales: Optional dict scaling GW likelihood uncertainties.
                             Keys: 'sigma_td', 'sigma_dL_eff', 'epsilon'.
            use_mst:         If True, include mass-sheet ``k_mst`` in GW forward model.
            lens_parametrization: "e1e2" (default) or "q_phi" -- see ellipticity_reparam.py.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image      = lens_image
        self.lens_gw         = lens_gw
        self.noise           = noise
        self.use_mst         = bool(use_mst)
        self.solver          = solver
        self.solver_params   = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.pix_scl         = 0.4
        self.lens_parametrization = validate_parametrization(lens_parametrization)
        warn_if_ellipticity_prior_keys_unused(priors, self.lens_parametrization, ("lens",))

        _source_plane_defaults = {
            'y0gw': lambda: numpyro.sample('y0gw', dist.Uniform(0.045, 0.055)),
            'y1gw': lambda: numpyro.sample('y1gw', dist.Uniform(9e-7,  2e-6)),
        }
        base = {**_make_default_priors_em_gw(self.pix_scl), **_source_plane_defaults}
        self.priors = {**base, **(priors or {})}
        self.gw_error_scales = {
            "sigma_td": 0.3,
            "sigma_dL_eff": 0.3,
            "epsilon": 0.005,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors

        T_star = p['T_star']()
        dL     = p['dL']()

        prior_source = [{
            'amp':      p['source_amp'](),
            'R_sersic': p['source_R_sersic'](),
            'n_sersic': p['source_n'](),
            'e1':       p['source_e1'](),
            'e2':       p['source_e2'](),
            'center_x': p['source_center_x'](),
            'center_y': p['source_center_y'](),
        }]

        prior_lens_light = [{
            'amp':      p['light_amp'](),
            'R_sersic': p['light_R_sersic'](),
            'n_sersic': p['light_n'](),
            'e1':       p['light_e1'](),
            'e2':       p['light_e2'](),
            'center_x': p['light_center_x'](),
            'center_y': p['light_center_y'](),
        }]

        lens_center_x = p['lens_center_x']()
        lens_center_y = p['lens_center_y']()
        # Sample theta_E before e1/e2 so the numpyro RNG-split order matches the
        # original (theta_E, e1, e2, gamma, ...) exactly in default e1e2 mode.
        lens_theta_E = p['lens_theta_E']()
        lens_e1, lens_e2 = _sample_lens_ellipticity(p, self.lens_parametrization)
        prior_lens = _build_prior_lens(
            lens_theta_E  = lens_theta_E,
            lens_e1       = lens_e1,
            lens_e2       = lens_e2,
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = lens_center_x,
            lens_center_y = lens_center_y,
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        sigma_bkg = p['noise_sigma_bkg']()

        model_image = self.lens_image.model(
            kwargs_lens       = prior_lens,
            kwargs_lens_light = prior_lens_light,
            kwargs_source     = prior_source,
        )
        em_data   = self.em_observations['data']
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample('obs',
                       dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
                       obs=em_data)

        y0gw  = p['y0gw']()
        y1gw  = p['y1gw']()
        betas = jnp.array([y0gw, y1gw])

        x_pos_array, y_pos_array, _, sel_flags = solve_and_select(
            self.solver, self.solver_params, betas, prior_lens, self.lens_gw,
            self.n_images, lens_center_x, lens_center_y)
        # Reject parameters whose image count does not match the observation.
        numpyro.factor("image_count", image_count_penalty(sel_flags, self.n_images))

        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, _, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs       = self.gw_observations
        sigma_td     = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs['time_delays'],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs['dL_eff']
        epsilon      = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)

    def params2kwargs(self, params):
        return {
            'kwargs_lens': [
                {
                    'theta_E':  params['lens_theta_E'],
                    'e1':       params['lens_e1'],
                    'e2':       params['lens_e2'],
                    'gamma':    params['lens_gamma'],
                    'center_x': params.get('lens_center_x', 0.0),
                    'center_y': params.get('lens_center_y', 0.0),
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0':   0.0,
                    'dec_0':  0.0,
                }
            ],
            'kwargs_source': [{
                'amp':      params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1':       params['source_e1'],
                'e2':       params['source_e2'],
                'center_x': params.get('source_center_x', 0.05),
                'center_y': params.get('source_center_y', 0.1),
            }],
            'kwargs_lens_light': [{
                'amp':      params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1':       params['light_e1'],
                'e2':       params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y'],
            }],
            'y0gw': params.get('y0gw', 0.0),
            'y1gw': params.get('y1gw', 0.0),
        }


# ---------------------------------------------------------------------------
# ProbModelSourcePlane_GW_only  (GW only, source-plane positions)
# ---------------------------------------------------------------------------

class ProbModelSourcePlane_GW_only(hcl.NumpyroModel):
    """GW-only probabilistic model, source-plane parametrisation.

    Samples y0gw/y1gw directly (instead of image positions), solves the lens
    equation *inside* the model via `solver.solve(...)`, then computes the GW
    likelihood from the solved images -- mirrors ProbModel_GW_only's structure,
    but source-plane instead of image-plane. Pass a DifferentiableLensEquationSolver
    (differentiable_solver.py) as `solver` for gradient-based inference
    (deriv-approx-source / Fisher); the raw helens solver gives exact-zero
    gradients and must not be used here for that purpose. nautilus-source (no
    gradients needed) is unaffected -- it already uses the raw solver via
    nautilus_source_inference.py.

    Note: unlike ProbModel_GW_only (image-plane), this model adds no
    betx_x_diff/bety_y_diff/log_jacobian terms. Those exist on the image-plane
    side to (a) softly penalize inconsistent sampled images and (b) correct a
    flat image-plane prior to a flat source-plane prior. Neither is needed
    here: the solver enforces beta self-consistency exactly by construction,
    and the y0gw/y1gw prior is already flat in the source plane directly.
    This also keeps this model's likelihood directly comparable to
    nautilus-source's _gw_loglike_from_images (same forward model, same terms).
    """

    def __init__(self, n_images=4, gw_observations=None, lens_gw=None,
                 solver=None, solver_params=None, priors=None,
                 gw_error_scales=None, use_mst: bool = False,
                 lens_parametrization="e1e2"):
        """
        Args:
            n_images:        Number of lensed images (excluding central image).
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            lens_gw:         LensImageGW instance.
            solver:          DifferentiableLensEquationSolver instance (required
                             for correct gradients; see class docstring).
            solver_params:   Dict of solver parameters. Defaults to SOLVER_PARAMS.
            priors:          Optional override dict. Also accepts 'y0gw'/'y1gw'
                             lambdas overriding the default Uniform(-1, 1) box.
            gw_error_scales: Optional dict scaling GW likelihood uncertainties.
                             Keys: 'sigma_td', 'sigma_dL_eff'.
            use_mst:         If True, include mass-sheet `k_mst` in GW forward model.
            lens_parametrization: "e1e2" (default) or "q_phi" -- see ellipticity_reparam.py.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw         = lens_gw
        self.use_mst         = bool(use_mst)
        self.solver          = solver
        self.solver_params   = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.lens_parametrization = validate_parametrization(lens_parametrization)
        warn_if_ellipticity_prior_keys_unused(priors, self.lens_parametrization, ("lens",))

        _source_plane_defaults = {
            'y0gw': lambda: numpyro.sample('y0gw', dist.Uniform(-1.0, 1.0)),
            'y1gw': lambda: numpyro.sample('y1gw', dist.Uniform(-1.0, 1.0)),
        }
        base = {**DEFAULT_PRIORS_GW_ONLY, **_source_plane_defaults}
        self.priors = {**base, **(priors or {})}
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.02,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors

        T_star = p['T_star']()
        dL     = p['dL']()

        lens_center_x = p['lens_center_x']()
        lens_center_y = p['lens_center_y']()
        # Sample theta_E before e1/e2 so the numpyro RNG-split order matches the
        # original (theta_E, e1, e2, gamma, ...) exactly in default e1e2 mode.
        lens_theta_E = p['lens_theta_E']()
        lens_e1, lens_e2 = _sample_lens_ellipticity(p, self.lens_parametrization)
        prior_lens = _build_prior_lens(
            lens_theta_E  = lens_theta_E,
            lens_e1       = lens_e1,
            lens_e2       = lens_e2,
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = lens_center_x,
            lens_center_y = lens_center_y,
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        y0gw  = p['y0gw']()
        y1gw  = p['y1gw']()
        betas = jnp.array([y0gw, y1gw])

        x_pos_array, y_pos_array, _, sel_flags = solve_and_select(
            self.solver, self.solver_params, betas, prior_lens, self.lens_gw,
            self.n_images, lens_center_x, lens_center_y)
        # Reject parameters whose image count does not match the observation.
        numpyro.factor("image_count", image_count_penalty(sel_flags, self.n_images))

        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, _, model_dL_eff,
         _, _, _, _) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs       = self.gw_observations
        sigma_td     = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs['time_delays'],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs['dL_eff']

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])

    def params2kwargs(self, params):
        return {
            'kwargs_lens': [
                {
                    'theta_E':  params['lens_theta_E'],
                    'e1':       params['lens_e1'],
                    'e2':       params['lens_e2'],
                    'gamma':    params['lens_gamma'],
                    'center_x': params.get('lens_center_x', 0.0),
                    'center_y': params.get('lens_center_y', 0.0),
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0':   0.0,
                    'dec_0':  0.0,
                }
            ],
            'y0gw': params.get('y0gw', 0.0),
            'y1gw': params.get('y1gw', 0.0),
        }


# ---------------------------------------------------------------------------
# ProbModelFisher  (EM + GW, Fisher / banana approximate likelihood)
# ---------------------------------------------------------------------------

class ProbModelFisher(hcl.NumpyroModel):
    """Probabilistic model with approximate likelihood from Fisher matrix."""

    def __init__(self, keys_to_include, approx_logp,
                 priors=None, image_position_priors=None, image_positions=None):
        """
        Args:
            keys_to_include: Ordered list of parameter keys stacked into approx_logp vector.
            approx_logp:     Callable f(u: jnp.ndarray) -> scalar.
            priors:          Optional override dict.
            image_position_priors: Optional image-position prior geometry override.
            image_positions: Legacy alias for image_position_priors (kept for compatibility).
        """
        self.keys_to_include = keys_to_include
        self.approx_logp     = approx_logp
        self.pix_scl         = 0.4
        self.priors          = {**_make_default_priors_em_gw(self.pix_scl), **(priors or {})}
        if image_positions is not None and image_position_priors is None:
            image_position_priors = image_positions
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_EM,
            **(image_position_priors or {}),
        }
        super().__init__()

    def model(self):
        p  = self.priors
        ip = self.image_position_priors
        param_dict = {}

        for key in self.keys_to_include:
            if key in p:
                param_dict[key] = p[key]()
            elif key.startswith('image_x'):
                i    = int(key[-1]) - 1
                if 'x_image_true' in ip and 'delx' in ip:
                    mean = ip['x_image_true'][i]
                    half = ip['delx'][i] / 2
                    param_dict[key] = numpyro.sample(key, dist.Uniform(mean - half, mean + half))
                else:
                    param_dict[key] = numpyro.sample(key, dist.Uniform(ip['minx'], ip['maxx']))
            elif key.startswith('image_y'):
                i    = int(key[-1]) - 1
                if 'y_image_true' in ip and 'dely' in ip:
                    mean = ip['y_image_true'][i]
                    half = ip['dely'][i] / 2
                    param_dict[key] = numpyro.sample(key, dist.Uniform(mean - half, mean + half))
                else:
                    param_dict[key] = numpyro.sample(key, dist.Uniform(ip['miny'], ip['maxy']))
            else:
                raise ValueError(
                    f"Parameter '{key}' in keys_to_include is not recognised. "
                    f"Add it to the priors dict or handle it as an image position.")

        uarr = jnp.array([param_dict[k] for k in self.keys_to_include])
        numpyro.factor("banana_logprob", self.approx_logp(uarr))


# ---------------------------------------------------------------------------
# ProbModel_EM_only (EM only, image-plane likelihood)
# ---------------------------------------------------------------------------
class ProbModel_EM_only(hcl.NumpyroModel):
    """EM-only probabilistic model (no GW likelihood)."""

    def __init__(self, em_observations=None, lens_image=None, noise=None,
                 priors=None, lens_parametrization="e1e2"):
        """
        Args:
            em_observations: Dict with 'data' key containing the EM observation.
            lens_image: hcl.LensImage instance configured for inference.
            noise: hcl.Noise instance configured for inference (background_rms=None).
            priors: Optional override dict. Expected to follow the numpyro-style
                    priors registry used in this module (zero-arg callables).
            lens_parametrization: "e1e2" (default) or "q_phi" -- see ellipticity_reparam.py.
        """
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.noise = noise
        self.pix_scl = 0.4
        self.lens_parametrization = validate_parametrization(lens_parametrization)
        warn_if_ellipticity_prior_keys_unused(priors, self.lens_parametrization, ("lens",))
        self.priors = {**DEFAULT_PRIORS_EM_ONLY, **(priors or {})}
        super().__init__()

    def model(self):
        p = self.priors

        sigma_bkg = p["noise_sigma_bkg"]()

        prior_source = [{
            "amp": p["source_amp"](),
            "R_sersic": p["source_R_sersic"](),
            "n_sersic": p["source_n"](),
            "e1": p["source_e1"](),
            "e2": p["source_e2"](),
            "center_x": p["source_center_x"](),
            "center_y": p["source_center_y"](),
        }]

        prior_lens_light = [{
            "amp": p["light_amp"](),
            "R_sersic": p["light_R_sersic"](),
            "n_sersic": p["light_n"](),
            "e1": p["light_e1"](),
            "e2": p["light_e2"](),
            "center_x": p["light_center_x"](),
            "center_y": p["light_center_y"](),
        }]

        # Sample theta_E before e1/e2 so the numpyro RNG-split order matches the
        # original (theta_E, e1, e2, gamma, ...) exactly in default e1e2 mode.
        lens_theta_E = p["lens_theta_E"]()
        lens_e1, lens_e2 = _sample_lens_ellipticity(p, self.lens_parametrization)
        prior_lens = _build_prior_lens(
            lens_theta_E=lens_theta_E,
            lens_e1=lens_e1,
            lens_e2=lens_e2,
            lens_gamma=p["lens_gamma"](),
            lens_center_x=p["lens_center_x"](),
            lens_center_y=p["lens_center_y"](),
            gamma1=p["lens_gamma1"](),
            gamma2=p["lens_gamma2"](),
        )

        # Build predicted image.
        model_image = self.lens_image.model(
            kwargs_lens=prior_lens,
            kwargs_lens_light=prior_lens_light,
            kwargs_source=prior_source,
        )

        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )


# ---------------------------------------------------------------------------
# ProbModel_GW_only  (GW only, image-plane positions)
# ---------------------------------------------------------------------------

class ProbModel_GW_only(hcl.NumpyroModel):
    """GW-only probabilistic model (no EM likelihood)."""

    def __init__(self, n_images=4, gw_observations=None, lens_gw=None,
                 priors=None, image_position_priors=None, image_positions=None,
                 gw_error_scales=None, use_mst: bool = False,
                 lens_parametrization="e1e2"):
        """
        Args:
            n_images:        Number of lensed images.
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            lens_gw:         LensImageGW instance.
            priors:          Optional override dict from DEFAULT_PRIORS_GW_ONLY.
            image_position_priors: Optional image-position prior geometry override.
            image_positions: Legacy alias for image_position_priors (kept for compatibility).
            gw_error_scales: Optional dict scaling GW likelihood uncertainties.
                             Keys: 'sigma_td', 'sigma_dL_eff', 'epsilon'.
            use_mst:         If True, include mass-sheet ``k_mst`` in GW forward model.
            lens_parametrization: "e1e2" (default) or "q_phi" -- see ellipticity_reparam.py.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw         = lens_gw
        self.use_mst         = bool(use_mst)
        self.pix_scl         = 0.4
        self.lens_parametrization = validate_parametrization(lens_parametrization)
        warn_if_ellipticity_prior_keys_unused(priors, self.lens_parametrization, ("lens",))
        self.priors          = {**DEFAULT_PRIORS_GW_ONLY, **(priors or {})}
        if image_positions is not None and image_position_priors is None:
            image_position_priors = image_positions
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_GW,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.02,
            "epsilon": 0.001,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors

        T_star = p['T_star']()
        dL     = p['dL']()

        # Sample theta_E before e1/e2 so the numpyro RNG-split order matches the
        # original (theta_E, e1, e2, gamma, ...) exactly in default e1e2 mode.
        lens_theta_E = p['lens_theta_E']()
        lens_e1, lens_e2 = _sample_lens_ellipticity(p, self.lens_parametrization)
        prior_lens = _build_prior_lens(
            lens_theta_E  = lens_theta_E,
            lens_e1       = lens_e1,
            lens_e2       = lens_e2,
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = p['lens_center_x'](),
            lens_center_y = p['lens_center_y'](),
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors)

        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs       = self.gw_observations
        sigma_td     = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs['time_delays'],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs['dL_eff']
        epsilon      = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))


# ---------------------------------------------------------------------------
# ProbModelFisher_GW_only  (GW only, Fisher / banana approximate likelihood)
# ---------------------------------------------------------------------------

class ProbModelFisher_GW_only(hcl.NumpyroModel):
    """GW-only probabilistic model with approximate Fisher likelihood."""

    def __init__(self, keys_to_include, approx_logp,
                 priors=None, image_position_priors=None, image_positions=None):
        """
        Args:
            keys_to_include: Ordered list of parameter keys.
            approx_logp:     Callable f(u: jnp.ndarray) -> scalar.
            priors:          Optional override dict from DEFAULT_PRIORS_GW_ONLY.
            image_position_priors: Optional image-position prior geometry override.
            image_positions: Legacy alias for image_position_priors (kept for compatibility).
        """
        self.keys_to_include = keys_to_include
        self.approx_logp     = approx_logp
        self.pix_scl         = 0.4
        self.priors          = {**DEFAULT_PRIORS_GW_ONLY, **(priors or {})}
        if image_positions is not None and image_position_priors is None:
            image_position_priors = image_positions
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_GW,
            **(image_position_priors or {}),
        }
        super().__init__()

    def model(self):
        p  = self.priors
        ip = self.image_position_priors
        param_dict = {}

        for key in self.keys_to_include:
            if key in p:
                param_dict[key] = p[key]()
            elif key.startswith('image_x'):
                # Per-image centered box fallback if available; otherwise global min/max.
                i = int(key[-1]) - 1
                if 'x_image_true' in ip and 'delx' in ip:
                    mean_x = ip['x_image_true'][i]
                    half_dx = ip['delx'][i] / 2
                    param_dict[key] = numpyro.sample(key, dist.Uniform(mean_x - half_dx, mean_x + half_dx))
                else:
                    param_dict[key] = numpyro.sample(key, dist.Uniform(ip['minx'], ip['maxx']))
            elif key.startswith('image_y'):
                i = int(key[-1]) - 1
                if 'y_image_true' in ip and 'dely' in ip:
                    mean_y = ip['y_image_true'][i]
                    half_dy = ip['dely'][i] / 2
                    param_dict[key] = numpyro.sample(key, dist.Uniform(mean_y - half_dy, mean_y + half_dy))
                else:
                    param_dict[key] = numpyro.sample(key, dist.Uniform(ip['miny'], ip['maxy']))
            else:
                raise ValueError(
                    f"Parameter '{key}' in keys_to_include is not recognised. "
                    f"Add it to the priors dict or handle it as an image position.")

        uarr = jnp.array([param_dict[k] for k in self.keys_to_include])
        numpyro.factor("banana_logprob", self.approx_logp(uarr))