"""
Probabilistic model for EM+GW joint inference.

This module contains the ProbModel class which defines the joint
probabilistic model for electromagnetic and gravitational wave observations.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import herculens as hcl
from .data_sim import compute_gw_from_images
from .config import arcsecond_to_radians, Mpc_to_m, c, SOLVER_PARAMS, e1e2_to_qphi
from .lens_setup import remove_central_image


class ProbModel(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW parameter estimation."""
    
    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None):
        """Initialize probabilistic model for EM+GW joint inference.
        
        Args:
            n_images: Number of lensed images
            gw_observations: Dict with 'time_delays' and 'dL_eff'
            em_observations: Dict with 'data' (observed image)
            lens_image: hcl.LensImage instance (from simulate_em)
            lens_gw: LensImageGW instance (from simulate_gw)
            noise: hcl.Noise instance (created with background_rms=None for inference)
        """
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.lens_gw = lens_gw
        self.noise = noise
        self.pix_scl = 0.4  # Pixel scale in arcsec
        super().__init__()
    
    def model(self):
        """Define the probabilistic model with priors and likelihoods.
        
        During inference, this is called many times with different parameter values.
        Key operations:
        1. Sample parameters (lens, source, light, noise_sigma_bkg, T_star, dL, image positions)
        2. Compute model_image = lens_image.model(kwargs_lens, kwargs_source, kwargs_lens_light)
        3. Compute noise variance: model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
           where sigma_bkg is sampled at each step
        4. EM likelihood: numpyro.sample('obs', Normal(model_image, sqrt(model_var)), obs=em_data)
        5. Compute GW observables: model_gw = lens_gw.compute(x_pos, y_pos, kwargs_lens, D_dt)
           where D_dt = (T_star*c)/(Mpc_to_m*arcsecond_to_radians**2)
        6. GW likelihood: numpyro.sample('tdelays_obs', Normal(model_time_delays, sigma_td), ...)
        """
        # GW parameters
        T_star = numpyro.sample('T_star', dist.Uniform(1e4, 1e8))  # in seconds
        dL = numpyro.sample('dL', dist.Uniform(10000.0, 21800.0))  # in Mpc
        
        # Source light parameters
        source_amp = numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=2.4, high=10.0))
        source_R_sersic = numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))
        source_n = numpyro.sample('source_n', dist.Uniform(1., 2.5))
        source_e1 = numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))
        source_e2 = numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))
        # True source center position (hardcoded)
        source_center_x = jnp.asarray(0.05)
        source_center_y = jnp.asarray(0.1)
        
        prior_source = [{
            'amp': source_amp,
            'R_sersic': source_R_sersic,
            'n_sersic': source_n,
            'e1': source_e1,
            'e2': source_e2,
            'center_x': source_center_x,
            'center_y': source_center_y
        }]
        
        # Lens light parameters
        cx_l = numpyro.sample('light_center_x', dist.Normal(0., self.pix_scl/2))
        cy_l = numpyro.sample('light_center_y', dist.Normal(0., self.pix_scl/2))
        e1_l = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))
        e2_l = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))
        light_amp = numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=9.5))
        light_R_sersic = numpyro.sample('light_R_sersic', dist.TruncatedNormal(1.0, 0.5, low=0.88, high=1.15))
        light_n = numpyro.sample('light_n', dist.Uniform(2.4, 5.))
        
        prior_lens_light = [{
            'amp': light_amp,
            'R_sersic': light_R_sersic,
            'n_sersic': light_n,
            'e1': e1_l,
            'e2': e2_l,
            'center_x': cx_l,
            'center_y': cy_l
        }]
        
        # Lens mass parameters
        lens_theta_E = numpyro.sample('lens_theta_E', dist.Uniform(1.99, 2.01))
        lens_e1 = numpyro.sample('lens_e1', dist.Uniform(-0.065, -0.050))
        lens_e2 = numpyro.sample('lens_e2', dist.Uniform(0.075, 0.11))
        lens_gamma = numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.05))
        # True lens center position (hardcoded)
        lens_center_x = jnp.asarray(0.0)
        lens_center_y = jnp.asarray(0.0)
        gamma1 = numpyro.sample('lens_gamma1', dist.Uniform(-0.006, 0.005))
        gamma2 = numpyro.sample('lens_gamma2', dist.Uniform(-0.005, 0.009))
        
        prior_lens = [
            {
                'theta_E': lens_theta_E,
                'e1': lens_e1,
                'e2': lens_e2,
                'gamma': lens_gamma,
                'center_x': lens_center_x,
                'center_y': lens_center_y
            },
            {
                'gamma1': gamma1,
                'gamma2': gamma2,
                'ra_0': jnp.asarray(0.0),
                'dec_0': jnp.asarray(0.0)
            }
        ]
        
        # Noise parameter
        sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012))
        
        # EM likelihood
        model_params = dict(
            kwargs_lens=prior_lens,
            kwargs_lens_light=prior_lens_light,
            kwargs_source=prior_source
        )
        
        model_image = self.lens_image.model(**model_params)
        em_data = self.em_observations['data']
        
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        model_std = jnp.sqrt(model_var)
        
        numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=em_data)
        
        # Image positions
        # True image positions (hardcoded - these are typical values for the default lens configuration)
        # These would normally come from solving the lens equation with true parameters
        x_image_true = jnp.array([ 1.90461434, -1.63544685,  0.70943792, -1.14517025])   # True GW image x positions
        y_image_true = jnp.array([-0.90999308,  1.19344445,  1.80599259, -1.50457468])   # True GW image y positions
        
        image_positions = []
        x_pos_array = []
        y_pos_array = []
        delx = jnp.array([0.2, 0.35, 0.49, 0.3])
        dely = jnp.array([0.4, 0.4, 0.35, 0.3])
        
        for i in range(self.n_images):
            mean_x = x_image_true[i]
            mean_y = y_image_true[i]
            
            minx = mean_x - delx[i]/2
            maxx = mean_x + delx[i]/2
            miny = mean_y - dely[i]/2
            maxy = mean_y + dely[i]/2
            
            x_pos = numpyro.sample(f'image_x{i+1}', dist.Uniform(minx, maxx))
            y_pos = numpyro.sample(f'image_y{i+1}', dist.Uniform(miny, maxy))
            
            image_positions.append((x_pos, y_pos))
            x_pos_array.append(x_pos)
            y_pos_array.append(y_pos)
        
        x_pos_array = jnp.array(x_pos_array)
        y_pos_array = jnp.array(y_pos_array)
        
        # GW likelihood
        (model_gw, model_time_delays, model_magnifications, model_dL_eff,
         beta_x, beta_y, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL
        )
        
        # GW likelihood terms
        gw_obs = self.gw_observations
        sigma_td = 0.3 * gw_obs['time_delays']
        sigma_dL_eff = 0.3 * gw_obs['dL_eff']
        epsilon = 0.005 * jnp.ones_like(betx_x_diff)
        
        numpyro.sample('tdelays_obs', dist.Independent(dist.Normal(model_time_delays, sigma_td), 1), 
                      obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs', dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1), 
                      obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1), 
                      obs=betx_x_diff)
        numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1), 
                      obs=bety_y_diff)
    
    def params2kwargs(self, params):
        """Convert flat parameter dict to lens_image.model() kwargs format.
        
        Args:
            params: Dictionary of sampled parameters
        
        Returns:
            Dictionary with kwargs_lens, kwargs_source, kwargs_lens_light, etc.
        """
        kw = {
            'kwargs_lens': [
                {
                    'theta_E': params['lens_theta_E'],
                    'e1': params['lens_e1'],
                    'e2': params['lens_e2'],
                    'gamma': params['lens_gamma'],
                    'center_x': params['lens_center_x'],
                    'center_y': params['lens_center_y']
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0': 0.0,
                    'dec_0': 0.0
                }
            ],
            'kwargs_source': [{
                'amp': params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1': params['source_e1'],
                'e2': params['source_e2'],
                'center_x': params['source_center_x'],
                'center_y': params['source_center_y']
            }],
            'kwargs_lens_light': [{
                'amp': params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1': params['light_e1'],
                'e2': params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y']
            }],
            'image_positions': [
                (params.get(f'image_x{i+1}', 0.0),
                 params.get(f'image_y{i+1}', 0.0))
                for i in range(self.n_images)
            ]
        }
        return kw


class ProbModelSourcePlane(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW parameter estimation in source plane.
    
    This model samples source positions and solves the lens equation to get
    image positions, rather than sampling image positions directly.
    """
    
    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None, 
                 solver=None, solver_params=None):
        """Initialize probabilistic model for EM+GW joint inference in source plane.
        
        Args:
            n_images: Number of lensed images (excluding central image)
            gw_observations: Dict with 'time_delays' and 'dL_eff'
            em_observations: Dict with 'data' (observed image)
            lens_image: hcl.LensImage instance (from simulate_em)
            lens_gw: LensImageGW instance (from simulate_gw)
            noise: hcl.Noise instance (created with background_rms=None for inference)
            solver: LensEquationSolver_helens instance (from setup_helens_solver)
            solver_params: Dict of solver parameters (nsolutions, niter, etc.)
        """
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.lens_gw = lens_gw
        self.noise = noise
        self.solver = solver
        # Use default solver params from config if not provided
        self.solver_params = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.pix_scl = 0.4  # Pixel scale in arcsec
        super().__init__()
    
    def model(self):
        """Define the probabilistic model with priors and likelihoods.
        
        This version samples source positions (y0gw, y1gw) and solves the
        lens equation to get image positions.
        """
        # GW parameters
        T_star = numpyro.sample('T_star', dist.Uniform(1e4, 1e8))  # in seconds
        dL = numpyro.sample('dL', dist.Uniform(10000.0, 21800.0))  # in Mpc
        
        # Source light parameters
        source_amp = numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=2.4, high=10.0))
        source_R_sersic = numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))
        source_n = numpyro.sample('source_n', dist.Uniform(1., 2.5))
        source_e1 = numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))
        source_e2 = numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))
        # True source center position (hardcoded)
        source_center_x = jnp.asarray(0.05)
        source_center_y = jnp.asarray(0.1)
        
        prior_source = [{
            'amp': source_amp,
            'R_sersic': source_R_sersic,
            'n_sersic': source_n,
            'e1': source_e1,
            'e2': source_e2,
            'center_x': source_center_x,
            'center_y': source_center_y
        }]
        
        # Lens light parameters
        cx_l = numpyro.sample('light_center_x', dist.Normal(0., self.pix_scl/2))
        cy_l = numpyro.sample('light_center_y', dist.Normal(0., self.pix_scl/2))
        e1_l = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))
        e2_l = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))
        light_amp = numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=9.5))
        light_R_sersic = numpyro.sample('light_R_sersic', dist.TruncatedNormal(1.0, 0.5, low=0.88, high=1.15))
        light_n = numpyro.sample('light_n', dist.Uniform(2.4, 5.))
        
        prior_lens_light = [{
            'amp': light_amp,
            'R_sersic': light_R_sersic,
            'n_sersic': light_n,
            'e1': e1_l,
            'e2': e2_l,
            'center_x': cx_l,
            'center_y': cy_l
        }]
        
        # Lens mass parameters
        lens_theta_E = numpyro.sample('lens_theta_E', dist.Uniform(1.99, 2.01))
        lens_e1 = numpyro.sample('lens_e1', dist.Uniform(-0.065, -0.050))
        lens_e2 = numpyro.sample('lens_e2', dist.Uniform(0.075, 0.11))
        lens_gamma = numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.05))
        # True lens center position (hardcoded)
        lens_center_x = jnp.asarray(0.0)
        lens_center_y = jnp.asarray(0.0)
        gamma1 = numpyro.sample('lens_gamma1', dist.Uniform(-0.006, 0.005))
        gamma2 = numpyro.sample('lens_gamma2', dist.Uniform(-0.005, 0.009))
        
        prior_lens = [
            {
                'theta_E': lens_theta_E,
                'e1': lens_e1,
                'e2': lens_e2,
                'gamma': lens_gamma,
                'center_x': lens_center_x,
                'center_y': lens_center_y
            },
            {
                'gamma1': gamma1,
                'gamma2': gamma2,
                'ra_0': jnp.asarray(0.0),
                'dec_0': jnp.asarray(0.0)
            }
        ]
        
        # Noise parameter
        sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012))
        
        # EM likelihood
        model_params = dict(
            kwargs_lens=prior_lens,
            kwargs_lens_light=prior_lens_light,
            kwargs_source=prior_source
        )
        
        model_image = self.lens_image.model(**model_params)
        em_data = self.em_observations['data']
        
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        model_std = jnp.sqrt(model_var)
        
        numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=em_data)
        
        # Sample GW source position and solve for image positions
        y0gw = numpyro.sample('y0gw', dist.Uniform(0.045, 0.055))
        y1gw = numpyro.sample('y1gw', dist.Uniform(9e-7, 2e-6))
        betas = jnp.array([y0gw, y1gw])
        
        # Solve lens equation to get image positions
        result_thetas, result_betas = self.solver.solve(betas, prior_lens, **self.solver_params)
        
        # Remove central image using the sampled lens center coordinates
        result_theta_x_no_central, result_theta_y_no_central, \
        result_beta_x_no_central, result_beta_y_no_central = remove_central_image(
            result_thetas, result_betas, lens_center_x, lens_center_y
        )
        
        x_pos_array = jnp.array(result_theta_x_no_central)
        y_pos_array = jnp.array(result_theta_y_no_central)
        
        # GW likelihood
        (model_gw, model_time_delays, model_magnifications, model_dL_eff,
         beta_x, beta_y, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL
        )
        
        # GW likelihood terms
        gw_obs = self.gw_observations
        sigma_td = 0.3 * gw_obs['time_delays']
        sigma_dL_eff = 0.3 * gw_obs['dL_eff']
        epsilon = 0.005 * jnp.ones_like(betx_x_diff)
        
        numpyro.sample('tdelays_obs', dist.Independent(dist.Normal(model_time_delays, sigma_td), 1), 
                      obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs', dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1), 
                      obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1), 
                      obs=betx_x_diff)
        numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1), 
                      obs=bety_y_diff)
    
    def params2kwargs(self, params):
        """Convert flat parameter dict to lens_image.model() kwargs format.
        
        Args:
            params: Dictionary of sampled parameters
        
        Returns:
            Dictionary with kwargs_lens, kwargs_source, kwargs_lens_light, etc.
        """
        kw = {
            'kwargs_lens': [
                {
                    'theta_E': params['lens_theta_E'],
                    'e1': params['lens_e1'],
                    'e2': params['lens_e2'],
                    'gamma': params['lens_gamma'],
                    'center_x': params['lens_center_x'],
                    'center_y': params['lens_center_y']
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0': 0.0,
                    'dec_0': 0.0
                }
            ],
            'kwargs_source': [{
                'amp': params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1': params['source_e1'],
                'e2': params['source_e2'],
                'center_x': params['source_center_x'],
                'center_y': params['source_center_y']
            }],
            'kwargs_lens_light': [{
                'amp': params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1': params['light_e1'],
                'e2': params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y']
            }],
            'y0gw': params.get('y0gw', 0.0),
            'y1gw': params.get('y1gw', 0.0)
        }
        return kw


class ProbModelFisher(hcl.NumpyroModel):
    """Probabilistic model with approximate likelihood from Fisher matrix.
    
    This model uses a Taylor expansion (banana model) approximation of the
    log-probability instead of computing the full likelihood.
    """
    
    def __init__(self, keys_to_include, approx_logp):
        """Initialize Fisher model with approximate likelihood.
        
        Args:
            keys_to_include: List of parameter keys to include in the model
            approx_logp: Approximate log-probability function from compute_fisher
        """
        self.keys_to_include = keys_to_include
        self.approx_logp = approx_logp
        self.pix_scl = 0.4  # Pixel scale in arcsec
        super().__init__()
    
    def model(self):
        """Define the probabilistic model with approximate likelihood.
        
        Uses the same priors as ProbModel but replaces the full likelihood
        with the approximate log-probability from Fisher matrix expansion.
        """
        # True image positions (same as ProbModel)
        x_image_true = jnp.array([ 1.90461434, -1.63544685,  0.70943792, -1.14517025])
        y_image_true = jnp.array([-0.90999308,  1.19344445,  1.80599259, -1.50457468])
        
        # Prior distributions (matching ProbModel.model())
        priors = {
            'T_star': lambda: numpyro.sample('T_star', dist.Uniform(1e4, 1e8)),
            'dL': lambda: numpyro.sample('dL', dist.Uniform(10000.0, 21800.0)),
            'source_amp': lambda: numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=2.4, high=10.0)),
            'source_R_sersic': lambda: numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05)),
            'source_n': lambda: numpyro.sample('source_n', dist.Uniform(1., 2.5)),
            'source_e1': lambda: numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
            'source_e2': lambda: numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
            'light_center_x': lambda: numpyro.sample('light_center_x', dist.Normal(0., self.pix_scl/2)),
            'light_center_y': lambda: numpyro.sample('light_center_y', dist.Normal(0., self.pix_scl/2)),
            'light_e1': lambda: numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
            'light_e2': lambda: numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
            'light_amp': lambda: numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=9.5)),
            'light_R_sersic': lambda: numpyro.sample('light_R_sersic', dist.TruncatedNormal(1.0, 0.5, low=0.88, high=1.15)),
            'light_n': lambda: numpyro.sample('light_n', dist.Uniform(2.4, 5.)),
            'lens_theta_E': lambda: numpyro.sample('lens_theta_E', dist.Uniform(1.99, 2.01)),
            'lens_e1': lambda: numpyro.sample('lens_e1', dist.Uniform(-0.065, -0.050)),
            'lens_e2': lambda: numpyro.sample('lens_e2', dist.Uniform(0.075, 0.11)),
            'lens_gamma': lambda: numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.05)),
            'lens_gamma1': lambda: numpyro.sample('lens_gamma1', dist.Uniform(-0.006, 0.005)),
            'lens_gamma2': lambda: numpyro.sample('lens_gamma2', dist.Uniform(-0.005, 0.009)),
            'noise_sigma_bkg': lambda: numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012)),
        }
        
        # Sample parameters in keys_to_include
        param_dict = {}
        delx = jnp.array([0.2, 0.35, 0.49, 0.3])
        dely = jnp.array([0.4, 0.4, 0.35, 0.3])
        
        for key in self.keys_to_include:
            if key in priors:
                param_dict[key] = priors[key]()  # Sample from prior
            elif key.startswith('image_x'):
                i = int(key[-1]) - 1
                mean_x = x_image_true[i]
                minx = mean_x - delx[i]/2
                maxx = mean_x + delx[i]/2
                param_dict[key] = numpyro.sample(key, dist.Uniform(minx, maxx))
            elif key.startswith('image_y'):
                i = int(key[-1]) - 1
                mean_y = y_image_true[i]
                miny = mean_y - dely[i]/2
                maxy = mean_y + dely[i]/2
                param_dict[key] = numpyro.sample(key, dist.Uniform(miny, maxy))
            else:
                raise ValueError(f"Parameter '{key}' in keys_to_include is not recognized. "
                               f"Add it to priors dict or handle image positions.")
        
        # Extract values in correct order and use approximate likelihood
        uarr = jnp.array([param_dict[key] for key in self.keys_to_include])
        numpyro.factor("banana_logprob", self.approx_logp(uarr))

