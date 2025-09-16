#!/usr/bin/env python3
"""
Parameter Estimation Analysis for Gravitational Lensing
Converted from herculens__Starting_guide.ipynb

This script performs a complete parameter estimation analysis including:
- Simulation of an HST-like observation of a strong lens
- Full modeling with smooth profiles
- Fisher Information Matrix (FIM) estimation
- Stochastic Variational Inference (SVI)
- Hamiltonian Monte Carlo (HMC) sampling

All plots and data are automatically saved to the output directory.
"""

import os
import sys
import time
import pickle as pkl
from copy import deepcopy
from pprint import pprint

# Plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
plt.rc('image', interpolation='none')

# Basic imports
import numpy as np
import corner

# JAX
import jax
jax.config.update("jax_enable_x64", True)  # comment for single precision
import jax.numpy as jnp
# optimizers
import optax

# NUTS Hamiltonian MC sampling
import blackjax

# probabilistic model and variational inference
import numpyro
import numpyro.distributions as dist
from numpyro import infer
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

# Herculens
import herculens as hcl
from herculens.Util import param_util, plot_util

def setup_output_directory():
    """Create output directory for saving plots and data"""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_plot(fig, filename, output_dir):
    """Save plot to output directory"""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {filepath}")
    plt.close(fig)

def main():
    """Main analysis function"""
    
    # Setup
    output_dir = setup_output_directory()
    
    # Configuration
    SAVE_SAMPLES_TO_DISK = True
    LOAD_SAMPLES_FROM_DISK = False
    SEED = 87651  # fixes the stochasticity
    
    print("="*80)
    print("GRAVITATIONAL LENSING PARAMETER ESTIMATION ANALYSIS")
    print("="*80)
    
    # ============================================================================
    # 1. SIMULATE AN OBSERVATION OF A STRONG LENS
    # ============================================================================
    
    print("\n1. SIMULATING STRONG LENS OBSERVATION")
    print("-" * 50)
    
    # Define the coordinates grid
    npix = 80  # number of pixel on a side
    pix_scl = 0.08  # pixel size in arcsec
    half_size = npix * pix_scl / 2
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
    transform_pix2angle = pix_scl * np.eye(2)  # transformation matrix pixel <-> angle
    kwargs_pixel = {'nx': npix, 'ny': npix,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                    'transform_pix2angle': transform_pix2angle}

    # create the PixelGrid class
    pixel_grid = hcl.PixelGrid(**kwargs_pixel)
    xgrid, ygrid = pixel_grid.pixel_coordinates
    extent = pixel_grid.extent

    print(f"image size : ({npix}, {npix}) pixels")
    print(f"pixel size : {pix_scl} arcsec")
    print(f"x range    : {xgrid[0, 0], xgrid[0, -1]} arcsec")
    print(f"y range    : {ygrid[0, 0], ygrid[-1, 0]} arcsec")
    
    # Setup point spread function (PSF) and observation/noise properties
    psf = hcl.PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=pix_scl)

    background_rms_simu = 1e-2
    exposure_time_simu = 1e3
    noise_simu = hcl.Noise(npix, npix, background_rms=background_rms_simu, exposure_time=exposure_time_simu)
    noise = hcl.Noise(npix, npix, exposure_time=exposure_time_simu)  # we will sample background_rms
    
    # Lens galaxy - SIE embedded in an external shear
    lens_mass_model_input = hcl.MassModel([hcl.SIE(), hcl.Shear()])

    # position of the lens
    cx0, cy0 = 0., 0.
    # position angle, here in degree
    phi = 8.0
    # axis ratio, b/a
    q = 0.75
    # conversion to ellipticities
    e1, e2 = param_util.phi_q2_ellipticity(phi * np.pi / 180, q)
    # external shear orientation, here in degree
    phi_ext = 54.0
    # external shear strength
    gamma_ext = 0.03 
    # conversion to polar coordinates
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi_ext * np.pi / 180, gamma_ext)
    print(f"Ellipticity: e1={e1:.6f}, e2={e2:.6f}")
    print(f"Shear: gamma1={gamma1:.6f}, gamma2={gamma2:.6f}")
    
    kwargs_lens_input = [
        {'theta_E': 1.5, 'e1': e1, 'e2': e2, 'center_x': cx0, 'center_y': cy0},  # SIE
        {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0.0, 'dec_0': 0.0}  # external shear
    ]

    # Lens light
    lens_light_model_input = hcl.LightModel([hcl.SersicElliptic()])
    kwargs_lens_light_input = [
        {'amp': 8.0, 'R_sersic': 1.0, 'n_sersic': 3., 'e1': e1, 'e2': e2, 'center_x': cx0, 'center_y': cy0}
    ]

    # Source light
    source_model_input = hcl.LightModel([hcl.SersicElliptic()])
    kwargs_source_input = [
        {'amp': 4.0, 'R_sersic': 0.2, 'n_sersic': 2., 'e1': 0.05, 'e2': 0.05, 'center_x': 0.05, 'center_y': 0.1}
    ]
    
    # Generate the lens image
    kwargs_numerics_simu = {'supersampling_factor': 5}
    lens_image_simu = hcl.LensImage(pixel_grid, psf, noise_class=noise_simu,
                             lens_mass_model_class=lens_mass_model_input,
                             source_model_class=source_model_input,
                             lens_light_model_class=lens_light_model_input,
                             kwargs_numerics=kwargs_numerics_simu)

    kwargs_all_input = dict(kwargs_lens=kwargs_lens_input,
                            kwargs_source=kwargs_source_input,
                            kwargs_lens_light=kwargs_lens_light_input)

    # clean image (no noise)
    image = lens_image_simu.model(**kwargs_all_input)

    # simulated observation including noise
    key = jax.random.PRNGKey(SEED)
    key, key_sim = jax.random.split(key)
    data = lens_image_simu.simulation(**kwargs_all_input, compute_true_noise_map=True, prng_key=key_sim)
    
    # Plotting engine
    plotter = hcl.Plotter(flux_vmin=8e-3, flux_vmax=6e-1)
    plotter.set_data(data)
    source_input = lens_image_simu.source_surface_brightness(kwargs_source_input, de_lensed=True, unconvolved=True)
    plotter.set_ref_source(source_input)
    
    # Visualize simulated products
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    img1 = ax1.imshow(image, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(img1)
    ax1.set_title("Clean lensing image")
    img2 = ax2.imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    ax2.set_title("Noisy observation data")
    plot_util.nice_colorbar(img2)
    fig.tight_layout()
    save_plot(fig, "01_simulated_observation.png", output_dir)
    
    # ============================================================================
    # 2. FIT THE IMAGE AND FIND BEST-FIT PARAMETERS
    # ============================================================================
    
    print("\n2. FITTING IMAGE AND FINDING BEST-FIT PARAMETERS")
    print("-" * 50)
    
    kwargs_numerics_fit = {'supersampling_factor': 2}
    lens_image = hcl.LensImage(deepcopy(pixel_grid), deepcopy(psf), noise_class=deepcopy(noise),
                             lens_mass_model_class=deepcopy(lens_mass_model_input),
                             source_model_class=deepcopy(source_model_input),
                             lens_light_model_class=deepcopy(lens_light_model_input),
                             kwargs_numerics=kwargs_numerics_fit)
    
    # Define probabilistic model
    class ProbModel(hcl.NumpyroModel):
        
        def model(self):
            # Parameters of the source
            prior_source = [
              {
                  'amp': numpyro.sample('source_amp', dist.LogNormal(1.0, 0.1)),
             'R_sersic': numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.2, 0.1, low=0.05)), 
             'n_sersic': numpyro.sample('source_n', dist.Uniform(1., 3.)), 
             'e1': numpyro.sample('source_e1', dist.TruncatedNormal(0.0, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('source_e2', dist.TruncatedNormal(0.0, 0.05, low=-0.3, high=0.3)),
             'center_x': numpyro.sample('source_center_x', dist.Normal(0.05, 0.02)), 
            'center_y': numpyro.sample('source_center_y', dist.Normal(0.1, 0.02))}
            ]

            # Parameters of the lens light that are used for the lens mass
            cx = numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))
            cy = numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))
            e1 = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.1, low=-0.3, high=0.3))
            e2 = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.1, low=-0.3, high=0.3))

            # Parameters of the lens light, with center relative the lens mass
            prior_lens_light = [
            {'amp': numpyro.sample('light_amp', dist.LogNormal(2., 0.1)), 
             'R_sersic': numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.1)), 
             'n_sersic': numpyro.sample('light_n', dist.Uniform(2., 5.)), 
             'e1': e1,
             'e2': e2,
             'center_x': cx, 
             'center_y': cy}
            ]

            prior_lens = [
            # power-law
            {
                'theta_E': numpyro.sample('lens_theta_E', dist.Normal(1.5, 0.1)),
             'e1': numpyro.sample('lens_e1', dist.Normal(e1, 0.005)),
             'e2': numpyro.sample('lens_e2', dist.Normal(e2, 0.005)),
             'center_x': numpyro.sample('lens_center_x', dist.Normal(cx, 50.005)), 
             'center_y': numpyro.sample('lens_center_y', dist.Normal(cy, 50.005))},
            # external shear, with fixed origin
            {'gamma1': numpyro.sample('lens_gamma1', dist.TruncatedNormal(0., 0.1, low=-0.3, high=0.3)), 
             'gamma2': numpyro.sample('lens_gamma2', dist.TruncatedNormal(0., 0.1, low=-0.3, high=0.3)), 
             'ra_0': 0.0, 'dec_0': 0.0}
            ]
            
            # wrap up all parameters for the lens_image.model() method
            model_params = dict(kwargs_lens=prior_lens, 
                                kwargs_lens_light=prior_lens_light,
                                kwargs_source=prior_source)
            
            # generates the model image
            model_image = lens_image.model(**model_params)
            
            # estimate the error per pixel
            sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=1e-3, high=1e-1))
            model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
            model_std = jnp.sqrt(model_var)
            
            # finally defines the observed node, conditioned on the data assuming a Gaussian distribution
            numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=data)
        
        def params2kwargs(self, params):
            # functions that takes the flatten dictionary of numpyro parameters
            # and reshape it back to the argument of lens_image.model()
            kw = {'kwargs_lens': [{'theta_E': params['lens_theta_E'],
            'e1': params['lens_e1'],
            'e2': params['lens_e2'],
            'center_x': params['lens_center_x'],
            'center_y': params['lens_center_y']},
            {'gamma1': params['lens_gamma1'],
            'gamma2': params['lens_gamma2'],
            'ra_0': 0.0,
            'dec_0': 0.0}],
            'kwargs_source': [{'amp': params['source_amp'],
            'R_sersic': params['source_R_sersic'],
            'n_sersic': params['source_n'],
            'e1': params['source_e1'],
            'e2': params['source_e2'],
            'center_x': params['source_center_x'],
            'center_y': params['source_center_y']}],
            'kwargs_lens_light': [{'amp': params['light_amp'],
            'R_sersic': params['light_R_sersic'],
            'n_sersic': params['light_n'],
            'e1': params['light_e1'],
            'e2': params['light_e2'],
            'center_x': params['light_center_x'],
            'center_y': params['light_center_y']}]}
            return kw

    prob_model = ProbModel()
    n_param = prob_model.num_parameters
    print(f"Number of parameters: {n_param}")
    
    # Visualize initial guess
    key, key_init = jax.random.split(key)
    init_params = prob_model.get_sample(key_init)  # constrained space
    init_params_unconst = prob_model.unconstrain(init_params)  # UNconstrained space
    kwargs_init = prob_model.params2kwargs(init_params)  # constrained space
    initial_model = lens_image.model(**kwargs_init)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.set_title("Initial guess model")
    im = ax.imshow(initial_model, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)
    ax = axes[1]
    ax.set_title("Simulated data")
    im = ax.imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)
    ax = axes[2]
    ax.set_title("Difference")
    im = ax.imshow(initial_model - data, origin='lower', norm=TwoSlopeNorm(0), cmap=plotter.cmap_res)
    plot_util.nice_colorbar(im)
    fig.tight_layout()
    save_plot(fig, "02_initial_guess.png", output_dir)
    
    # Define the loss function
    loss = hcl.Loss(prob_model)
    print(f"Initial loss = {loss(init_params_unconst)}")
    
    # Set the optimizer to minimize the loss function
    optimizer = hcl.JaxoptOptimizer(loss, loss_norm_optim=data.size)  # loss_norm_optim is to reduce loss magnitude
    best_fit, logL_best_fit, extra_fields, runtime = optimizer.run_scipy(init_params_unconst, method='BFGS', maxiter=600)
    print(f"Optimization runtime: {runtime:.2f} seconds")
    print(f"Final loss = {loss(best_fit)}")

    # Plot optimization history
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(range(len(extra_fields['loss_history'])), extra_fields['loss_history'])
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_title("Optimization History")
    save_plot(fig, "03_optimization_history.png", output_dir)
    
    # Best-fit model visualization
    best_fit_constrained = prob_model.constrain(best_fit)
    kwargs_best_fit = prob_model.params2kwargs(best_fit_constrained)

    fig = plotter.model_summary(lens_image, 
                                kwargs_best_fit, 
                                kwargs_noise={'background_rms': best_fit_constrained['noise_sigma_bkg']},
                                show_source=True)
    save_plot(fig, "04_best_fit_model.png", output_dir)
    
    # Print resulting parameters
    print("\nBEST-FIT VALUES:")
    pprint(kwargs_best_fit)
    print("\n" + "="*80)
    print("INPUT VALUES:")
    pprint(kwargs_all_input)
    
    print(f"\nNoise comparison: Best-fit={best_fit_constrained['noise_sigma_bkg']:.6f}, True={noise_simu.background_rms:.6f}")
    
    # ============================================================================
    # 3. STOCHASTIC VARIATIONAL INFERENCE (SVI)
    # ============================================================================
    
    print("\n3. STOCHASTIC VARIATIONAL INFERENCE (SVI)")
    print("-" * 50)
    
    # create the guide automatically, i.e. the surrogate posterior model
    guide = AutoLowRankMultivariateNormal(
        prob_model.model, 
        init_loc_fn=infer.init_to_median,
    )

    # setup and initialize the optimizer
    schedule_fn = optax.polynomial_schedule(init_value=-1e-6, end_value=-3e-3, 
                                            power=2, transition_steps=300)
    opt = optax.chain(
      optax.scale_by_adam(),
      optax.scale_by_schedule(schedule_fn),
    )
    optimizer_vi = numpyro.optim.optax_to_numpyro(opt)

    svi = SVI(prob_model.model, guide, optimizer_vi, Trace_ELBO(num_particles=20))
    
    num_iter_svi = 10  # Reduced for test run
    key, key_svi = jax.random.split(key)

    print("Running SVI...")
    start = time.time()
    svi_result = svi.run(key_svi, num_iter_svi, stable_update=True, progress_bar=True)
    svi_runtime = time.time() - start
    print(f"SVI runtime: {svi_runtime:.2f} seconds")

    # Check convergence of ELBO loss
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(svi_result.losses)
    ax.set_xlabel("iterations", fontsize=18)
    ax.set_ylabel("loss (ELBO)", fontsize=18)
    ax.set_title("SVI Convergence")
    save_plot(fig, "05_svi_convergence.png", output_dir)
    
    # retrieve samples from the guide posteriors (still in unconstrained space!)
    key, key_svi_post, key_svi_prior = jax.random.split(key, 3)
    svi_samples = guide.sample_posterior(
        key_svi_post, svi_result.params, 
        sample_shape=(1000,)  # Increased for better statistics
    )

    # for comparison we also retrieve samples from the prior (i.e. before VI)
    prior_samples = Predictive(prob_model.model, num_samples=1000)(key_svi_prior)  # Increased for better statistics
    del prior_samples['obs']
    
    # Plot prior distribution
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(prior_samples['lens_center_x'], bins=50, alpha=0.7, density=True)
    ax.set_xlabel('lens_center_x')
    ax.set_ylabel('Density')
    ax.set_title('Prior Distribution (lens_center_x)')
    save_plot(fig, "06_prior_distribution.png", output_dir)
    
    # ============================================================================
    # 4. FISHER INFORMATION MATRIX (FIM)
    # ============================================================================
    
    print("\n4. FISHER INFORMATION MATRIX (FIM)")
    print("-" * 50)
    
    @jax.jit
    def loss_constrained(params_const):
        params_unconst = prob_model.unconstrain(params_const)
        return loss(params_unconst)

    @jax.jit
    def hessian_constrained(params_const):
        return jax.jacfwd(jax.jacrev(loss_constrained))(params_const)

    print("Computing Fisher Information Matrix...")
    fisher_matrix = hessian_constrained(prob_model.constrain(best_fit))  # pytree
    fisher_matrix, _ = jax.flatten_util.ravel_pytree(fisher_matrix)  # get the array
    fisher_matrix = fisher_matrix.reshape((22, 22))  # reshape as a matrix
    cov_matrix = jnp.linalg.inv(fisher_matrix) # invert to get covariance matrix
    
    print(f"Fisher matrix determinant: {jnp.linalg.det(fisher_matrix):.2e}")
    
    # Plot covariance matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(cov_matrix, origin='upper', cmap='coolwarm', 
                    norm=SymLogNorm(linthresh=1e-5, linscale=1, vmin=-1e-1, vmax=1e-1))
    ax.set_title("Covariance Matrix for Best-fit Parameters")
    plot_util.nice_colorbar(im)
    save_plot(fig, "07_covariance_matrix.png", output_dir)
    
    # Generate FIM samples
    mean, unravel_fn = jax.flatten_util.ravel_pytree(prob_model.constrain(best_fit))
    cov = cov_matrix
    fim_samples = jax.vmap(unravel_fn)(np.random.multivariate_normal(mean, cov, size=1000))  # Increased for better statistics
    
    # ============================================================================
    # 5. HAMILTONIAN MONTE CARLO (HMC)
    # ============================================================================
    
    print("\n5. HAMILTONIAN MONTE CARLO (HMC)")
    print("-" * 50)
    
    @jax.jit
    def logdensity_fn(args):
        return -loss(args)

    adapt = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, 
        target_acceptance_rate=0.8, 
        is_mass_matrix_diagonal=True,
        initial_step_size=1.0,
        progress_bar=True, 
    )

    num_steps_adaptation = 5  # Reduced for test run
    key, key_hmc_adapt, key_hmc_run = jax.random.split(key, 3)

    print("Running HMC adaptation...")
    start = time.time()
    (last_state, adapted_settings), info = adapt.run(key_hmc_adapt, best_fit, 
                                                     num_steps=num_steps_adaptation)
    hmc_adapt_runtime = time.time() - start
    print(f"HMC adaptation runtime: {hmc_adapt_runtime:.2f} seconds")
    print("Warmup state (lens only, unconstrained):", {k: v for k, v in last_state.position.items() if 'lens' in k})
    
    # Visualize the inverse mass matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(adapted_settings['inverse_mass_matrix'])
    ax.set_title("Adapted Inverse Mass Matrix (Diagonal Elements)")
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Inverse Mass")
    save_plot(fig, "08_hmc_mass_matrix.png", output_dir)
    
    # Setup the NUTS kernel with adapted settings
    kernel = blackjax.nuts(logdensity_fn, **adapted_settings).step

    # Define the inference loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    num_steps_nuts = 10  # Reduced for test run

    print("Running HMC sampling...")
    start = time.time()
    states, infos = inference_loop(key_hmc_run, kernel, last_state, num_steps_nuts)
    _ = states.position['lens_theta_E'].block_until_ready()
    hmc_sample_runtime = time.time() - start
    print(f"HMC sampling runtime: {hmc_sample_runtime:.2f} seconds")
    
    # Plot HMC diagnostics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Energy plot
    ax1.plot(infos.energy)
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("energy")
    ax1.set_title("HMC Energy")
    
    # Momentum plot
    for k, v in infos.momentum.items():
        if 'lens' in k:
            ax2.plot((v - np.median(v)) / np.median(v), label=k, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("momentum (normalized)")
    ax2.set_title("HMC Momentum")
    
    plt.tight_layout()
    save_plot(fig, "09_hmc_diagnostics.png", output_dir)
    
    acceptance_rate = np.mean(infos.acceptance_rate)
    num_divergent = np.mean(infos.is_divergent)

    print(f"Average acceptance rate: {acceptance_rate:.2f}")
    print(f"There were {100*num_divergent:.2f}% divergent transitions")
    
    # Retrieve the HMC posterior samples
    hmc_samples = jax.vmap(prob_model.constrain)(states.position)
    
    # ============================================================================
    # 6. COMPARE POSTERIOR DISTRIBUTIONS
    # ============================================================================
    
    print("\n6. COMPARING POSTERIOR DISTRIBUTIONS")
    print("-" * 50)
    
    # Create input parameters for reference
    input_params = {
        'lens_theta_E': kwargs_all_input['kwargs_lens'][0]['theta_E'],
        'lens_e1': kwargs_all_input['kwargs_lens'][0]['e1'],
        'lens_e2': kwargs_all_input['kwargs_lens'][0]['e2'],
        'lens_center_x': kwargs_all_input['kwargs_lens'][0]['center_x'],
        'lens_center_y': kwargs_all_input['kwargs_lens'][0]['center_y'],
        'lens_gamma1': kwargs_all_input['kwargs_lens'][1]['gamma1'],
        'lens_gamma2': kwargs_all_input['kwargs_lens'][1]['gamma2'],
        'light_amp': kwargs_all_input['kwargs_lens_light'][0]['amp'],
        'light_R_sersic': kwargs_all_input['kwargs_lens_light'][0]['R_sersic'],
        'light_n': kwargs_all_input['kwargs_lens_light'][0]['n_sersic'],
        'light_e1': kwargs_all_input['kwargs_lens_light'][0]['e1'],
        'light_e2': kwargs_all_input['kwargs_lens_light'][0]['e2'],
        'light_center_x': kwargs_all_input['kwargs_lens_light'][0]['center_x'],
        'light_center_y': kwargs_all_input['kwargs_lens_light'][0]['center_y'],
        'source_amp': kwargs_all_input['kwargs_source'][0]['amp'],
        'source_R_sersic': kwargs_all_input['kwargs_source'][0]['R_sersic'],
        'source_n': kwargs_all_input['kwargs_source'][0]['n_sersic'],
        'source_e1': kwargs_all_input['kwargs_source'][0]['e1'],
        'source_e2': kwargs_all_input['kwargs_source'][0]['e2'],
        'source_center_x': kwargs_all_input['kwargs_source'][0]['center_x'],
        'source_center_y': kwargs_all_input['kwargs_source'][0]['center_y'],
        'noise_sigma_bkg': background_rms_simu,
    }
    
    # Save samples to disk
    if SAVE_SAMPLES_TO_DISK:
        samples_to_disk = {
            'prior_samples': prior_samples,
            'fim_samples': fim_samples,
            'svi_samples': svi_samples,
            'hmc_samples': hmc_samples,
            'input_params': input_params,
            'best_fit_params': best_fit_constrained,
            'kwargs_best_fit': kwargs_best_fit,
            'kwargs_all_input': kwargs_all_input,
            'fisher_matrix': fisher_matrix,
            'cov_matrix': cov_matrix,
            'svi_result': svi_result,
            'hmc_info': {
                'acceptance_rate': acceptance_rate,
                'divergent_rate': num_divergent,
                'adaptation_runtime': hmc_adapt_runtime,
                'sampling_runtime': hmc_sample_runtime
            }
        }
        
        with open(os.path.join(output_dir, 'samples_all.pkl'), 'wb') as f:
            pkl.dump(samples_to_disk, f)
        print(f"Saved all samples and results to: {os.path.join(output_dir, 'samples_all.pkl')}")
    
    # Format labels and markers for corner plot
    var_names_and_ranges = {
        'lens_theta_E': 0.3, 'lens_gamma1': 0.3, 'lens_gamma2': 0.3, 'lens_e1': 0.3, 'lens_e2':0.3, 
        'source_R_sersic': 0.3, 'source_n': 0.3, 'source_center_x': 0.3, 'source_center_y': 0.3,
        'light_R_sersic': 0.3, 'light_center_x': 0.3, 'light_center_y': 0.3,
        'noise_sigma_bkg': 0.3,
    }
    var_names = list(var_names_and_ranges.keys())
    ranges = list(var_names_and_ranges.values())

    # Create corner plot comparing all methods
    fig = None
    fig = corner.corner(fim_samples, 
                        color='tab:orange', 
                        var_names=var_names,
                        fill_contours=True,
                        plot_datapoints=False,
                        hist_kwargs=dict(label="Fisher matrix estimate", linewidth=2),
                        fig=fig)
    fig = corner.corner(svi_samples, 
                        color='tab:blue', 
                        var_names=var_names,
                        fill_contours=True,
                        plot_datapoints=False,
                        hist_kwargs=dict(label="Variational inference", linewidth=2),
                        fig=fig)
    fig = corner.corner(hmc_samples, 
                        color='tab:red', 
                        var_names=var_names,
                        fill_contours=True,
                        truths=input_params, truth_color='black', 
                        hist_kwargs=dict(label="Hamiltonian Monte Carlo", linewidth=2),
                        fig=fig)
    axes = np.array(fig.get_axes()).reshape(len(var_names), len(var_names))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, len(var_names)-1].legend(handles, labels, fontsize=18)
    fig.suptitle("Posterior Distribution Comparison", fontsize=16)
    save_plot(fig, "10_posterior_comparison.png", output_dir)
    
    # Print sample shapes
    print(f"Sample shapes:")
    print(f"  Prior: {prior_samples['lens_theta_E'].shape}")
    print(f"  FIM: {fim_samples['lens_theta_E'].shape}")
    print(f"  SVI: {svi_samples['lens_theta_E'].shape}")
    print(f"  HMC: {hmc_samples['lens_theta_E'].shape}")
    
    # ============================================================================
    # 7. SUMMARY AND FINAL OUTPUTS
    # ============================================================================
    
    print("\n7. ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {n_param}")
    print(f"Optimization runtime: {runtime:.2f} seconds")
    print(f"SVI runtime: {svi_runtime:.2f} seconds")
    print(f"HMC adaptation runtime: {hmc_adapt_runtime:.2f} seconds")
    print(f"HMC sampling runtime: {hmc_sample_runtime:.2f} seconds")
    print(f"Total runtime: {runtime + svi_runtime + hmc_adapt_runtime + hmc_sample_runtime:.2f} seconds")
    print(f"\nHMC diagnostics:")
    print(f"  Acceptance rate: {acceptance_rate:.2f}")
    print(f"  Divergent transitions: {100*num_divergent:.2f}%")
    
    print(f"\nAll plots and data saved to: {output_dir}/")
    print("Files generated:")
    for i in range(1, 11):
        print(f"  {i:02d}_*.png - Various analysis plots")
    print(f"  samples_all.pkl - All samples and results")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
