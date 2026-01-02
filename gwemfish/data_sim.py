"""
EM and GW data simulation functions.

This module provides functions to set up observation infrastructure
(pixel grid, PSF, noise) and simulate EM and GW observations.
"""

import jax.numpy as jnp
import jax.random as random
import herculens as hcl
from copy import deepcopy
import sys
import os

# Import lensimage_gw from scripts directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
import lensimage_gw
from .config import arcsecond_to_radians, Mpc_to_m, c


def setup_pixel_grid(npix=20, pix_scl=0.4):
    """Setup pixel grid for EM observations.
    
    Args:
        npix: Number of pixels on a side (default: 20)
        pix_scl: Pixel size in arcsec (default: 0.4)
    
    Returns: 
        pixel_grid (hcl.PixelGrid instance)
    """
    half_size = npix * pix_scl / 2
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2
    transform_pix2angle = pix_scl * jnp.eye(2)
    
    kwargs_pixel = {
        'nx': npix, 
        'ny': npix,
        'ra_at_xy_0': ra_at_xy_0, 
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle
    }
    
    pixel_grid = hcl.PixelGrid(**kwargs_pixel)
    return pixel_grid


def setup_psf(psf_type='GAUSSIAN', fwhm=0.2, pixel_size=0.4):
    """Setup PSF for EM observations.
    
    Args:
        psf_type: PSF type (default: 'GAUSSIAN')
        fwhm: Full width at half maximum in arcsec (default: 0.2)
        pixel_size: Pixel size in arcsec (default: 0.4)
    
    Returns: 
        psf (hcl.PSF instance)
    """
    psf = hcl.PSF(psf_type=psf_type, fwhm=fwhm, pixel_size=pixel_size)
    return psf


def setup_noise(npix=20, background_rms=None, exposure_time=1e3):
    """Setup noise class for EM observations.
    
    For simulation: pass background_rms to create noise with fixed variance.
    For inference: pass background_rms=None, then use noise.C_D_model() with 
    sampled background_rms values during MCMC.
    
    Args:
        npix: Number of pixels per side (default: 20)
        background_rms: Background RMS (None for inference, value for simulation)
        exposure_time: Exposure time (default: 1e3)
    
    Returns: 
        noise (hcl.Noise instance)
    
    Note: During inference, noise.C_D_model(model_image, background_rms=sigma_bkg)
    is called with different sigma_bkg values sampled at each MCMC step.
    """
    noise = hcl.Noise(npix, npix, background_rms=background_rms, 
                      exposure_time=exposure_time)
    return noise


def simulate_em(kwargs_lens, kwargs_source, kwargs_lens_light,
                lens_mass_model, source_model_class, lens_light_model_class,
                pixel_grid, psf, noise_class, seed=None, 
                kwargs_numerics=None, exposure_time=1e3):
    """Simulate EM observation with noise.
    
    Args:
        kwargs_lens: List of lens mass model kwargs
        kwargs_source: List of source light model kwargs
        kwargs_lens_light: List of lens light model kwargs
        lens_mass_model: hcl.MassModel instance
        source_model_class: hcl.LightModel instance for source
        lens_light_model_class: hcl.LightModel instance for lens light
        pixel_grid: hcl.PixelGrid instance
        psf: hcl.PSF instance
        noise_class: hcl.Noise instance (with background_rms for simulation)
        seed: Random seed for noise (default: None)
        kwargs_numerics: Optional numerics kwargs (e.g., {'supersampling_factor': 1})
        exposure_time: Exposure time for inference noise (default: 1e3)
    
    Returns: 
        em_obs: dict with 'data' key containing the observed image
        lens_image: hcl.LensImage instance (for inference)
    """
    if kwargs_numerics is None:
        kwargs_numerics = {'supersampling_factor': 1}
    
    # Create lens image instance for simulation
    lens_image_simu = hcl.LensImage(
        pixel_grid, psf, noise_class=noise_class,
        lens_mass_model_class=lens_mass_model,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )
    
    # Prepare all kwargs
    kwargs_all_input = dict(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light
    )
    
    # Generate simulated observation with noise
    if seed is not None:
        key = random.PRNGKey(seed)
        key, key_sim = random.split(key)
    else:
        key_sim = random.PRNGKey(42)  # default seed
    
    data = lens_image_simu.simulation(
        **kwargs_all_input, 
        compute_true_noise_map=True, 
        prng_key=key_sim
    )
    
    em_obs = {'data': data}
    
    # Create lens image instance for inference (with noise without background_rms)
    # num_pixel is total pixels (e.g., 400 for 20x20 grid), compute per side
    import math
    npix_per_side = int(math.sqrt(pixel_grid.num_pixel))
    npix_x = npix_y = npix_per_side
    
    # Create noise instance for inference (hcl.Noise doesn't store exposure_time as attribute)
    noise_inference = hcl.Noise(
        npix_x, 
        npix_y, 
        exposure_time=exposure_time
    )
    
    lens_image = hcl.LensImage(
        deepcopy(pixel_grid), 
        deepcopy(psf), 
        noise_class=noise_inference,
        lens_mass_model_class=deepcopy(lens_mass_model),
        source_model_class=deepcopy(source_model_class),
        lens_light_model_class=deepcopy(lens_light_model_class),
        kwargs_numerics=kwargs_numerics
    )
    
    return em_obs, lens_image


def simulate_gw(source_pos, kwargs_lens, lens_mass_model, 
                cosmology, zl, zs, lens_model_list=None, solver_params=None):
    """Simulate GW observation data by solving lens equation from source position.
    
    This function takes a source position, solves the lens equation to find
    image positions, then computes GW observables (time delays, magnifications, etc.).
    
    Args:
        source_pos: Tuple (x, y) of source position in arcsec
        kwargs_lens: List of lens mass model kwargs
        lens_mass_model: hcl.MassModel instance
        cosmology: Cosmology instance (e.g., JAXCosmology)
        zl: Lens redshift
        zs: Source redshift
        lens_model_list: List of lens model names (e.g., ['EPL', 'SHEAR']). 
                        If None, will try to infer from lens_mass_model.
        solver_params: Optional solver parameters dict. If None, uses defaults.
    
    Returns: 
        x_image: Array of image x positions (arcsec)
        y_image: Array of image y positions (arcsec)
        gw_obs: dict with 'time_delays' and 'dL_eff'
        data_GW: dict with full GW data including 'Tstar_in_seconds', 'phi_in_arcsecsq', etc.
        lens_gw: LensImageGW instance (for inference)
    """
    # Solve lens equation to get image positions
    from .lens_setup import setup_lens
    
    # If lens_model_list not provided, try to get it from mass_model
    if lens_model_list is None:
        # Try to get from mass_model if it has the attribute
        if hasattr(lens_mass_model, 'lens_model_list'):
            lens_model_list = lens_mass_model.lens_model_list
        else:
            raise ValueError("lens_model_list must be provided if lens_mass_model doesn't have lens_model_list attribute")
    
    _, x_image, y_image, _ = setup_lens(
        lens_model_list, 
        kwargs_lens, 
        zl, 
        zs, 
        source_pos,
        solver_params=solver_params
    )
    
    # Compute time delay distance
    time_delay_distance = cosmology.time_delay_distance(zl, zs)
    
    # Create lens_gw instance
    lens_gw = lensimage_gw.LensImageGW(lens_mass_model)
    
    # Compute GW observables
    data_GW = lens_gw.compute(x_image, y_image, kwargs_lens, time_delay_distance)
    
    # Compute luminosity distance and effective distances
    dL = cosmology.luminosity_distance(zs)
    magnifications = data_GW['mu']
    dL_eff = dL / jnp.sqrt(jnp.abs(magnifications))
    time_delays = data_GW['time_delays_in_seconds']
    
    # Create gw_obs dict
    gw_obs = {
        'time_delays': time_delays,
        'dL_eff': dL_eff
    }
    
    return x_image, y_image, gw_obs, data_GW, lens_gw


def compute_gw_from_images(x_image, y_image, kwargs_lens, lens_gw, 
                          T_star, dL):
    """Compute GW observables from image positions (for use during HMC inference).
    
    This function is used inside ProbModel.model() during inference when
    image positions are sampled. It computes GW observables directly from
    image positions without solving the lens equation. D_dt is computed
    from T_star internally: D_dt = (T_star*c)/(Mpc_to_m*arcsecond_to_radians**2)
    
    This matches the usage: model_gw = lens_gw.compute(x_pos_array, y_pos_array, 
                                                         prior_lens, D_dt)
    
    Args:
        x_image: Array of image x positions (arcsec) - sampled during inference
        y_image: Array of image y positions (arcsec) - sampled during inference
        kwargs_lens: List of lens mass model kwargs - sampled during inference
        lens_gw: LensImageGW instance (from simulate_gw)
        T_star: Characteristic time scale (seconds) - sampled during inference
        dL: Luminosity distance (Mpc) - sampled during inference
    
    Returns:
        model_gw: dict from lens_gw.compute() with keys:
            - 'beta_x', 'beta_y': Ray-shooted source positions
            - 'psi': Lens potential
            - 'mu': Magnifications
            - 'phi_in_arcsecsq': Fermat potential
            - 'Tstar_in_seconds': Characteristic time scale
            - 'tarrivals_in_seconds': Arrival times
            - 'time_delays_in_seconds': Time delays between images
            - 'time_delays_in_days': Time delays in days
        model_time_delays: Array of time delays (seconds) computed as 
                          abs(diff(T_star * phi_in_arcsecsq))
        model_magnifications: Array of magnifications from model_gw['mu']
        model_dL_eff: Array of effective luminosity distances computed as 
                     dL / sqrt(abs(magnifications))
        beta_x: Ray-shooted source x positions (for consistency check)
        beta_y: Ray-shooted source y positions (for consistency check)
        betx_x_diff: Difference in beta_x between images (for likelihood)
        bety_y_diff: Difference in beta_y between images (for likelihood)
    """
    # Compute D_dt from T_star
    D_dt = (T_star * c) / (Mpc_to_m * arcsecond_to_radians**2)  # in Mpc
    
    # Compute GW observables
    model_gw = lens_gw.compute(x_image, y_image, kwargs_lens, D_dt)
    
    # Extract quantities
    beta_x = model_gw['beta_x']
    beta_y = model_gw['beta_y']
    
    # Compute differences for consistency check
    betx_x_diff = jnp.diff(beta_x)
    bety_y_diff = jnp.diff(beta_y)
    
    # Compute time delays from arrival times
    model_arrival_time = T_star * model_gw['phi_in_arcsecsq']
    model_time_delays = jnp.abs(jnp.diff(model_arrival_time))
    
    # Extract magnifications and compute effective distances
    model_magnifications = model_gw['mu']
    model_dL_eff = dL / jnp.sqrt(jnp.abs(model_magnifications))
    
    return (model_gw, model_time_delays, model_magnifications, model_dL_eff,
            beta_x, beta_y, betx_x_diff, bety_y_diff)

