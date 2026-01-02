"""
Simple usage example for GWEMFISH pipeline.
This script matches the structure of example_notebook.ipynb
"""

import sys
import os
import jax
import jax.numpy as jnp
from jaxcosmo import JAXCosmology
from gwemfish import (
    setup_jax,
    setup_lens,
    setup_pixel_grid, setup_psf, setup_noise,
    simulate_em, simulate_gw,
    ProbModel, ProbModelSourcePlane, ProbModelFisher,
    setup_helens_solver,
    run_mcmc,
    compute_fisher,
    DEFAULT_LENS_MODEL_LIST, DEFAULT_KWARGS_LENS,
    DEFAULT_ZL, DEFAULT_ZS, DEFAULT_SOURCE_POS_EM, DEFAULT_SOURCE_POS_GW,
    DEFAULT_KWARGS_SOURCE, DEFAULT_KWARGS_LENS_LIGHT,
    DEFAULT_PIXEL_GRID_KWARGS, DEFAULT_PSF_KWARGS,
    DEFAULT_NOISE_KWARGS_SIMU, DEFAULT_NOISE_KWARGS_INFERENCE,
    DEFAULT_KWARGS_NUMERICS,
    DEFAULT_SOURCE_LIGHT_MODEL, DEFAULT_LENS_LIGHT_MODEL
)
import herculens as hcl
import lensimage_gw
from herculens.Util import param_util, plot_util
import matplotlib.pyplot as plt

# ============================================================================
# 0. Setup JAX configuration (do this first!)
# ============================================================================
NCPUS = 8
os.environ['OMP_NUM_THREADS'] = str(NCPUS)
os.environ['MKL_NUM_THREADS'] = str(NCPUS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NCPUS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NCPUS)

# Set JAX configuration
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={NCPUS}'

# Configure JAX using setup_jax
setup_jax(ncpus=NCPUS, enable_x64=True, platform='cpu', verbose=True)

print("=" * 60)
print(f"Requested CPU cores: {NCPUS}")
print(f"Thread limits: OMP={os.environ['OMP_NUM_THREADS']}")
print(f"JAX configuration: X64={os.environ['JAX_ENABLE_X64']}, Platform={os.environ['JAX_PLATFORM_NAME']}")
print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
devices = jax.devices()
print(f"JAX devices: {devices}")
print("=" * 60)

# ============================================================================
# 1. Setup lens and get image positions
# ============================================================================
kwargs_lens, x_image_true, y_image_true, lens_mass_model = setup_lens(
    lens_model_list=DEFAULT_LENS_MODEL_LIST,
    kwargs_lens=DEFAULT_KWARGS_LENS,
    zl=DEFAULT_ZL,
    zs=DEFAULT_ZS,
    source_pos=DEFAULT_SOURCE_POS_GW  # GW source position
)

# Optional: Plot lens model with lenstronomy
try:
    from lenstronomy.Plots import lens_plot
    from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel
    
    lens_model_plot = LenstronomyLensModel(DEFAULT_LENS_MODEL_LIST, z_lens=DEFAULT_ZL, z_source=DEFAULT_ZS)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    lens_plot.lens_model_plot(
        ax, lensModel=lens_model_plot,
        kwargs_lens=kwargs_lens,
        sourcePos_x=DEFAULT_SOURCE_POS_GW[0], sourcePos_y=DEFAULT_SOURCE_POS_GW[1],
        point_source=True, with_caustics=True, fast_caustic=True,
        numPix=600, deltaPix=0.01, cmap_string="RdPu"
    )
    
    # Ensure all images and collections use the desired colormap
    cmap_string = "RdPu"
    for obj in list(ax.get_images()) + list(ax.collections):
        if hasattr(obj, 'set_cmap'):
            obj.set_cmap(cmap_string)
    
    plt.title("Lens System with EPL + SHEAR Model")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("lenstronomy not available, skipping lens plot")

# ============================================================================
# 2. Setup cosmology
# ============================================================================
cosmology = JAXCosmology(H0=67.3, Om0=0.316)

# ============================================================================
# 3. Setup EM observation components
# ============================================================================
pixel_grid = setup_pixel_grid(**DEFAULT_PIXEL_GRID_KWARGS)
psf = setup_psf(**DEFAULT_PSF_KWARGS)
noise_simu = setup_noise(**DEFAULT_NOISE_KWARGS_SIMU)
noise_inf = setup_noise(**DEFAULT_NOISE_KWARGS_INFERENCE)
exposure_time = DEFAULT_NOISE_KWARGS_INFERENCE['exposure_time']

# ============================================================================
# 4. Setup light models
# ============================================================================
source_model = DEFAULT_SOURCE_LIGHT_MODEL()  # or: hcl.LightModel([hcl.SersicElliptic()])
lens_light_model = DEFAULT_LENS_LIGHT_MODEL()  # or: hcl.LightModel([hcl.SersicElliptic()])

# ============================================================================
# 5. Simulate EM data
# ============================================================================
em_obs, lens_image = simulate_em(
    kwargs_lens=kwargs_lens,
    kwargs_source=DEFAULT_KWARGS_SOURCE,
    kwargs_lens_light=DEFAULT_KWARGS_LENS_LIGHT,
    lens_mass_model=lens_mass_model,
    source_model_class=source_model,
    lens_light_model_class=lens_light_model,
    pixel_grid=pixel_grid,
    psf=psf,
    noise_class=noise_simu,
    kwargs_numerics=DEFAULT_KWARGS_NUMERICS,
    exposure_time=exposure_time,
    seed=87651
)

# ============================================================================
# 6. Plotting with herculens Plotter
# ============================================================================
# Plotting engine
plotter = hcl.Plotter(flux_vmin=8e-3, flux_vmax=6e-1)

# inform the plotter of the data and, if any, the true source 
plotter.set_data(em_obs['data'])

# Compute source surface brightness
source_input = lens_image.source_surface_brightness(DEFAULT_KWARGS_SOURCE, de_lensed=True, unconvolved=True)
plotter.set_ref_source(source_input)

# ============================================================================
# 7. Visualize simulated products using the image grid xx and yy and scatter image positions
# ============================================================================
xx, yy = pixel_grid.pixel_coordinates

# Generate clean image (no noise)
image_clean = lens_image.model(
    kwargs_lens=kwargs_lens,
    kwargs_source=DEFAULT_KWARGS_SOURCE,
    kwargs_lens_light=DEFAULT_KWARGS_LENS_LIGHT
)

# Get noisy data
data = em_obs['data']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot clean image
img1 = ax1.pcolormesh(xx, yy, image_clean, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img1)
ax1.set_title("Clean lensing image (RA/Dec)")
ax1.set_xlabel("RA [arcsec]")
ax1.set_ylabel("Dec [arcsec]")
# Scatter the true image positions
ax1.scatter(x_image_true, y_image_true, color='k', marker='x', s=60, label='GW')
legend = ax1.legend()
for text in legend.get_texts():
    text.set_color('white')

# Plot noisy data
img2 = ax2.pcolormesh(xx, yy, data, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img2)
ax2.set_title("Noisy observation data (RA/Dec)")
ax2.set_xlabel("RA [arcsec]")
ax2.set_ylabel("Dec [arcsec]")
# Scatter the true image positions
ax2.scatter(x_image_true, y_image_true, color='k', marker='x', s=60, label='GW')
legend = ax2.legend()
for text in legend.get_texts():
    text.set_color('white')

fig.tight_layout()
plt.show()

# Get pixel coordinate of gw images
x_pix_gw, y_pix_gw = pixel_grid.map_coord2pix(x_image_true, y_image_true)

# visualize simulated products in pixel coordinates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
img1 = ax1.imshow(image_clean, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img1)
ax1.set_title("Clean lensing image")
ax1.scatter(x_pix_gw, y_pix_gw, color='black', marker='x', s=60, label='GW')
ax1.legend()
img2 = ax2.imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
ax2.set_title("Noisy observation data")
plot_util.nice_colorbar(img2)
ax2.scatter(x_pix_gw, y_pix_gw, color='black', marker='x', s=60, label='GW')
ax2.legend()
fig.tight_layout()
plt.show()

# ============================================================================
# 8. Simulate GW data
# ============================================================================
x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(
    source_pos=DEFAULT_SOURCE_POS_GW,
    kwargs_lens=kwargs_lens,
    lens_mass_model=lens_mass_model,
    cosmology=cosmology,
    zl=DEFAULT_ZL,
    zs=DEFAULT_ZS,
    lens_model_list=DEFAULT_LENS_MODEL_LIST
)

# ============================================================================
# 9. Run inference - Image Plane (directly sample image positions)
# ============================================================================
probmodel = ProbModel(
    n_images=4,
    gw_observations=gw_obs,
    em_observations=em_obs,
    lens_image=lens_image,
    lens_gw=lens_gw,
    noise=noise_inf
)

samples, summary, extra_fields, mcmc = run_mcmc(
    probmodel.model,
    num_warmup=6000,
    num_samples=12000,
    num_chains=2
)

# ============================================================================
# 10. Create complete input parameter dictionary (for truth values)
# ============================================================================
# Compute luminosity distance from source redshift using jaxcosmo
dL_true = cosmology.luminosity_distance(DEFAULT_ZS)

input_params = {
    # ========================================================================
    # Cosmology and Redshifts (Fixed)
    # ========================================================================
    'zs': DEFAULT_ZS,  # 2.0 - Source redshift
    'zl': DEFAULT_ZL,  # 0.5 - Lens redshift
    
    # ========================================================================
    # Lens Mass Model Parameters
    # ========================================================================
    'lens_theta_E': DEFAULT_KWARGS_LENS[0]['theta_E'],  # 2.0 arcsec - Einstein radius
    'lens_e1': DEFAULT_KWARGS_LENS[0]['e1'],  # Computed from phi=60°, q=0.8
    'lens_e2': DEFAULT_KWARGS_LENS[0]['e2'],  # Computed from phi=60°, q=0.8
    'lens_gamma': DEFAULT_KWARGS_LENS[0]['gamma'],  # 2.0 - Power-law slope (EPL)
    'lens_center_x': DEFAULT_KWARGS_LENS[0]['center_x'],  # 0.0 - Lens center x (fixed)
    'lens_center_y': DEFAULT_KWARGS_LENS[0]['center_y'],  # 0.0 - Lens center y (fixed)
    'lens_gamma1': DEFAULT_KWARGS_LENS[1]['gamma1'],  # 0.0 - External shear component 1
    'lens_gamma2': DEFAULT_KWARGS_LENS[1]['gamma2'],  # 0.0 - External shear component 2
    
    # ========================================================================
    # Source Light Model Parameters (Sersic)
    # ========================================================================
    'source_amp': DEFAULT_KWARGS_SOURCE[0]['amp'],  # 4.0 - Source amplitude
    'source_R_sersic': DEFAULT_KWARGS_SOURCE[0]['R_sersic'],  # 0.5 - Source Sersic radius
    'source_n': DEFAULT_KWARGS_SOURCE[0]['n_sersic'],  # 2.0 - Source Sersic index
    'source_e1': DEFAULT_KWARGS_SOURCE[0]['e1'],  # 0.05 - Source ellipticity component 1
    'source_e2': DEFAULT_KWARGS_SOURCE[0]['e2'],  # 0.05 - Source ellipticity component 2
    'source_center_x': DEFAULT_KWARGS_SOURCE[0]['center_x'],  # 0.05 - Source center x
    'source_center_y': DEFAULT_KWARGS_SOURCE[0]['center_y'],  # 0.1 - Source center y
    
    # ========================================================================
    # Lens Light Model Parameters (Sersic)
    # ========================================================================
    'light_amp': DEFAULT_KWARGS_LENS_LIGHT[0]['amp'],  # 8.0 - Lens light amplitude
    'light_R_sersic': DEFAULT_KWARGS_LENS_LIGHT[0]['R_sersic'],  # 1.0 - Lens light Sersic radius
    'light_n': DEFAULT_KWARGS_LENS_LIGHT[0]['n_sersic'],  # 3.0 - Lens light Sersic index
    'light_e1': DEFAULT_KWARGS_LENS_LIGHT[0]['e1'],  # e1_true - Lens light ellipticity component 1
    'light_e2': DEFAULT_KWARGS_LENS_LIGHT[0]['e2'],  # e2_true - Lens light ellipticity component 2
    'light_center_x': DEFAULT_KWARGS_LENS_LIGHT[0]['center_x'],  # 0.0 - Lens light center x
    'light_center_y': DEFAULT_KWARGS_LENS_LIGHT[0]['center_y'],  # 0.0 - Lens light center y
    
    # ========================================================================
    # Gravitational Wave Source Position
    # ========================================================================
    'y0gw': DEFAULT_SOURCE_POS_GW[0],  # 0.05 - GW source position x (arcsec)
    'y1gw': DEFAULT_SOURCE_POS_GW[1],  # 1e-6 - GW source position y (arcsec)
    
    # ========================================================================
    # Image Positions (Fixed - 4 images)
    # ========================================================================
    'image_x1': float(x_img_gw[0]),  # Image 1 x position
    'image_y1': float(y_img_gw[0]),  # Image 1 y position
    'image_x2': float(x_img_gw[1]),  # Image 2 x position
    'image_y2': float(y_img_gw[1]),  # Image 2 y position
    'image_x3': float(x_img_gw[2]),  # Image 3 x position
    'image_y3': float(y_img_gw[2]),  # Image 3 y position
    'image_x4': float(x_img_gw[3]),  # Image 4 x position
    'image_y4': float(y_img_gw[3]),  # Image 4 y position
    
    # ========================================================================
    # Gravitational Wave and Cosmology Parameters
    # ========================================================================
    'T_star': float(data_GW['Tstar_in_seconds']),  # Characteristic time scale
    'dL': float(dL_true),  # Luminosity distance (Mpc)
    
    # ========================================================================
    # Noise Parameter
    # ========================================================================
    'noise_sigma_bkg': DEFAULT_NOISE_KWARGS_SIMU['background_rms'],  # 0.01 - Background noise RMS
}

# ============================================================================
# 11. Create parameter groups for visualization
# ============================================================================
param_groups = {
    'lens_light': [k for k in samples.keys() if k.startswith('light_')],
    'source_light': [k for k in samples.keys() if k.startswith('source_')],
    'lens_mass': [k for k in samples.keys() if k.startswith('lens_')],
    'cosmology_params': [k for k in samples.keys() if k in ['T_star', 'dL']],
    'GW image_positions': [k for k in samples.keys() if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']],
    'GW source_position': [k for k in samples.keys() if k in ['y0gw', 'y1gw']],
    'noise_params': [k for k in samples.keys() if k in ['noise_sigma_bkg']],
}
param_groups = {k: [p for p in v if p in samples] for k, v in param_groups.items() if any(p in samples for p in v)}
truths_dict = {k: {p: input_params[p] for p in v if p in input_params} for k, v in param_groups.items()}

print(f"\nParameter groups created: {list(param_groups.keys())}")

# ============================================================================
# 12. Process samples and prepare for visualization
# ============================================================================
# Exclude certain keys from samples
keys_to_exclude = ['D_dt']  # Add any other keys to exclude
keys_to_include = [k for k in samples.keys() if k not in keys_to_exclude]
# Ensure the order matches keys_to_include
samples_no_sc = {k: samples[k] for k in keys_to_include if k in samples}
truths = {k: input_params[k] for k in keys_to_include if k in input_params}
print(f"\nSamples (after filtering): {len(samples_no_sc)} parameters")
print(f"Truths (after filtering): {len(truths)} parameters")
    
# ============================================================================
# 13. Fisher Matrix Calculation and Posterior Estimation
# ============================================================================
print("\n" + "=" * 80)
print("Computing Fisher Matrix and Fisher Posterior")
print("=" * 80)

print(f"Computing Fisher matrix for {len(keys_to_include)} parameters:")
print(f"  {keys_to_include}")

# Extract true parameter values in the correct order
u0 = jnp.array([input_params[k] for k in keys_to_include])

# Compute Fisher matrix
print("\nComputing gradient and Hessian (this may take a while)...")
approx_logp, logp0, g0, H0 = compute_fisher(
    model=probmodel.model,
    input_params=input_params,
    keys_to_include=keys_to_include,
    u0=u0,
    rng_key=jax.random.PRNGKey(42)
)

# Test at expansion point
print("\nTesting approximate log-probability function...")
approx_logp_value = approx_logp(u0)
print(f"Log-probability at expansion point: {approx_logp_value:.6f}")

# Test with a small perturbation
u_test = u0 + 0.01 * jnp.ones_like(u0)
approx_logp_value_test = approx_logp(u_test)
print(f"Log-probability with small perturbation: {approx_logp_value_test:.6f}")

# Fisher matrix is the negative Hessian (information matrix)
FM = -H0
print(f"\nFisher matrix shape: {FM.shape}")
print(f"Fisher matrix condition number: {jnp.linalg.cond(FM):.2e}")

# Compute covariance matrix (inverse of Fisher matrix)
try:
    cov = jnp.linalg.inv(FM)
    print(f"Covariance matrix computed successfully")
except:
    print("Warning: Fisher matrix is singular, using pseudo-inverse")
    cov = jnp.linalg.pinv(FM)

# Extract standard deviations (diagonal of covariance matrix)
fisher_std = jnp.sqrt(jnp.diag(cov))
print(f"\nFisher standard deviations:")
for i, key in enumerate(keys_to_include):
    print(f"  {key:20s}: {fisher_std[i]:.6f}")

# Sample from the covariance matrix
print("\nSampling from Fisher posterior (multivariate Gaussian)...")
n_fisher_samples = 5000
key = jax.random.PRNGKey(123)
samples_cov_array = jax.random.multivariate_normal(key, u0, cov, shape=(n_fisher_samples,))
samples_cov = {keys_to_include[i]: samples_cov_array[:, i] 
                for i in range(len(keys_to_include))}
print(f"Generated {n_fisher_samples} Fisher samples from covariance matrix")

# Also run MCMC with Fisher approximation model for comparison
print("\nRunning MCMC with Fisher approximation model (approximate likelihood)...")
fisher_prob_model = ProbModelFisher(
    keys_to_include=keys_to_include,
    approx_logp=approx_logp)

samples_approx_banana, summary_dict_approx_banana, extra_fields_approx_banana, mcmc_obj_approx_banana = run_mcmc(
    fisher_prob_model.model,
    num_warmup=1000,
    num_samples=5000,
    num_chains=2
)
print("Fisher approximation model MCMC complete!")

# Filter samples_approx to only include keys_to_include
samples_approx = {k: samples_approx_banana[k] for k in keys_to_include}

# Use samples_cov for Fisher samples (from covariance matrix)
fisher_samples = samples_cov

print("\nAll computations complete!")
