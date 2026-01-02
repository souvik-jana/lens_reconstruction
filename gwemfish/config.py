"""
Configuration constants and default kwargs for the GWEMFISH pipeline.
"""

import jax.numpy as jnp


def e1e2_to_qphi(e1, e2):
    """Convert ellipticity parameters e1, e2 to axis ratio q and position angle phi.
    
    Args:
        e1: First ellipticity component
        e2: Second ellipticity component
    
    Returns:
        q: Axis ratio (b/a, where b is minor axis, a is major axis)
        phi: Position angle in radians
    """
    e = jnp.sqrt(e1**2 + e2**2)
    q = jnp.sqrt((1 - e) / (1 + e))
    phi = 0.5 * jnp.arctan2(e2, e1)
    return float(q), float(phi)

# Physical constants
arcsecond_to_radians = 4.84813681109536e-06
Mpc_to_m = 3.085677581491367e+22
c = 299792458.0  # Speed of light in m/s
seconds_to_days = 1.1574074074074073e-05

# Default solver parameters
SOLVER_PARAMS = {
    'nsolutions': 5,
    'niter': 8,
    'scale_factor': 2,
    'nsubdivisions': 5,
}

# Default pixel grid kwargs
DEFAULT_PIXEL_GRID_KWARGS = {
    'npix': 20,
    'pix_scl': 0.4,
}

# Default PSF kwargs
DEFAULT_PSF_KWARGS = {
    'psf_type': 'GAUSSIAN',
    'fwhm': 0.2,
    'pixel_size': 0.4,
}

# Default noise kwargs
DEFAULT_NOISE_KWARGS_SIMU = {
    'npix': 20,
    'background_rms': 1e-2,
    'exposure_time': 1e3,
}

DEFAULT_NOISE_KWARGS_INFERENCE = {
    'npix': 20,
    'background_rms': None,  # Will be sampled during inference
    'exposure_time': 1e3,
}

# Default lens model list
DEFAULT_LENS_MODEL_LIST = ['EPL', 'SHEAR']

# Default light model types
# Note: These are functions that return the light model instances
# Usage: source_model = DEFAULT_SOURCE_LIGHT_MODEL()
#        lens_light_model = DEFAULT_LENS_LIGHT_MODEL()
def DEFAULT_SOURCE_LIGHT_MODEL():
    """Return default source light model instance."""
    import herculens as hcl
    return hcl.LightModel([hcl.SersicElliptic()])

def DEFAULT_LENS_LIGHT_MODEL():
    """Return default lens light model instance."""
    import herculens as hcl
    return hcl.LightModel([hcl.SersicElliptic()])

# Default lens kwargs (EPL + SHEAR)
# Note: e1, e2 computed from phi=60°, q=0.8
DEFAULT_KWARGS_LENS = [
    {
        'theta_E': 2.0,
        'e1': -0.05555555555555552,  # from phi=60°, q=0.8
        'e2': 0.0962250448649376,    # from phi=60°, q=0.8
        'gamma': 2.0,
        'center_x': 0.0,
        'center_y': 0.0,
    },
    {
        'gamma1': 0.0,
        'gamma2': 0.0,
        'ra_0': 0.0,
        'dec_0': 0.0,
    }
]

# Default redshifts
DEFAULT_ZL = 0.5
DEFAULT_ZS = 2.0

# Default source positions
DEFAULT_SOURCE_POS_EM = (0.05, 0.1)  # EM source position (x, y) in arcsec # this is source light center
DEFAULT_SOURCE_POS_GW = (0.05, 1e-6)  # GW source position (x, y) in arcsec

# Default source light kwargs (Sersic)
DEFAULT_KWARGS_SOURCE = [
    {
        'amp': 4.0,
        'R_sersic': 0.5,
        'n_sersic': 2.0,
        'e1': 0.05,
        'e2': 0.05,
        'center_x': 0.05,
        'center_y': 0.1,
    }
]

# Default lens light kwargs (Sersic)
# Note: e1, e2 same as lens mass (from phi=60°, q=0.8)
DEFAULT_KWARGS_LENS_LIGHT = [
    {
        'amp': 8.0,
        'R_sersic': 1.0,
        'n_sersic': 3.0,
        'e1': -0.05555555555555552,  # same as lens mass
        'e2': 0.0962250448649376,    # same as lens mass
        'center_x': 0.0,
        'center_y': 0.0,
    }
]

# Default numerics kwargs
DEFAULT_KWARGS_NUMERICS = {
    'supersampling_factor': 1,
}

