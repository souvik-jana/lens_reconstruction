"""
Configuration constants and default kwargs for the GWEMFISH pipeline.
"""

import jax.numpy as jnp

# Physical constants
arcsecond_to_radians = 4.84813681109536e-06
Mpc_to_m = 3.085677581491367e+22
c = 299792458.0  # Speed of light in m/s
seconds_to_days = 1.1574074074074073e-05

# Lens-equation solver settings, nested by image-finder backend.
#
# The shared keys apply whichever backend runs; each backend's own knobs live under
# its name, so settings for the backend you are *not* using are carried along rather
# than silently dropped, and you can flip "backend" without re-editing anything.
#
# Legacy flat cfgs are still accepted: ``normalize_solver_params`` in lens_setup routes
# a flat key into its nest with a DeprecationWarning.
SOLVER_PARAMS = {
    # --- shared ---
    # "auto" picks jaxtronomy's closed-form solver when the lens model list supports it
    # (see ANALYTICAL_LENS_MODELS), else helens' triangle search.
    "backend": "auto",          # "auto" | "helens" | "jaxtronomy"
    # Solution slots. "auto" -> n_images + 1: exactly one spare, which holds the central
    # image when the profile has one (gamma < 2) and stays padding when it does not.
    "nsolutions": "auto",
    # Newton-polish steps. A step count, NOT an on/off switch (must be >= 1): the four
    # gradient-based *-source methods cannot run without polishing, since it is what
    # supplies their derivatives. To skip polishing on nautilus-source, which needs no
    # derivatives, use cfg["nautilus"]["polish"].
    "n_newton": 8,
    # Separation below which two solutions are the same image. None -> resolved from the
    # position accuracy actually achieved (see resolve_duplicate_tol).
    "duplicate_tol": None,

    # --- helens triangle search: any lens model, positions accurate to ~the final triangle ---
    "helens": {
        "niter": 8,
        "scale_factor": 2,
        "nsubdivisions": 5,     # raise this first when an image is missed
        "pixel_scale_factor": 0.8,
    },

    # --- jaxtronomy: "analytical" is closed-form (exact), "lenstronomy" is grid + Newton ---
    "jaxtronomy": {
        "solver": "analytical",
        # analytical only
        "magnification_limit": 1e-4,   # junk-root filter; low enough to keep the central image
        "Nmeas": 400,
        "Nmeas_extra": 80,
        # lenstronomy only
        "min_distance": 0.01,
        "search_window": 15,
        "precision_limit": 1e-10,
        "num_iter_max": 1200,
        # applies to both jaxtronomy solvers
        "arrival_time_sort": True,
    },
}

# Shared keys consumed when the finder is built, not forwarded to solve().
SHARED_SOLVER_KEYS = frozenset({"backend", "nsolutions", "n_newton", "duplicate_tol"})

# Backend names that own a nested block in SOLVER_PARAMS.
SOLVER_BACKENDS = frozenset({"helens", "jaxtronomy"})

# Lens model lists jaxtronomy's closed-form solver supports; "auto" picks it for these.
ANALYTICAL_LENS_MODELS = frozenset({"EPL", "EPL_NUMBA", "SIE", "SIS"})
ANALYTICAL_EXTRA_MODELS = frozenset({"SHEAR", "CONVERGENCE"})

# Only meaningful for jaxtronomy's "analytical" solver; dropped for "lenstronomy".
ANALYTICAL_ONLY_KWARGS = frozenset({"magnification_limit", "Nmeas", "Nmeas_extra"})

# Only meaningful for jaxtronomy's "lenstronomy" grid solver; dropped for "analytical".
LENSTRONOMY_GRID_KWARGS = frozenset(
    {
        "min_distance",
        "search_window",
        "precision_limit",
        "num_iter_max",
        "initial_guess_cut",
        "verbose",
        "x_center",
        "y_center",
        "num_random",
    }
)

# --- Deprecated flat views, kept so existing scripts and cfgs keep importing. ---
# Prefer SOLVER_PARAMS["jaxtronomy"] / SOLVER_PARAMS["helens"]. Passing either of
# these as cfg["gw"]["solver_params"] still works: normalize_solver_params migrates
# flat keys into their nest.
IMAGE_POSITION_SOLVER_DEFAULTS = {
    "solver": "lenstronomy",
    "min_distance": 0.01,
    "search_window": 15,
    "precision_limit": 1e-10,
    "num_iter_max": 1200,
    "arrival_time_sort": True,
}

HELEN_LENS_SOLVER_PARAM_KEYS = frozenset(
    {"nsolutions", "niter", "scale_factor", "nsubdivisions"}
)

# Where a legacy flat solver_params key now lives. Used to migrate old cfgs.
LEGACY_SOLVER_KEY_HOME = {
    "niter": "helens",
    "scale_factor": "helens",
    "nsubdivisions": "helens",
    "pixel_scale_factor": "helens",
    "solver": "jaxtronomy",
    "magnification_limit": "jaxtronomy",
    "Nmeas": "jaxtronomy",
    "Nmeas_extra": "jaxtronomy",
    "min_distance": "jaxtronomy",
    "search_window": "jaxtronomy",
    "precision_limit": "jaxtronomy",
    "num_iter_max": "jaxtronomy",
    "arrival_time_sort": "jaxtronomy",
    "initial_guess_cut": "jaxtronomy",
    "verbose": "jaxtronomy",
    "x_center": "jaxtronomy",
    "y_center": "jaxtronomy",
    "num_random": "jaxtronomy",
}

# Default pixel grid kwargs
DEFAULT_PIXEL_GRID_KWARGS = {
    'npix': 20,
    'pix_scl': 0.4,
}

# Default PSF kwargs. No 'pixel_size': the GAUSSIAN kernel must be rendered on the image
# grid, so setup_em_observation fills it from pixel_grid_kwargs['pix_scl']. Pinning it here
# would make every cfg that only changes pix_scl a mismatch.
DEFAULT_PSF_KWARGS = {
    'psf_type': 'GAUSSIAN',
    'fwhm': 0.2,
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
    # return hcl.LightModel([hcl.SersicElliptic()])
    return hcl.LightModel([hcl.Sersic()])

def DEFAULT_LENS_LIGHT_MODEL():
    """Return default lens light model instance."""
    import herculens as hcl
    # return hcl.LightModel([hcl.SersicElliptic()])
    return hcl.LightModel([hcl.Sersic()])

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

