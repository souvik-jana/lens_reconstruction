"""
GWEMFISH: Gravitational Wave + Electromagnetic Fisher Information
Joint parameter estimation pipeline for strongly lensed GW+EM systems.
"""

from .lens_setup import setup_lens, setup_helens_solver, remove_central_image
from .jax_config import setup_jax
from .data_sim import (
    setup_pixel_grid,
    setup_psf,
    setup_noise,
    simulate_em,
    simulate_gw,
    compute_gw_from_images
)
from .prob_model import ProbModel, ProbModelSourcePlane, ProbModelFisher
from .inference import run_mcmc
from .fisher import compute_fisher
from .config import (
    arcsecond_to_radians,
    Mpc_to_m,
    c,
    seconds_to_days,
    SOLVER_PARAMS,
    DEFAULT_PIXEL_GRID_KWARGS,
    DEFAULT_PSF_KWARGS,
    DEFAULT_NOISE_KWARGS_SIMU,
    DEFAULT_NOISE_KWARGS_INFERENCE,
    DEFAULT_LENS_MODEL_LIST,
    DEFAULT_KWARGS_LENS,
    DEFAULT_ZL,
    DEFAULT_ZS,
    DEFAULT_SOURCE_POS_EM,
    DEFAULT_SOURCE_POS_GW,
    DEFAULT_KWARGS_SOURCE,
    DEFAULT_KWARGS_LENS_LIGHT,
    DEFAULT_KWARGS_NUMERICS,
    DEFAULT_SOURCE_LIGHT_MODEL,
    DEFAULT_LENS_LIGHT_MODEL,
)

__all__ = [
    'setup_lens',
    'setup_pixel_grid',
    'setup_psf',
    'setup_noise',
    'simulate_em',
    'simulate_gw',
    'compute_gw_from_images',
    'ProbModel',
    'ProbModelSourcePlane',
    'ProbModelFisher',
    'setup_helens_solver',
    'remove_central_image',
    'setup_jax',
    'run_mcmc',
    'compute_fisher',
    'arcsecond_to_radians',
    'Mpc_to_m',
    'c',
    'seconds_to_days',
    'SOLVER_PARAMS',
    'DEFAULT_PIXEL_GRID_KWARGS',
    'DEFAULT_PSF_KWARGS',
    'DEFAULT_NOISE_KWARGS_SIMU',
    'DEFAULT_NOISE_KWARGS_INFERENCE',
    'DEFAULT_LENS_MODEL_LIST',
    'DEFAULT_KWARGS_LENS',
    'DEFAULT_ZL',
    'DEFAULT_ZS',
    'DEFAULT_SOURCE_POS_EM',
    'DEFAULT_SOURCE_POS_GW',
    'DEFAULT_KWARGS_SOURCE',
    'DEFAULT_KWARGS_LENS_LIGHT',
    'DEFAULT_KWARGS_NUMERICS',
    'DEFAULT_SOURCE_LIGHT_MODEL',
    'DEFAULT_LENS_LIGHT_MODEL',
]

