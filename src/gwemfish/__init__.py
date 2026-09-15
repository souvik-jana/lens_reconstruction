"""
GWEMFISH: Gravitational Wave + Electromagnetic Fisher Information
Joint parameter estimation pipeline for strongly lensed GW+EM systems.
"""

from .lens_setup import (
    setup_lens,
    setup_helens_solver,
    setup_differentiable_helens_solver,
    build_lens_solver,
    normalize_solver_params,
    remove_central_image,
    select_images,
    solve_and_select,
)
from .image_finders import make_image_finder
from .diagnostics import diagnose_system
from .jax_config import setup_jax
from .data_sim import (
    setup_pixel_grid,
    setup_psf,
    setup_noise,
    simulate_em,
    simulate_gw,
    compute_gw_from_images
)
# from .prob_model_PL import ProbModel, ProbModelSourcePlane, ProbModelFisher, ProbModel_EM_GW
from .prob_model import (
    ProbModel,
    ProbModelSourcePlane,
    ProbModelSourcePlane_GW_only,
    ProbModelFisher,
    ProbModel_EM_only,
)
from .differentiable_solver import DifferentiableLensEquationSolver
from .profile_prior_rules import required_default_sampler
from .parameter_layout import (
    build_mass_parameter_entries,
    build_parameter_layout,
    build_priors_registry,
    build_vectorised_lens_kwargs_fn,
    flat_keys,
    make_infer_array_shape,
    truth_vector_from_kwargs,
)
from .flex_prob_model import (
    FlexProbModelEMGW,
    FlexProbModelEMOnly,
    FlexProbModelGWOnly,
    FlexProbModelSourcePlaneEMGW,
    FlexProbModelSourcePlaneGWOnly,
)
from .inference import run_mcmc
from .nautilus_common import run_nautilus, build_em_only_nautilus_problem
from .nautilus_source_inference import (
    build_gw_source_plane_problem,
    build_em_gw_source_plane_problem,
    build_em_only_problem,
    validate_helens_solver,
)
from .nautilus_image_inference import build_image_plane_problem
from .fisher import (
    compute_fisher,
    default_param_floors,
    format_map_params,
    newton_raphson_maxp,
)
from .config import (
    arcsecond_to_radians,
    Mpc_to_m,
    c,
    seconds_to_days,
    SOLVER_PARAMS,
    IMAGE_POSITION_SOLVER_DEFAULTS,
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

# Ultra-simple default-driven pipeline
from .simple_pipeline import (
    setup_em_observation,
    setup_gw_observation,
    prune_gw_images,
    run_inference,
    plot_posterior,
    plot_system_observation,
    plot_psf,
    compute_noise_snr_maps,
    recommend_supersampling,
    check_supersampling_convergence,
    plot_lens_system_with_source_localization,
    plot_lens_system_with_source_local_setup,
    plot_source_plane_caustic_with_localization,
    plot_source_plane_caustic_with_localization_from_setup,
    to_source_plane_samples,
    plot_source_posterior,
    make_default_cfg,
    deep_merge_cfg,
    to_serializable,
)

# Opt-in ctx -> PyAutoLens bridge (autolens imported lazily inside the functions)
from .pal_bridge import simulate_in_pal, save_pal_outputs, plot_system_observation_pal

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
    'ProbModelSourcePlane_GW_only',
    'ProbModelFisher',
    'setup_helens_solver',
    'setup_differentiable_helens_solver',
    'DifferentiableLensEquationSolver',
    'remove_central_image',
    'select_images',
    'solve_and_select',
    'build_lens_solver',
    'normalize_solver_params',
    'make_image_finder',
    'diagnose_system',
    'setup_jax',
    'run_mcmc',
    'run_nautilus',
    'build_gw_source_plane_problem',
    'build_em_gw_source_plane_problem',
    'build_em_only_problem',
    'build_em_only_nautilus_problem',
    'build_image_plane_problem',
    'validate_helens_solver',
    'compute_fisher',
    'newton_raphson_maxp',
    'format_map_params',
    'default_param_floors',
    'arcsecond_to_radians',
    'Mpc_to_m',
    'c',
    'seconds_to_days',
    'SOLVER_PARAMS',
    'IMAGE_POSITION_SOLVER_DEFAULTS',
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
    # Simple pipeline
    'setup_em_observation',
    'setup_gw_observation',
    'prune_gw_images',
    'run_inference',
    'plot_posterior',
    'plot_system_observation',
    'plot_psf',
    'compute_noise_snr_maps',
    'recommend_supersampling',
    'check_supersampling_convergence',
    'plot_lens_system_with_source_localization',
    'plot_lens_system_with_source_local_setup',
    'plot_source_plane_caustic_with_localization',
    'plot_source_plane_caustic_with_localization_from_setup',
    'to_source_plane_samples',
    'plot_source_posterior',
    'make_default_cfg',
    'deep_merge_cfg',
    'to_serializable',
    'simulate_in_pal',
    'save_pal_outputs',
    'plot_system_observation_pal',
    # EM-only model
    'ProbModel_EM_only',
    # Flexible layout (lens0_*, source0_*, light0_*) + profile priors
    'required_default_sampler',
    'build_parameter_layout',
    'build_mass_parameter_entries',
    'build_priors_registry',
    'truth_vector_from_kwargs',
    'flat_keys',
    'make_infer_array_shape',
    'build_vectorised_lens_kwargs_fn',
    'FlexProbModelEMGW',
    'FlexProbModelEMOnly',
    'FlexProbModelGWOnly',
    'FlexProbModelSourcePlaneEMGW',
    'FlexProbModelSourcePlaneGWOnly',
]

