"""
Lens setup and image position solver.

This module provides functions to set up lens mass models compatible with
herculens and solve for image positions given a source position.
"""

import jax.numpy as jnp
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from herculens.MassModel.mass_model import MassModel
from .config import SOLVER_PARAMS

# Import helens solver if available
try:
    from helens import LensEquationSolver as LensEquationSolver_helens
    HELENS_AVAILABLE = True
except ImportError:
    HELENS_AVAILABLE = False
    LensEquationSolver_helens = None


def setup_lens(lens_model_list, kwargs_lens, zl, zs, source_pos, 
               solver_params=None):
    """Setup lens mass model and solve for image positions.
    
    This function is general and accepts any lens model compatible with
    herculens. The kwargs_lens should match the lens_model_list.
    
    Args:
        lens_model_list: List of lens model names (e.g., ['EPL', 'SHEAR'])
        kwargs_lens: List of kwargs dicts for each lens model component.
                     Each dict should contain parameters for that model.
        zl: Lens redshift
        zs: Source redshift
        source_pos: Tuple (x, y) of source position in arcsec
        solver_params: Optional solver parameters dict. If None, uses defaults.
    
    Returns: 
        kwargs_lens: List of lens kwargs (same as input, for consistency)
        x_image_true: Array of image x positions (arcsec)
        y_image_true: Array of image y positions (arcsec)
        lens_mass_model: hcl.MassModel instance for use in herculens
    """
    if solver_params is None:
        solver_params = SOLVER_PARAMS.copy()
    
    # Create herculens MassModel
    lens_mass_model = MassModel(lens_model_list)
    
    # Setup jaxtronomy lens model for solving
    lensModel = LensModel(
        lens_model_list=lens_model_list,
        z_lens=zl,
        z_source=zs
    )
    solver_lenstronomy = LensEquationSolver(lensModel)
    
    # Convert kwargs to floats for lenstronomy compatibility
    kwargs_lens_fixed = []
    for kw in kwargs_lens:
        kw_fixed = {}
        for key, value in kw.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                kw_fixed[key] = float(value)
            else:
                kw_fixed[key] = float(value) if not isinstance(value, (int, float)) else value
        kwargs_lens_fixed.append(kw_fixed)
    
    # Extract source position
    source_x, source_y = source_pos
    source_x_float = float(source_x)
    source_y_float = float(source_y)
    
    # Solve for image positions
    x_image_true, y_image_true = solver_lenstronomy.image_position_from_source(
        kwargs_lens=kwargs_lens_fixed,
        sourcePos_x=source_x_float,
        sourcePos_y=source_y_float,
        min_distance=0.01,
        search_window=15,
        precision_limit=1e-10,
        num_iter_max=1200,
        solver='lenstronomy'
    )
    
    # Convert to JAX arrays
    x_image_true = jnp.array(x_image_true)
    y_image_true = jnp.array(y_image_true)
    
    return kwargs_lens, x_image_true, y_image_true, lens_mass_model


def remove_central_image(thetas, betas, cx0, cy0):
    """Remove the central image (closest to lens center) from solver results.
    
    Args:
        thetas: Array of shape (N, 2) with image plane positions
        betas: Array of shape (N, 2) with source plane positions
        cx0: Lens center x coordinate
        cy0: Lens center y coordinate
    
    Returns:
        theta_x_no_central: Array of image x positions without central image
        theta_y_no_central: Array of image y positions without central image
        beta_x_no_central: Array of source x positions without central image
        beta_y_no_central: Array of source y positions without central image
    """
    # thetas, betas: shape (N, 2)
    theta_x, theta_y = thetas.T
    beta_x, beta_y = betas.T

    idx = jnp.argmin(jnp.hypot(theta_x - cx0, theta_y - cy0))  # int index
    n = theta_x.shape[0]  # static

    # Create a mask: True for all indices except idx
    mask = jnp.arange(n) != idx
    
    # Reorder: all masked elements first, then the idx element
    # Use argsort on ~mask to put False (idx) at the end
    order = jnp.argsort(~mask, stable=True)
    
    # Apply reordering
    theta_x_reordered = theta_x[order]
    theta_y_reordered = theta_y[order]
    beta_x_reordered = beta_x[order]
    beta_y_reordered = beta_y[order]
    
    # Return only the non-central images (first n-1 elements)
    theta_x_no_central = theta_x_reordered[:n-1]
    theta_y_no_central = theta_y_reordered[:n-1]
    beta_x_no_central = beta_x_reordered[:n-1]
    beta_y_no_central = beta_y_reordered[:n-1]
    
    return theta_x_no_central, theta_y_no_central, beta_x_no_central, beta_y_no_central


def setup_helens_solver(pixel_grid, lens_gw, pixel_scale_factor=0.8, solver_params=None):
    """Setup helens LensEquationSolver for source plane inference.
    
    This solver is used during MCMC inference when sampling source positions
    and solving for image positions.
    
    Args:
        pixel_grid: hcl.PixelGrid instance (main observation grid)
        lens_gw: LensImageGW instance with ray_shoot method
        pixel_scale_factor: Factor to create coarser solver grid (default 0.8)
        solver_params: Optional solver parameters dict. If None, uses defaults.
    
    Returns:
        solver: LensEquationSolver_helens instance
        solver_pixel_grid: Coarser pixel grid for solver
        solver_params: Solver parameters dict
    """
    if not HELENS_AVAILABLE:
        raise ImportError("helens package is required for source plane inference. "
                         "Install it with: pip install helens")
    
    if solver_params is None:
        solver_params = SOLVER_PARAMS.copy()
    
    # Create coarser solver grid
    solver_pixel_grid = pixel_grid.create_model_grid(pixel_scale_factor=pixel_scale_factor)
    
    # Extract solver grid coordinates
    solver_grid_x = solver_pixel_grid.pixel_coordinates[0]
    solver_grid_y = solver_pixel_grid.pixel_coordinates[1]
    
    # Initialize helens solver with ray_shoot function
    solver = LensEquationSolver_helens(solver_grid_x, solver_grid_y, lens_gw.ray_shoot)
    
    return solver, solver_pixel_grid, solver_params

