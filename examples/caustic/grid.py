import herculens as hcl
import jax.numpy as jnp

def create_grid(xmin, xmax, ymin, ymax, nx, ny):
    """
    Create a pixel grid for lensing calculations.

    :param xmin: minimum x-coordinate (in angles)
    :param xmax: maximum x-coordinate (in angles)
    :param ymin: minimum y-coordinate (in angles)
    :param ymax: maximum y-coordinate (in angles)
    :param nx: number of pixels in x-direction
    :param ny: number of pixels in y-direction
    :return: pixel grid object and flattened x, y coordinates
    """
    # Create a pixel grid
    ra_at_xt_0, dec_at_xy_0 = xmin, ymin
    transform_pix2angle = jnp.array([[ (xmax-xmin)/nx, 0.0], [0.0, (ymax-ymin)/ny]])
    pixel_grid = hcl.PixelGrid(nx, ny, transform_pix2angle, ra_at_xt_0, dec_at_xy_0)
    grid_x, grid_y = pixel_grid.pixel_coordinates; x,y = grid_x.flatten(), grid_y.flatten()
    return pixel_grid, grid_x, grid_y, x, y