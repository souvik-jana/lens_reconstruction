import jax.numpy as jnp
from jax import grad
import jax.scipy.spatial.transform
import herculens as hcl
rotation = jax.scipy.spatial.transform.Rotation

# Define the fold caustic potential 
class FoldCausticPotential(object):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['psixx', 'psixy', 'psixxx', 'psixxy', 'psixyy', 'psiyyy']
    lower_limit_default = {'psixx': -100, 'psixy': -100, 'psixxx': -100, 'psixxy': -100, 'psixyy': -100, 'psiyyy': -100}
    upper_limit_default = {'psixx': 100, 'psixy': 100, 'psixxx': 100, 'psixxy': 100, 'psixyy': 100, 'psiyyy': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self, kwargs_lens_list_default=[{"psixx": 0.91, "psixy": 0.0, "psixxx": -0.1, "psixxy": 0.0, "psixyy": 0.0, "psiyyy": 0.0}]):
        self.center_x, self.center_y = 0., 0.
        self.kwargs_lens_list_default = kwargs_lens_list_default
        super(FoldCausticPotential, self).__init__()

    def _get_psiyy(self, psixx, psixy):
        """Compute psiyy using the constraint psiyy = psixy**2/(1-psixx) - 1"""
        return psixy**2 / (1 - psixx) - 1

    def function(self, x, y, psixx=None, psixy=None, psixxx=None, psixxy=None, psixyy=None, psiyyy=None):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psixy: mixed second derivative of the potential (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :param psixxy: third derivative of the potential (in angles)
        :param psixyy: third derivative of the potential (in angles)
        :param psiyyy: third derivative of the potential in y (in angles)
        :return: lensing potential
        """
        if any(param is None for param in [psixx, psixy, psixxx, psixxy, psixyy, psiyyy]):
            defaults = self.kwargs_lens_list_default[0]
            psixx = psixx if psixx is not None else defaults['psixx']
            psixy = psixy if psixy is not None else defaults['psixy']
            psixxx = psixxx if psixxx is not None else defaults['psixxx']
            psixxy = psixxy if psixxy is not None else defaults['psixxy']
            psixyy = psixyy if psixyy is not None else defaults['psixyy']
            psiyyy = psiyyy if psiyyy is not None else defaults['psiyyy']
        
        psiyy = self._get_psiyy(psixx, psixy)
        
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        
        phi = (psixx * x_**2/2. + psixy * x_ * y_ + psiyy * y_**2/2. + 
               psixxx * x_**3/6. + psixxy * x_**2 * y_/2. + 
               psixyy * x_ * y_**2/2. + psiyyy * y_**3/6.)
        return phi

    def derivatives(self, x, y, psixx=None, psixy=None, psixxx=None, psixxy=None, psixyy=None, psiyyy=None):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psixy: mixed second derivative of the potential (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :param psixxy: third derivative of the potential (in angles)
        :param psixyy: third derivative of the potential (in angles)
        :param psiyyy: third derivative of the potential in y (in angles)
        :return: deflection angle (in radian)
        """
        if any(param is None for param in [psixx, psixy, psixxx, psixxy, psixyy, psiyyy]):
            defaults = self.kwargs_lens_list_default[0]
            psixx = psixx if psixx is not None else defaults['psixx']
            psixy = psixy if psixy is not None else defaults['psixy']
            psixxx = psixxx if psixxx is not None else defaults['psixxx']
            psixxy = psixxy if psixxy is not None else defaults['psixxy']
            psixyy = psixyy if psixyy is not None else defaults['psixyy']
            psiyyy = psiyyy if psiyyy is not None else defaults['psiyyy']
        
        psiyy = self._get_psiyy(psixx, psixy)
        
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        
        alpha_x = (psixx * x_ + psixy * y_ + 
                   psixxx * x_**2/2. + psixxy * x_ * y_ + psixyy * y_**2/2.)
        alpha_y = (psixy * x_ + psiyy * y_ + 
                   psixxy * x_**2/2. + psixyy * x_ * y_ + psiyyy * y_**2/2.)
        return alpha_x, alpha_y

    def hessian(self, x, y, psixx=None, psixy=None, psixxx=None, psixxy=None, psixyy=None, psiyyy=None):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psixy: mixed second derivative of the potential (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :param psixxy: third derivative of the potential (in angles)
        :param psixyy: third derivative of the potential (in angles)
        :param psiyyy: third derivative of the potential in y (in angles)
        :return: hessian matrix (in radian)
        """
        if any(param is None for param in [psixx, psixy, psixxx, psixxy, psixyy, psiyyy]):
            defaults = self.kwargs_lens_list_default[0]
            psixx = psixx if psixx is not None else defaults['psixx']
            psixy = psixy if psixy is not None else defaults['psixy']
            psixxx = psixxx if psixxx is not None else defaults['psixxx']
            psixxy = psixxy if psixxy is not None else defaults['psixxy']
            psixyy = psixyy if psixyy is not None else defaults['psixyy']
            psiyyy = psiyyy if psiyyy is not None else defaults['psiyyy']
        
        psiyy = self._get_psiyy(psixx, psixy)
        
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        
        f_xx = psixx + psixxx * x_ + psixxy * y_
        f_yy = psiyy + psixyy * x_ + psiyyy * y_
        f_xy = psixy + psixxy * x_ + psixyy * y_
        return f_xx, f_yy, f_xy
    
    def image_positions(self, x_src, y_src, psixx=None, psixy=None, psixxx=None, psixxy=None, psixyy=None, psiyyy=None):
        ''' 
        :param x_src: source x-coordinate (in angles)
        :param y_src: source y-coordinate (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psixy: mixed second derivative of the potential (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :param psixxy: third derivative of the potential (in angles)
        :param psixyy: third derivative of the potential (in angles)
        :param psiyyy: third derivative of the potential in y (in angles)
        :return: image positions (in angles)
        '''
        # Note: This analytical solution may not be valid for the general case
        # and would need to be solved numerically for arbitrary parameter combinations
        if any(param is None for param in [psixx, psixy, psixxx, psixxy, psixyy, psiyyy]):
            defaults = self.kwargs_lens_list_default[0]
            psixx = psixx if psixx is not None else defaults['psixx']
            psixy = psixy if psixy is not None else defaults['psixy']
            psixxx = psixxx if psixxx is not None else defaults['psixxx']
            psixxy = psixxy if psixxy is not None else defaults['psixxy']
            psixyy = psixyy if psixyy is not None else defaults['psixyy']
            psiyyy = psiyyy if psiyyy is not None else defaults['psiyyy']
        
        # For the general case, analytical solutions become complex
        # This would typically require numerical root finding
        raise NotImplementedError("Analytical image positions for general parameter combinations require numerical solving")
        
        
class CausticExpansion():
    def __init__(self, lens_mass_model, kwargs_lens_list):
        self.lens_mass_model = lens_mass_model
        self.kwargs_lens_list = kwargs_lens_list
        self.fold_caustic_potential = FoldCausticPotential()
        # self.fold_caustic_potential.kwargs_lens_list_default = self.fold_caustic_potential.kwargs_lens_list_default

    def get_inverse_magnification_eigenvectors(self, x, y):
        # Compute the inverse magnification tensor at the origin:
        A = jnp.diag(jnp.ones(2)) - jnp.array(self.lens_mass_model.hessian(x, y, self.kwargs_lens_list)).reshape(2,2)
        # Compute the eigenvectors:
        eigenvalues, eigenvectors = jnp.linalg.eig(A) #jnp.linalg.eig(jnp.linalg.inv(A))
        v1, v2 = eigenvectors[:,0], eigenvectors[:,1]
        v1, v2 = v1.real, v2.real # Make into real-valued
        lambda1, lambda2 = eigenvalues[0].real, eigenvalues[1].real
        return v1, v2, lambda1, lambda2
    
    def get_detA(self, x, y):
        f_xx, f_xy, f_yx, f_yy = self.lens_mass_model.hessian(x, y, self.kwargs_lens_list)
        detA = (1.-f_xx)*(1.-f_yy) - f_xy**2
        return detA
    
    def get_nabla_detA(self, x, y):
        """Compute gradient of detA at (x, y) and its 90-degree rotation.

        Args:
            x: x-coordinate(s)
            y: y-coordinate(s)

        Returns:
            nabla_detA: [∂detA/∂x, ∂detA/∂y]
            nabla_detA_rot: nabla_detA rotated by 90 degrees
        """
        # Use jax automatic differentiation to compute the gradient of detA:
        grad_detA_x = grad(self.get_detA, argnums=0)(x, y)
        grad_detA_y = grad(self.get_detA, argnums=1)(x, y)
        nabla_detA = jnp.array([grad_detA_x, grad_detA_y])
        rotation_matrix = rotation.from_euler('z', 90, degrees=True).as_matrix()[:2, :2]
        nabla_detA_rot = rotation_matrix @ nabla_detA
        return nabla_detA, nabla_detA_rot
    
    def get_derivatives_to_third(self, x, y, R=None):
        ''' Compute all derivatives of the potential up to third order at (x, y).
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param R: optional rotation matrix to apply to the derivatives. Options: None, 'x-align-critical' (align x-axis perpendicular to critical curve), or a custom 2x2 rotation matrix.
        
        :return: dictionary of derivatives
        '''
        # Define the potential as a function of (x, y)
        f = lambda xy: self.lens_mass_model.potential(xy[0], xy[1], self.kwargs_lens_list)
        # Compute gradients using jax automatic differentiation
        grad_func = grad(f)
        hess_func = jax.jacfwd(grad_func)
        third_deriv_func = jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))
        f_ij = hess_func(jnp.array([x, y]))
        f_ijk = third_deriv_func(jnp.array([x, y]))
        if R == 'x-align-critical':
            nabla_detA, nabla_detA_rot = self.get_nabla_detA(x, y)
            x_unit = jnp.array([1.0, 0.0])
            R=create_rotation_matrix(vec_original=nabla_detA, vec_target=x_unit) # Align the x-axis perpendicular to the critical curve
        if R is not None:
            # Rotate the derivatives using the rotation matrix R
            f_ij = R @ f_ij @ R.T
            f_ijk = jnp.einsum('ia,jb,kc,abc->ijk', R, R, R, f_ijk)
        return f_ij, f_ijk

from grid import create_grid
def get_critical_lines_caustics(lens_mass_model, kwargs_lens_list, pixel_grid=create_grid(-3, 3, -3, 3, 300, 300)[0]):
    lens_image = hcl.LensImage(lens_mass_model_class=lens_mass_model, grid_class=pixel_grid, psf_class=hcl.PSF(psf_type="NONE"))
    return hcl.Util.model_util.critical_lines_caustics(lens_image, kwargs_lens_list)


def select_critical_point_by_angle(crit_lines, angle_deg):
    """
    Select a point on the critical curve closest to the given angle (in degrees)
    relative to the origin (0,0).
    """
    x_crit, y_crit = crit_lines[0][0], crit_lines[0][1]
    angles = jnp.degrees(jnp.arctan2(y_crit, x_crit))
    idx = jnp.argmin(jnp.abs(angles - angle_deg))
    return x_crit[idx], y_crit[idx]

def create_rotation_matrix(vec_original, vec_target):
    """Create a 2D rotation matrix that aligns the vec_original to vec_target.

    Args:
        vec_original: Original vector
        vec_target: Target vector

    Returns:
        A 2x2 rotation matrix.
    """
    v_norm = vec_original / jnp.linalg.norm(vec_original)
    u_norm = vec_target / jnp.linalg.norm(vec_target)
    cross = v_norm[0] * u_norm[1] - v_norm[1] * u_norm[0]
    dot = jnp.dot(v_norm, u_norm)
    theta = jnp.arctan2(cross, dot)
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta),  jnp.cos(theta)]])
    return R

def compute_caustic_expansion_from_macro(lens_mass_model, kwargs_lens_list, pixel_grid, angle_degrees=45, solve_images=False):
    """
    Compute caustic expansion parameters at a critical point.
    
    Parameters:
    -----------
    lens_mass_model : MassModel
        The lens mass model
    kwargs_lens_list : list
        List of lens parameters
    pixel_grid : Grid
        Grid on which to compute critical lines and caustics
    angle_degrees : float
        Angle in degrees to select point on critical curve
        
    Returns:
    --------
    dict : Dictionary containing all computed quantities
    """
    # Compute eigenvectors at the caustics:
    caustic_expansion = CausticExpansion(lens_mass_model, kwargs_lens_list)
    # Compute caustics
    crit_lines = get_critical_lines_caustics(lens_mass_model, kwargs_lens_list, pixel_grid)[0]
    # Select a point on the critical curve by angle (in degrees)
    x0, y0 = select_critical_point_by_angle(crit_lines, angle_degrees)
    # Get eigenvectors and eigenvalues of the inverse magnification matrix
    v1, v2, lambda1, lambda2 = caustic_expansion.get_inverse_magnification_eigenvectors(x0, y0)
    # Compute the gradient of the determinant of the magnification matrix and its 90-degree rotation
    nabla_detA, nabla_detA_rot = caustic_expansion.get_nabla_detA(x0, y0)
    # Normalize eigenvectors
    v1_norm = v1 / jnp.linalg.norm(v1)
    v2_norm = v2 / jnp.linalg.norm(v2)
    # Select the eigenvector corresponding to the larger eigenvalue
    v_per = v2_norm if jnp.abs(lambda1) > jnp.abs(lambda2) else v1_norm # Perpendicular to the caustic
    v_par = v1_norm if jnp.abs(lambda1) > jnp.abs(lambda2) else v2_norm # Parallel to the caustic
    # Create rotation matrix to align v2 with y-axis
    R = create_rotation_matrix(vec_original=v_per, vec_target=jnp.array([0.0, 1.0]))
    # Compute second and third order derivatives of the potential
    f_ij, f_ijk = caustic_expansion.get_derivatives_to_third(x0, y0, R=R)
    # Make caustic expansion:
    kwargs_lens_caustic_list = [{
        "psixx": float(f_ij[0, 0]),
        "psiyy": float(f_ij[1, 1]),
        "psixy": float(f_ij[0, 1]),
        "psixxx": float(f_ijk[0, 0, 0]),
        "psixxy": float(f_ijk[0, 0, 1]),
        "psixyy": float(f_ijk[0, 1, 1]),
        "psiyyy": float(f_ijk[1, 1, 1])
    }]
    lens_mass_model_caustic = hcl.MassModel([FoldCausticPotential(kwargs_lens_caustic_list)])
    result = {
        'caustic_expansion': caustic_expansion,
        'x0': x0,
        'y0': y0,
        'v1': v1,
        'v2': v2,
        'nabla_detA': nabla_detA,
        'nabla_detA_rot': nabla_detA_rot,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'v1_norm': v1_norm,
        'v2_norm': v2_norm,
        'v_per': v_per,
        'v_par': v_par,
        'R': R,
        'f_ij': f_ij,
        'f_ijk': f_ijk,
        'kwargs_lens_caustic_list': kwargs_lens_caustic_list,
        'lens_mass_model_caustic': lens_mass_model_caustic
    }
    if solve_images == True:
        import helens
        # Get ray shooting function
        ray_shooting_func = lens_mass_model.ray_shooting
        x_src, y_src = jnp.array(ray_shooting_func(x0, y0, kwargs_lens_list)) + v_per*0.005*jnp.sign(v_per@nabla_detA)
        # Compute the image positions:
        lens_eq_solver = helens.LensEquationSolver(pixel_grid._x_grid, pixel_grid._y_grid, ray_shooting_func)
        beta   = jnp.array([x_src, y_src])
        img, src = lens_eq_solver.solve(beta, kwargs_lens_list, nsolutions=5, niter=5, nsubdivisions=8)
        result['x_src'] = x_src
        result['y_src'] = y_src
        result['img'] = img
        result['src'] = src
    return result


if __name__ == "__main__":
    import numpy as np

    # Test rotation matrix with different vectors:
    for i in range(5):
        vec_original = jnp.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
        vec_original = vec_original / jnp.linalg.norm(vec_original)
        angle = jnp.pi/4 * i
        vec_target = jnp.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
        vec_target = vec_target / jnp.linalg.norm(vec_target)
        R = create_rotation_matrix(vec_original, vec_target)
        print(f"Test {i+1}:")
        print("Original vector:", vec_original)
        print("Target vector:", vec_target)
        print("Rotation matrix:\n", R)
        rotated_vec = R @ vec_original
        print("Rotated vector:", rotated_vec)
        print("Difference from target:", rotated_vec - vec_target)
        if jnp.allclose(rotated_vec, vec_target):
            print("Rotation successful!")
        else:
            raise ValueError("Rotation failed!", rotated_vec, vec_target)
        print()