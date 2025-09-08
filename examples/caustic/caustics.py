import jax.numpy as jnp
from jax import grad
import jax.scipy.spatial.transform
rotation = jax.scipy.spatial.transform.Rotation

# Define the fold caustic potential 
class FoldCausticPotential(object):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['psixx', 'psiyy', 'psixxx']
    lower_limit_default = {'psixx': -100, 'psiyy': -100, 'psixxx': -100}
    upper_limit_default = {'psixx': 100, 'psiyy': 100, 'psixxx': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        self.center_x, self.center_y = 0., 0.
        self.kwargs_lens_list_default = [{"psixx": 0.91, "psiyy": -0.1, "psixxx": -0.1}]
        super(FoldCausticPotential, self).__init__()

    def function(self, x, y, psixx=None, psiyy=None, psixxx=None):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psiyy: second derivative of the potential in y (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :return: lensing potential
        """
        if psixx==None or psiyy==None or psixxx==None:
            psixx, psiyy, psixxx = self.kwargs_lens_list_default[0]['psixx'], self.kwargs_lens_list_default[0]['psiyy'], self.kwargs_lens_list_default[0]['psixxx']
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        phi = psixx * x_**2/2. + psiyy*y_**2/2. + ((1-psixx)*(1-psiyy))**0.5*x_*y_ + psixxx*x_**3/6.
        return phi

    def derivatives(self, x, y, psixx=None, psiyy=None, psixxx=None):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psiyy: second derivative of the potential in y (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :return: deflection angle (in radian)
        """
        if psixx==None or psiyy==None or psixxx==None:
            psixx, psiyy, psixxx = self.kwargs_lens_list_default[0]['psixx'], self.kwargs_lens_list_default[0]['psiyy'], self.kwargs_lens_list_default[0]['psixxx']
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        alpha_x = psixx * x_ + ((1-psixx)*(1-psiyy))**0.5*y_ + psixxx*x_**2/2.
        alpha_y = psiyy * y_ + ((1-psixx)*(1-psiyy))**0.5*x_ 
        return alpha_x, alpha_y

    def hessian(self, x, y, psixx=None, psiyy=None, psixxx=None):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psiyy: second derivative of the potential in y (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :return: hessian matrix (in radian)
        """
        if psixx==None or psiyy==None or psixxx==None:
            psixx, psiyy, psixxx = self.kwargs_lens_list_default[0]['psixx'], self.kwargs_lens_list_default[0]['psiyy'], self.kwargs_lens_list_default[0]['psixxx']
        center_x, center_y = self.center_x, self.center_y
        x_ = x - center_x
        y_ = y - center_y
        f_xx = psixx + psixxx*x_
        f_yy = psiyy
        f_xy = ((1.-psixx)*(1.-psiyy))**0.5
        return f_xx, f_yy, f_xy
    
    def image_positions(self, x_src, y_src, psixx=None, psiyy=None, psixxx=None):
        ''' 

        :param x_src: source x-coordinate (in angles)
        :param y_src: source y-coordinate (in angles)
        :param psixx: second derivative of the potential in x (in angles)
        :param psiyy: second derivative of the potential in y (in angles)
        :param psixxx: third derivative of the potential in x (in angles)
        :return: image positions (in angles)
        '''
        if psixx==None or psiyy==None or psixxx==None:
            psixx, psiyy, psixxx = self.kwargs_lens_list_default[0]['psixx'], self.kwargs_lens_list_default[0]['psiyy'], self.kwargs_lens_list_default[0]['psixxx']
        tauxx, tauyy = 1.-psixx, 1.-psiyy
        tauxxx = psixxx
        C = jnp.sqrt(tauxx*tauyy)/tauyy 
        x_plus = jnp.sqrt(-2/psixxx * (x_src + C * y_src) )
        x_minus= -x_plus
        y_plus = y_src/tauyy + C*x_plus
        y_minus= y_src/tauyy + C*x_minus
        return jnp.array([[x_plus, y_plus],
                          [x_minus, y_minus]])
        
        
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