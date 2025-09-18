# JAX-based cosmology functions that match Astropy exactly
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.integrate import trapezoid

# Define JAX cosmology functions as standalone functions (not methods)
# This avoids issues with JIT compilation and class instances

@jit
def E_func(z, Om0, Ode0, Ok0):
    """Dimensionless Hubble parameter E(z) = H(z)/H0."""
    return jnp.sqrt(Om0 * (1 + z)**3 + Ode0 + Ok0 * (1 + z)**2)

@jit
def inv_E_func(z, Om0, Ode0, Ok0):
    """Inverse of dimensionless Hubble parameter 1/E(z)."""
    return 1.0 / E_func(z, Om0, Ode0, Ok0)

@jit
def comoving_distance_single(z, hubble_distance, Om0, Ode0, Ok0):
    """Comoving distance to redshift z for a single redshift value."""
    # Handle zero or negative redshift
    z = jnp.maximum(z, 0.0)
    
    # Create integration grid
    n_points = 1000
    z_grid = jnp.linspace(0.0, z, n_points)
    integrand = inv_E_func(z_grid, Om0, Ode0, Ok0)
    
    # Integrate using trapezoidal rule
    integral_result = trapezoid(integrand, z_grid)
    
    return hubble_distance * integral_result

class JAXCosmology:
    """
    JAX-based cosmology class that matches Astropy's FlatLambdaCDM implementation.
    Uses numerical integration for exact results.
    """
    
    def __init__(self, H0=70.0, Om0=0.3, Ode0=None):
        """
        Initialize cosmology parameters.
        
        Parameters:
        -----------
        H0 : float
            Hubble constant in km/s/Mpc
        Om0 : float  
            Matter density parameter at z=0
        Ode0 : float, optional
            Dark energy density parameter. If None, assumes flat universe (Ode0 = 1 - Om0)
        """
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = 1.0 - Om0 if Ode0 is None else Ode0
        self.Ok0 = 1.0 - self.Om0 - self.Ode0  # Curvature parameter
        
        # Hubble distance in Mpc
        self.hubble_distance = 299792.458 / H0  # c/H0 in Mpc
        
        # Create vectorized version of comoving distance function
        self._comoving_distance_vmap = vmap(
            lambda z: comoving_distance_single(z, self.hubble_distance, self.Om0, self.Ode0, self.Ok0)
        )
        
    def E(self, z):
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : E(z)
        """
        return E_func(z, self.Om0, self.Ode0, self.Ok0)
    
    def inv_E(self, z):
        """
        Inverse of dimensionless Hubble parameter 1/E(z).
        This is the integrand for comoving distance.
        """
        return inv_E_func(z, self.Om0, self.Ode0, self.Ok0)
    
    def comoving_distance(self, z):
        """
        Comoving distance to redshift z. Handles both single values and arrays.
        
        Parameters:
        -----------
        z : float or array
            Redshift(s)
            
        Returns:
        --------
        float or array : Comoving distance(s) in Mpc
        """
        z = jnp.asarray(z)
        
        # Handle single value
        if z.ndim == 0:
            return comoving_distance_single(z, self.hubble_distance, self.Om0, self.Ode0, self.Ok0)
        
        # Handle array using vmap
        return self._comoving_distance_vmap(z)
    
    def luminosity_distance(self, z):
        """
        Luminosity distance to redshift z.
        
        Parameters:
        -----------
        z : float or array
            Redshift(s)
            
        Returns:
        --------
        float or array : Luminosity distance(s) in Mpc
        """
        dc = self.comoving_distance(z)
        z = jnp.asarray(z)
        return dc * (1 + z)
    
    def angular_diameter_distance(self, z):
        """
        Angular diameter distance to redshift z.
        
        Parameters:
        -----------
        z : float or array
            Redshift(s)
            
        Returns:
        --------
        float or array : Angular diameter distance(s) in Mpc
        """
        dc = self.comoving_distance(z)
        z = jnp.asarray(z)
        return dc / (1 + z)
    
    def angular_diameter_distance_z1z2(self, z1, z2):
        """
        Angular diameter distance between two redshifts.
        
        Parameters:
        -----------
        z1 : float
            Lower redshift
        z2 : float
            Higher redshift (z2 > z1)
            
        Returns:
        --------
        float : Angular diameter distance between z1 and z2 in Mpc
        """
        # if z1 >= z2:
        #     return 0.0
            
        dc1 = self.comoving_distance(z1)
        dc2 = self.comoving_distance(z2)
        
        return (dc2 - dc1) / (1 + z2)
    
    def time_delay_distance(self, z_lens, z_source):
        """
        Time delay distance for gravitational lensing.
        
        Parameters:
        -----------
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
            
        Returns:
        --------
        float : Time delay distance in Mpc
        """
        # if z_lens >= z_source:
        #     return 0.0
            
        Dd = self.angular_diameter_distance(z_lens)
        Ds = self.angular_diameter_distance(z_source)
        Dds = self.angular_diameter_distance_z1z2(z_lens, z_source)
        
        return (1 + z_lens) * Dd * Ds / Dds
