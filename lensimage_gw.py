# Finalized GW lensing helper for numpyro: init with MassModel only; pass D_dt per call
from functools import partial
import jax
import jax.numpy as jnp
import herculens as hcl
# import astropy.units as u
#from astropy import constants as const

arcsecond_to_radians = 4.84813681109536e-06  #(1*u.arcsecond).to(u.radian).value #4.84814e-6 
Mpc_to_m = 3.085677581491367e+22  #float(1*u.Mpc.to(u.m))
c = 299792458.0  #float(const.c.value)
seconds_to_days = 1.1574074074074073e-05  #float(1*u.s.to(u.day))
class LensImageGW:
    """
    Minimal JAX/numpyro-compatible wrapper around a provided herculens MassModel.
    Instantiate once outside your ProbModel with only the MassModel. Inside your
    probabilistic model, pass all inferred parameters (e.g., D_dt) as arguments
    to the methods. Uses herculens built-ins under the hood.

    Example usage:
      mass_model = hcl.MassModel([hcl.SIE(), hcl.Shear()])
      lens_gw = LensImageGW(mass_model)
      # inside ProbModel.model:
      # D_dt = numpyro.sample('D_dt', dist.LogNormal(...))
      # out = lens_gw.compute(x, y, kwargs_lens, D_dt)
    """

    def __init__(self, mass_model: hcl.MassModel):
        self.mass_model = mass_model

    # @partial(jax.jit, static_argnums=(0,))
    def ray_shoot(self, x, y, kwargs_lens):
        # x = jnp.asarray(x)
        # y = jnp.asarray(y)
        beta_x, beta_y = self.mass_model.ray_shooting(x, y, kwargs_lens)
        return beta_x, beta_y

    # @partial(jax.jit, static_argnums=(0,))
    def lens_potential(self, x, y, kwargs_lens):
        # x = jnp.asarray(x)
        # y = jnp.asarray(y)
        return self.mass_model.potential(x, y, kwargs_lens)

    # @partial(jax.jit, static_argnums=(0,))
    def magnification(self, x, y, kwargs_lens):
        # x = jnp.asarray(x)
        # y = jnp.asarray(y)
        return self.mass_model.magnification(x, y, kwargs_lens)

    # @partial(jax.jit, static_argnums=(0,))
    def fermat_potential(self, x, y, kwargs_lens):
        # it returns fermat potential in arcseconds^2
        # x = jnp.asarray(x)
        # y = jnp.asarray(y)
        # if beta is None:
        return self.mass_model.fermat_potential(x, y, kwargs_lens)
        # beta_x, beta_y = beta
        # return self.mass_model.fermat_potential(x, y, kwargs_lens,
        #                                         beta=(jnp.asarray(beta_x), jnp.asarray(beta_y)))

    # @partial(jax.jit, static_argnums=(0,))
    def time_of_arrival(self, x, y, kwargs_lens, D_dt):
        phi = self.fermat_potential(x, y, kwargs_lens)
        phi_in_radianssq = phi * arcsecond_to_radians**2
        D_dt_in_m = D_dt * Mpc_to_m
        return (D_dt_in_m/c) * phi_in_radianssq

    # @partial(jax.jit, static_argnums=(0,))
    def compute(self, x, y, kwargs_lens, D_dt):
        """
        Convenience method returning a dict with ray-shooted source, potential,
        magnification, Fermat potential and time delay for given D_dt.
        """
        beta_x, beta_y = self.ray_shoot(x, y, kwargs_lens)
        psi = self.lens_potential(x, y, kwargs_lens)
        mu = self.magnification(x, y, kwargs_lens)
        phi = self.fermat_potential(x, y, kwargs_lens)
        
        tarrivals = self.time_of_arrival(x, y, kwargs_lens, D_dt)
        Tstar = ((D_dt * Mpc_to_m)/c)*arcsecond_to_radians**2

        #Sorted according to arrival times
        idx = jnp.argsort(phi)
        phi = phi[idx]
        tarrivals = tarrivals[idx]
        beta_x = beta_x[idx]
        beta_y = beta_y[idx]
        psi = psi[idx]
        mu = mu[idx]
        return {
            'beta_x': beta_x,
            'beta_y': beta_y,
            'psi': psi,
            'mu': mu,
            'phi_in_arcsecsq': phi,
            'Tstar_in_seconds': Tstar,
            'tarrivals_in_seconds': tarrivals,
            'tarrivals_days': tarrivals*seconds_to_days,
            'time_delays_in_seconds': jnp.diff(tarrivals),
            'time_delays_in_days': jnp.diff(tarrivals)*seconds_to_days,
        }

# Notebook-side helper (commented):
# mass_model = hcl.MassModel([hcl.SIE(), hcl.Shear()])
# lens_gw = LensImageGW(mass_model)
# def gw_forward(x_images, y_images, kwargs_lens, D_dt):
#     out = lens_gw.compute(x_images, y_images, kwargs_lens, D_dt)
#     return out['tau'], out['mu']
