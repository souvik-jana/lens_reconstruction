import numpyro
import jax
import jax.numpy as jnp
from numpyro.distributions import transforms
from numpyro.infer.util import potential_energy
import numpyro.distributions as dist
from numpyro import infer
from numpyro.handlers import seed
from numpyro.infer.util import potential_energy # To compute loss
from numpyro.infer.util import log_likelihood # To compute log likelihood
from numpyro.distributions import constraints # To set constraints on variables
from numpyro.infer.util import log_density # To compute log density
from numpyro.infer import util
import copy
from numpyro import handlers



# NUTS Hamiltonian MC sampling
import blackjax


class NumpyroModelBase:
    """Base class with common NumPyro model utilities"""
    
    def num_parameters(self):
        """Count the number of model parameters (excluding observations)"""
        # Execute model once to get all sample sites
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(self.model).get_trace()
        
        # Count parameters (sample sites that are NOT observations)
        n_params = 0
        for site_name, site in trace.items():
            if site['type'] == 'sample' and not site.get('is_observed', False):
                # Get the shape of this parameter
                value = site['value']
                if isinstance(value, (int, float)):
                    n_params += 1
                else:
                    n_params += int(jnp.prod(jnp.array(value.shape)))
        
        return n_params

    def log_posterior(self, params):
        """Compute log probability at given parameters using pure NumPyro"""
        
        # Compute log posterior (prior + likelihood)
        log_post, _ = log_density(
            self.model,      # Your model function
            (),              # model_args
            {},              # model_kwargs
            params           # The parameters to evaluate at
        )
        
        return log_post

    def log_likelihood_components(self, params, prng_key):
        """Compute prior, likelihood, and posterior at given parameters"""

        # Generate a trace with all parameters fixed to the input values
        with numpyro.handlers.substitute(data=params):
            seeded_model = handlers.seed(self.model, prng_key)
            trace = numpyro.handlers.trace(seeded_model).get_trace()
            # with numpyro.handlers.seed(rng_seed=prng_key):
            #     trace = numpyro.handlers.trace(self.model).get_trace()
        
        log_prior = 0.0
        log_likelihood = 0.0
        likelihood_components = {}
        
        for site_name, site in trace.items():
            if site['type'] == 'sample':
                log_prob = jnp.sum(site['fn'].log_prob(site['value']))
                
                if site.get('is_observed', False):
                    log_likelihood += log_prob
                    likelihood_components[site_name] = log_prob
                else:
                    log_prior += log_prob
        
        return {
            'log_prior': log_prior,
            'log_likelihood': log_likelihood,
            'log_posterior': log_prior + log_likelihood,
            'likelihood_components': likelihood_components
        }
    def log_likelihood(self, params):
        """Compute log likelihood at given parameters using pure NumPyro"""
        ll = log_likelihood(self.model,params)
        total_ll = sum(ll.values())
        return total_ll
    
    def potential_energy(self, unconstrained_params):
        """
        Compute potential energy in unconstrained space
        This is same as -log posterior in constrained space
        loss = -log p(x) = -log p(z) - log |det J|, where z = T(x)
        where z is the unconstrained parameter and x is the constrained parameter
        and T is the transform from constrained to unconstrained space
        and J is the Jacobian of the transform T
        ----------------------------------------------------------------------
        Parameters:
        unconstrained_params: dict
            Dictionary of unconstrained parameter values
        ----------------------------------------------------------------------
        Returns:
        pe: float
            Potential energy at the given unconstrained parameters
        ----------------------------------------------------------------------
        Note: This requires the parameters to be in unconstrained space
        ----------------------------------------------------------------------  
        """
        pe = potential_energy(
            self.model,
            (),
            {},
            unconstrained_params
        )
        return pe

    def get_sample_sites(self):
        """Get the names of all sample sites in the model"""
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(self.model).get_trace()
        
        sample_sites = [
            site_name for site_name, site in trace.items()
            if site['type'] == 'sample'
        ]
        
        return sample_sites
     
    def get_observed_sites(self):
        """Get the names of all observed sites in the model"""
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(self.model).get_trace()
        
        observed_sites = [
            site_name for site_name, site in trace.items()
            if site['type'] == 'sample' and site.get('is_observed', False)
        ]
        
        return observed_sites   
    
    def constrain(self, params):
        return util.constrain_fn(self.model, (), {}, params)

    def unconstrain(self, params):
        return util.unconstrain_fn(self.model, (), {}, params)

    def seeded_model(self, prng_key):
        return handlers.seed(self.model, prng_key)
    
    def get_trace(self, prng_key):
        return handlers.trace(self.seeded_model(prng_key)).get_trace()

    def get_sample(self, prng_key=None):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        trace = self.get_trace(prng_key)
        return {site['name']: site['value'] for site in trace.values() if not site.get('is_observed', False)}

    def sample_prior(self, num_samples, prng_key=None):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        batch_ndims = 0 if num_samples else 1
        predictive = util.Predictive(self.model, 
                                     num_samples=num_samples, 
                                     batch_ndims=batch_ndims)
        samples = predictive(prng_key)
        # delete all sites whose key contain 'obs'
        sites_keys = copy.deepcopy(list(samples.keys()))
        for key in sites_keys:
            if 'obs' in key:
                del samples[key]
        return samples

    def render_model(self):
        return numpyro.render_model(self.model)



    