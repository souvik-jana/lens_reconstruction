# GW Workflow - Following the same steps as your notebook
# Copy this code into your notebook cells

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms
import herculens as hcl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import time

# =============================================================================
# STEP 1: Define GW ProbModel (similar to your EM ProbModel)
# =============================================================================

class GWProbModel(hcl.NumpyroModel):
    """
    GW ProbModel following the same structure as your EM ProbModel
    """
    
    def __init__(self, gw_observations):
        super().__init__()
        self.gw_observations = gw_observations
    
    def model(self):
        # GW lens parameters (constrained)
        prior_lens = [{
            'theta_E': numpyro.sample('lens_theta_E', dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), 
                transforms.ExpTransform()  # Ensures theta_E > 0
            )),
            'e1': numpyro.sample('lens_e1', dist.Normal(0.0, 0.005)),  # Unconstrained
            'e2': numpyro.sample('lens_e2', dist.Normal(0.0, 0.005)),  # Unconstrained
            'center_x': numpyro.sample('lens_center_x', dist.Normal(0.0, 0.005)),  # Unconstrained
            'center_y': numpyro.sample('lens_center_y', dist.Normal(0.0, 0.005))   # Unconstrained
        }]
        
        # GW-specific parameters (constrained)
        Tstar = numpyro.sample('Tstar', dist.TransformedDistribution(
            dist.Normal(0.0, 1.0), 
            transforms.ExpTransform()  # Ensures Tstar > 0
        ))
        dL = numpyro.sample('dL', dist.TransformedDistribution(
            dist.Normal(10.0, 1.0),  # log(1000) ≈ 6.9, but using 10.0 for larger values
            transforms.ExpTransform()  # Ensures dL > 0
        ))
        
        # Image position parameters (unconstrained - can be negative)
        num_images = self.gw_observations['num_images']
        image_positions = {}
        for i in range(num_images):
            # Prior for image positions (adjust these based on your observations)
            image_positions[f'image_x{i+1}'] = numpyro.sample(f'image_x{i+1}', dist.Normal(0.0, 1.0))
            image_positions[f'image_y{i+1}'] = numpyro.sample(f'image_y{i+1}', dist.Normal(0.0, 1.0))
        
        # Combine all parameters
        params = {
            'lens_theta_E': prior_lens[0]['theta_E'],
            'lens_e1': prior_lens[0]['e1'],
            'lens_e2': prior_lens[0]['e2'],
            'lens_center_x': prior_lens[0]['center_x'],
            'lens_center_y': prior_lens[0]['center_y'],
            'Tstar': Tstar,
            'dL': dL,
            **image_positions  # Add all image position parameters
        }
        
        # GW likelihood
        gw_likelihood = self.gw_likelihood(params)
        
        return gw_likelihood
    
    def gw_likelihood(self, params):
        """GW likelihood function using Herculens MassModel and ray shooting"""
        # Extract parameters
        Tstar = params['Tstar']
        dL = params['dL']
        num_images = self.gw_observations['num_images']
        
        # Extract lens parameters
        theta_E = params['lens_theta_E']
        e1 = params['lens_e1']
        e2 = params['lens_e2']
        center_x = params['lens_center_x']
        center_y = params['lens_center_y']
        
        # Extract image positions from params (these are sampled parameters)
        image_positions = []
        for i in range(num_images):
            x_key = f'image_x{i+1}'
            y_key = f'image_y{i+1}'
            image_positions.append([params[x_key], params[y_key]])
        
        image_positions = jnp.array(image_positions)  # Shape: (num_images, 2)
        
        # Create Herculens MassModel
        lens_mass_model = hcl.MassModel(["SIE"])
        
        # Define lens parameters in Herculens format
        kwargs_lens = [{
            'theta_E': theta_E,
            'e1': e1,
            'e2': e2,
            'center_x': center_x,
            'center_y': center_y
        }]
        
        # Convert image positions to JAX arrays
        im_coords = jnp.array(image_positions)
        
        # Use Herculens ray shooting to calculate source positions
        beta_x, beta_y = lens_mass_model.ray_shooting(
            im_coords[:, 0], im_coords[:, 1], kwargs_lens
        )
        
        # Check consistency: all images should map to the same source position
        mean_beta_x = jnp.mean(beta_x)
        mean_beta_y = jnp.mean(beta_y)
        beta_consistency = jnp.sum((beta_x - mean_beta_x)**2 + (beta_y - mean_beta_y)**2)
        
        # Calculate Fermat potential and magnification using Herculens
        fermat_potential = lens_mass_model.fermat_potential(
            im_coords[:, 0], im_coords[:, 1], kwargs_lens
        )
        magnification = lens_mass_model.magnification(
            im_coords[:, 0], im_coords[:, 1], kwargs_lens
        )
        
        # Calculate arrival times
        arrival_times = fermat_potential * Tstar
        
        # Differentiable sorting of arrival times and corresponding magnifications
        def differentiable_sort(arrival_times, magnifications, temperature=0.1):
            n = len(arrival_times)
            arrival_expanded = jnp.expand_dims(arrival_times, 1)
            arrival_tiled = jnp.tile(arrival_expanded, (1, n))
            arrival_diff = arrival_tiled - arrival_tiled.T
            soft_ranks = jnp.sum(jax.nn.sigmoid(arrival_diff / temperature), axis=1)
            sorting_weights = jax.nn.softmax(-soft_ranks / temperature)
            sorted_arrival_times = jnp.sum(sorting_weights * arrival_times, axis=1)
            sorted_magnifications = jnp.sum(sorting_weights * magnifications, axis=1)
            return sorted_arrival_times, sorted_magnifications
        
        # Sort arrival times and magnifications together
        sorted_arrival_times, sorted_magnifications = differentiable_sort(
            arrival_times, magnification, temperature=0.1
        )
        
        # Calculate time delays and effective luminosity distances
        time_delays = jnp.diff(sorted_arrival_times)
        dL_effectives = dL / jnp.sqrt(jnp.abs(sorted_magnifications))
        
        # Likelihood terms
        time_delay_error = 0.05  # 5% error
        dL_effective_error = 0.05  # 5% error
        
        # Time delay likelihood
        time_delay_likelihood = -0.5 * jnp.sum(
            (time_delays - self.gw_observations['time_delays_true'])**2 / 
            (time_delay_error * self.gw_observations['time_delays_true'])**2
        )
        
        # Effective luminosity distance likelihood
        dL_likelihood = -0.5 * jnp.sum(
            (dL_effectives - self.gw_observations['dL_effectives_true'])**2 / 
            (dL_effective_error * self.gw_observations['dL_effectives_true'])**2
        )
        
        # Source position consistency penalty
        consistency_penalty = -0.5 * beta_consistency / (0.005**2)
        
        return time_delay_likelihood + dL_likelihood + consistency_penalty

# =============================================================================
# STEP 2: Define GW observations (similar to your data)
# =============================================================================

# Define your GW observations
gw_observations = {
    'time_delays_true': jnp.array([0.1, 0.2, 0.15]),  # Observed time delays
    'dL_effectives_true': jnp.array([1000.0, 1200.0, 1100.0, 1300.0]),  # Observed effective dL
    'Tstar': 1.0,  # Time scale parameter
    'dL': 1000.0,  # Original luminosity distance
    'num_images': 4  # Number of images (quad lens)
}

# =============================================================================
# STEP 3: Create GW ProbModel and Loss (similar to your prob_model and loss)
# =============================================================================

# Create GW ProbModel
gw_prob_model = GWProbModel(gw_observations)

# Create GW Loss
gw_loss = hcl.Loss(gw_prob_model)

print("GW ProbModel created successfully!")
print("GW Parameters:", gw_prob_model.get_parameter_names())

# =============================================================================
# STEP 4: Visualize initial guess (similar to your initial guess)
# =============================================================================

# visualize initial guess
key, key_init = jax.random.split(key)
init_params = gw_prob_model.get_sample(key_init)  # constrained space
print("Initial GW params (constrained):", init_params)
init_params_unconst = gw_prob_model.unconstrain(init_params)  # UNconstrained space
print("Initial GW params (unconstrained):", init_params_unconst)

# =============================================================================
# STEP 5: HMC Sampling (similar to your HMC setup)
# =============================================================================

# Setup HMC for GW
@jax.jit
def gw_logdensity_fn(args):
    return -gw_loss(args)

# HMC adaptation
adapt = blackjax.window_adaptation(
    blackjax.nuts, gw_logdensity_fn, 
    target_acceptance_rate=0.8, 
    is_mass_matrix_diagonal=True,
    initial_step_size=1.0,
    progress_bar=True, 
)

num_steps_adaptation = 800
key, key_hmc_adapt, key_hmc_run = jax.random.split(key, 3)

start = time.time()
(last_state, adapted_settings), info = adapt.run(key_hmc_adapt, init_params_unconst, 
                                                 num_steps=num_steps_adaptation)
print("Time taken by GW HMC warmup phase:", time.time()-start)

print("GW Warmup state:", {k: v for k, v in last_state.position.items()})

# =============================================================================
# STEP 6: Main HMC sampling (similar to your main sampling)
# =============================================================================

# setup the NUTS kernel with adapted settings
kernel = blackjax.nuts(gw_logdensity_fn, **adapted_settings).step

# define the inference loop
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos

num_steps_nuts = 3_000  # number of samples

start = time.time()
states, infos = inference_loop(key_hmc_run, kernel, last_state, num_steps_nuts)
_ = states.position['lens_theta_E'].block_until_ready()
print("Time taken by GW HMC main phase:", time.time()-start)

# =============================================================================
# STEP 7: Analyze results (similar to your analysis)
# =============================================================================

# Convert samples back to constrained space
gw_samples = {}
for param_name in gw_prob_model.get_parameter_names():
    gw_samples[param_name] = states.position[param_name]

print("GW sampling completed!")
print("Number of samples:", len(gw_samples['lens_theta_E']))
print("GW parameter means:")
for param_name, values in gw_samples.items():
    print(f"  {param_name}: {jnp.mean(values):.4f} ± {jnp.std(values):.4f}")

# =============================================================================
# STEP 8: Combined Model (EM + GW)
# =============================================================================

class CombinedProbModel(hcl.NumpyroModel):
    """
    Combined EM + GW ProbModel
    """
    
    def __init__(self, gw_observations):
        super().__init__()
        self.gw_observations = gw_observations
    
    def model(self):
        # EM lens parameters (same as your original)
        prior_lens = [{
            'theta_E': numpyro.sample('lens_theta_E', dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), 
                transforms.ExpTransform()
            )),
            'e1': numpyro.sample('lens_e1', dist.Normal(0.0, 0.005)),
            'e2': numpyro.sample('lens_e2', dist.Normal(0.0, 0.005)),
            'center_x': numpyro.sample('lens_center_x', dist.Normal(0.0, 0.005)),
            'center_y': numpyro.sample('lens_center_y', dist.Normal(0.0, 0.005))
        }]
        
        # Source parameters (same as your original)
        prior_source = [{
            'amp': numpyro.sample('source_amp', dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), 
                transforms.ExpTransform()
            )),
            'R_sersic': numpyro.sample('source_R_sersic', dist.TransformedDistribution(
                dist.Normal(-1.6, 1.0), 
                transforms.ExpTransform()
            )),
            'n_sersic': numpyro.sample('source_n_sersic', dist.TransformedDistribution(
                dist.Normal(0.7, 1.0), 
                transforms.ExpTransform()
            )),
            'e1': numpyro.sample('source_e1', dist.Normal(0.05, 0.01)),
            'e2': numpyro.sample('source_e2', dist.Normal(0.05, 0.01)),
            'center_x': numpyro.sample('source_center_x', dist.Normal(0.08, 0.01)),
            'center_y': numpyro.sample('source_center_y', dist.Normal(0.0, 0.01))
        }]
        
        # GW-specific parameters
        Tstar = numpyro.sample('Tstar', dist.TransformedDistribution(
            dist.Normal(0.0, 1.0), 
            transforms.ExpTransform()
        ))
        dL = numpyro.sample('dL', dist.TransformedDistribution(
            dist.Normal(10.0, 1.0), 
            transforms.ExpTransform()
        ))
        
        # Image position parameters
        num_images = self.gw_observations['num_images']
        image_positions = {}
        for i in range(num_images):
            image_positions[f'image_x{i+1}'] = numpyro.sample(f'image_x{i+1}', dist.Normal(0.0, 1.0))
            image_positions[f'image_y{i+1}'] = numpyro.sample(f'image_y{i+1}', dist.Normal(0.0, 1.0))
        
        # Noise parameters
        noise_sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.TransformedDistribution(
            dist.Normal(-4.6, 1.0), 
            transforms.ExpTransform()
        ))
        
        # Combine all parameters
        params = {
            'lens_theta_E': prior_lens[0]['theta_E'],
            'lens_e1': prior_lens[0]['e1'],
            'lens_e2': prior_lens[0]['e2'],
            'lens_center_x': prior_lens[0]['center_x'],
            'lens_center_y': prior_lens[0]['center_y'],
            'source_amp': prior_source[0]['amp'],
            'source_R_sersic': prior_source[0]['R_sersic'],
            'source_n_sersic': prior_source[0]['n_sersic'],
            'source_e1': prior_source[0]['e1'],
            'source_e2': prior_source[0]['e2'],
            'source_center_x': prior_source[0]['center_x'],
            'source_center_y': prior_source[0]['source_center_y'],
            'Tstar': Tstar,
            'dL': dL,
            'noise_sigma_bkg': noise_sigma_bkg,
            **image_positions
        }
        
        # EM likelihood (you need to implement this with your existing EM likelihood)
        em_likelihood = 0.0  # Placeholder - replace with your EM likelihood
        
        # GW likelihood (same as above)
        gw_likelihood = self.gw_likelihood(params)
        
        # Combined likelihood
        total_likelihood = em_likelihood + gw_likelihood
        
        return total_likelihood
    
    def gw_likelihood(self, params):
        """Same GW likelihood as above"""
        # ... (same implementation as above)
        return 0.0  # Placeholder

print("Combined model structure ready!")
print("Next steps:")
print("1. Implement EM likelihood in CombinedProbModel")
print("2. Run combined HMC sampling")
print("3. Compare EM-only vs GW-only vs Combined results")
