# Combined EM + GW Likelihood Implementation
# Copy this code into a new cell in your notebook

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms
import herculens as hcl

def loglike_gw(params, gw_observations):
    """
    GW likelihood function using Herculens MassModel and ray shooting
    
    Parameters:
    -----------
    params : dict
        Dictionary containing lens parameters from prob_model AND image positions:
        - lens_theta_E, lens_e1, lens_e2, lens_center_x, lens_center_y: lens parameters
        - image_x1, image_y1, image_x2, image_y2, ...: image positions (sampled parameters)
    gw_observations : dict
        Dictionary containing GW observations:
        - time_delays_true: observed time delays
        - dL_effectives_true: observed effective luminosity distances
        - Tstar: time scale parameter
        - dL: original luminosity distance
        - num_images: number of images (e.g., 4 for quad lens)
    """
    # Extract parameters
    Tstar = gw_observations['Tstar']
    dL = gw_observations['dL']
    num_images = gw_observations['num_images']
    
    # Extract lens parameters from prob_model
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
    im_coords = jnp.array(image_positions)  # Shape: (4, 2)
    
    # Use Herculens ray shooting to calculate source positions
    # Ray shooting: beta = theta - alpha(theta)
    beta_x, beta_y = lens_mass_model.ray_shooting(
        im_coords[:, 0], im_coords[:, 1], kwargs_lens
    )
    
    # Check consistency: all images should map to the same source position
    # Calculate the mean source position
    mean_beta_x = jnp.mean(beta_x)
    mean_beta_y = jnp.mean(beta_y)
    
    # Calculate consistency penalty: sum of squared deviations from mean
    beta_consistency = jnp.sum((beta_x - mean_beta_x)**2 + (beta_y - mean_beta_y)**2)
    
    # Calculate Fermat potential using Herculens
    fermat_potential = lens_mass_model.fermat_potential(
        im_coords[:, 0], im_coords[:, 1], kwargs_lens
    )
    
    # Calculate magnification using Herculens
    magnification = lens_mass_model.magnification(
        im_coords[:, 0], im_coords[:, 1], kwargs_lens
    )
    
    # Calculate arrival times
    arrival_times = fermat_potential * Tstar
    
    # Differentiable sorting of arrival times and corresponding magnifications
    def differentiable_sort(arrival_times, magnifications, temperature=0.1):
        """
        Sort arrival times and corresponding magnifications in a differentiable way
        """
        n = len(arrival_times)
        
        # Create pairwise differences for arrival times
        arrival_expanded = jnp.expand_dims(arrival_times, 1)  # (n, 1)
        arrival_tiled = jnp.tile(arrival_expanded, (1, n))  # (n, n)
        arrival_diff = arrival_tiled - arrival_tiled.T  # (n, n)
        
        # Create soft ranking (how many elements each arrival time is larger than)
        soft_ranks = jnp.sum(jax.nn.sigmoid(arrival_diff / temperature), axis=1)
        
        # Create sorting matrix using softmax
        # Each row represents the probability of being at each position
        sorting_weights = jax.nn.softmax(-soft_ranks / temperature)  # (n, n)
        
        # Sort arrival times
        sorted_arrival_times = jnp.sum(sorting_weights * arrival_times, axis=1)
        
        # Sort magnifications using the same weights
        sorted_magnifications = jnp.sum(sorting_weights * magnifications, axis=1)
        
        return sorted_arrival_times, sorted_magnifications
    
    # Sort arrival times and magnifications together
    sorted_arrival_times, sorted_magnifications = differentiable_sort(
        arrival_times, magnification, temperature=0.1
    )
    
    # Calculate time delays (differences between consecutive sorted arrival times)
    time_delays = jnp.diff(sorted_arrival_times)
    
    # Calculate effective luminosity distances using sorted magnifications
    dL_effectives = dL / jnp.sqrt(jnp.abs(sorted_magnifications))
    
    # Likelihood terms
    time_delay_error = 0.05  # 5% error
    dL_effective_error = 0.05  # 5% error
    
    # Time delay likelihood
    time_delay_likelihood = -0.5 * jnp.sum(
        (time_delays - gw_observations['time_delays_true'])**2 / 
        (time_delay_error * gw_observations['time_delays_true'])**2
    )
    
    # Effective luminosity distance likelihood
    dL_likelihood = -0.5 * jnp.sum(
        (dL_effectives - gw_observations['dL_effectives_true'])**2 / 
        (dL_effective_error * gw_observations['dL_effectives_true'])**2
    )
    
    # Source position consistency penalty
    consistency_penalty = -0.5 * beta_consistency / (0.005**2)
    
    return time_delay_likelihood + dL_likelihood + consistency_penalty


# Combined EM + GW ProbModel
class CombinedProbModel(hcl.NumpyroModel):
    
    def __init__(self, gw_observations):
        """
        Initialize combined model with GW observations
        
        Parameters:
        -----------
        gw_observations : dict
            Dictionary containing GW observations
        """
        super().__init__()
        self.gw_observations = gw_observations
    
    def model(self):
        # EM lens parameters with proper constraints
        prior_lens = [{
            # Constrained parameters (use transforms)
            'theta_E': numpyro.sample('lens_theta_E', dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), 
                transforms.ExpTransform()  # Ensures theta_E > 0
            )),
            'e1': numpyro.sample('lens_e1', dist.Normal(0.0, 0.005)),  # Unconstrained
            'e2': numpyro.sample('lens_e2', dist.Normal(0.0, 0.005)),  # Unconstrained
            'center_x': numpyro.sample('lens_center_x', dist.Normal(0.0, 0.005)),  # Unconstrained
            'center_y': numpyro.sample('lens_center_y', dist.Normal(0.0, 0.005))   # Unconstrained
        }]
        
        # Source parameters with proper constraints
        prior_source = [{
            # Constrained parameters (must be positive)
            'amp': numpyro.sample('source_amp', dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), 
                transforms.ExpTransform()  # Ensures amp > 0
            )),
            'R_sersic': numpyro.sample('source_R_sersic', dist.TransformedDistribution(
                dist.Normal(-1.6, 1.0),  # log(0.2) ≈ -1.6
                transforms.ExpTransform()  # Ensures R_sersic > 0
            )),
            'n_sersic': numpyro.sample('source_n_sersic', dist.TransformedDistribution(
                dist.Normal(0.7, 1.0),  # log(2.0) ≈ 0.7
                transforms.ExpTransform()  # Ensures n_sersic > 0
            )),
            # Unconstrained parameters
            'e1': numpyro.sample('source_e1', dist.Normal(0.05, 0.01)),
            'e2': numpyro.sample('source_e2', dist.Normal(0.05, 0.01)),
            'center_x': numpyro.sample('source_center_x', dist.Normal(0.08, 0.01)),
            'center_y': numpyro.sample('source_center_y', dist.Normal(0.0, 0.01))
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
        
        # Noise parameters (constrained)
        noise_sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.TransformedDistribution(
            dist.Normal(-4.6, 1.0),  # log(1e-2) ≈ -4.6
            transforms.ExpTransform()  # Ensures noise_sigma_bkg > 0
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
            'source_center_y': prior_source[0]['center_y'],
            'Tstar': Tstar,
            'dL': dL,
            'noise_sigma_bkg': noise_sigma_bkg,
            **image_positions  # Add all image position parameters
        }
        
        # EM likelihood (you need to implement this)
        em_likelihood = 0.0  # Placeholder - replace with your EM likelihood
        
        # GW likelihood
        gw_likelihood = loglike_gw(params, self.gw_observations)
        
        # Combined likelihood
        total_likelihood = em_likelihood + gw_likelihood
        
        return total_likelihood


# Example usage:
# Define your GW observations
gw_observations = {
    'time_delays_true': jnp.array([0.1, 0.2, 0.15]),  # Observed time delays
    'dL_effectives_true': jnp.array([1000.0, 1200.0, 1100.0, 1300.0]),  # Observed effective dL
    'Tstar': 1.0,  # Time scale parameter
    'dL': 1000.0,  # Original luminosity distance
    'num_images': 4  # Number of images (quad lens)
}

# Create combined model
# combined_model = CombinedProbModel(gw_observations)
# combined_loss = hcl.Loss(combined_model)

print("Combined likelihood code ready to copy into your notebook!")
