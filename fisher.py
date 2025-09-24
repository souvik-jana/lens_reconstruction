"""
Fisher Matrix Analysis for Gravitational Lensing

This module provides Fisher matrix computation for both image plane and source plane
analysis, along with optimization and Jacobian matrix utilities.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad
import time
from typing import Dict, Any, Optional, Tuple, Union
from collections import OrderedDict
import re

# Herculens import
import herculens as hcl
# import lensimage_gw as lensimage_gw

class FisherMatrix:
    """
    Fisher Matrix analysis for gravitational lensing parameter estimation.
    
    This class provides methods for computing Fisher matrices in both image plane
    and source plane, along with optimization and Jacobian utilities.
    """
    
    def __init__(self, prob_model, lens_mass_model, lens_kwargs, gw_source_pos, x_image_true, y_image_true):
        """
        Initialize Fisher matrix analysis.
        
        Parameters:
        -----------
        prob_model : object
            User-provided ProbModel with all parameters and priors defined
        lens_image : object
            Herculens LensImage object for EM analysis
        lens_gw : object, optional
            LensImageGW object for GW analysis
        """
        self.prob_model = prob_model
        self.lens_mass_model = lens_mass_model
        self.lens_kwargs = lens_kwargs
        self.gw_source_pos = gw_source_pos
        self.x_image_true = x_image_true
        self.y_image_true = y_image_true
        self.image_pos_flat = jnp.concatenate([x_image_true, y_image_true])
        # self.lens_gw = lens_gw
        
        # Create loss function using hcl.Loss
        self.loss = hcl.Loss(self.prob_model)
        # self.image_plane_ordered_keys = image_plane_ordered_keys
    
    def compute_fisher_matrix(self, best_fit_params,key_prior):
        """
        Compute Fisher matrix for the given parameters using Hessian method.
        
        Parameters:
        -----------
        best_fit_params : dict
            Best fit parameters in constrained (physical) space
        image_plane : bool
            Whether to include image plane Fisher analysis
        source_plane : bool
            Whether to include source plane Fisher analysis
        gw_analysis : bool
            Whether to include GW Fisher analysis
            
        Returns:
        --------
        dict
            Dictionary containing Fisher matrices and related information
        """
        results = {}
        prior_samples = self.prob_model.sample_prior(5000, key_prior)
        # Use parameters directly in constrained space
        params_const = best_fit_params
        
        # Compute Image plane Fisher matrix
        fisher_matrix_image_plane, param_names_hessian_image_plane = self._compute_fisher_matrix_image_plane(best_fit_params)
        ordered_image_plane_dict, ordered_source_plane_dict = self._get_ordered_best_fit_dict(best_fit_params, param_names_hessian_image_plane)
        covariance_matrix_image_plane = self._compute_covariance_matrix(fisher_matrix_image_plane)
        fim_samples_image_plane, mean_image_plane = self._sample_from_covariance_matrix(covariance_matrix_image_plane, ordered_image_plane_dict, 20000)

        # Image plane analysis
        image_plane_results = {}
        # Add the main Fisher matrix results to image_plane
        image_plane_results['fisher_matrix'] = fisher_matrix_image_plane
        image_plane_results['covariance_matrix'] = covariance_matrix_image_plane
        image_plane_results['row_names'] = param_names_hessian_image_plane
        image_plane_results['fim_samples'] = fim_samples_image_plane
        # results['image_plane_mean'] = mean_image_plane
        image_plane_results['ordered_dict'] = ordered_image_plane_dict
        results['image_plane'] = image_plane_results
        # Source plane analysis
        
        source_plane_results = {}
        fisher_matrix_source_plane, param_names_hessian_source_plane, J = \
            self._compute_fisher_matrix_source_plane(fisher_matrix_image_plane, param_names_hessian_image_plane, self.image_pos_flat)
        
        covariance_matrix_source_plane = self._compute_covariance_matrix(fisher_matrix_source_plane)
        fim_samples_source_plane, mean_source_plane = self._sample_from_covariance_matrix(covariance_matrix_source_plane, ordered_source_plane_dict, 20000)

        source_plane_results['fisher_matrix'] = fisher_matrix_source_plane
        source_plane_results['covariance_matrix'] = covariance_matrix_source_plane
        source_plane_results['row_names'] = param_names_hessian_source_plane
        source_plane_results['fim_samples'] = fim_samples_source_plane
        source_plane_results['fim_samples_coverted'] = self._convert_image_plane_samples_to_source_plane_samples(fim_samples_image_plane, param_names_hessian_image_plane)
        source_plane_results['ordered_dict'] = ordered_source_plane_dict
        # results['source_plane_mean'] = mean_source_plane
        # source_plane_results['ordered_dict'] = ordered_source_plane_dict
        results['source_plane'] = source_plane_results
        results['Jacobian_matrix'] = J
        best_fit_params_unconst = self.prob_model.unconstrain(best_fit_params)
        results['loss_at_best_fit'] = self.loss(best_fit_params_unconst)
        results['prior_samples'] = prior_samples
        
        return results
    
    def _compute_fisher_matrix_image_plane(self, best_fit_params):
        """Compute Fisher matrix using Hessian of the log-likelihood."""
        def loss_constrained(params_const):
            params_unconst = self.prob_model.unconstrain(params_const)
            return self.loss(params_unconst)
        
        def hessian_constrained(params_const):
            return jax.jacfwd(jax.jacrev(loss_constrained))(params_const)
        
        fisher_matrix = hessian_constrained(best_fit_params)  # pytree

        fisher_matrix, unravel_fn = jax.flatten_util.ravel_pytree(fisher_matrix)  # get the array

        pytree_def = unravel_fn.args[0]
        pytree_str = str(pytree_def)
        param_names_hessian = re.findall(r"'([^']+)': \*", pytree_str)
        param_names_hessian = list(dict.fromkeys(param_names_hessian))  # Remove duplicates while preserving order


        n_params = len(param_names_hessian)
        fisher_matrix = fisher_matrix.reshape((n_params, n_params))  # reshape as a matrix

        # Return the Fisher matrix and the ordered parameter names
        return fisher_matrix, param_names_hessian
    
    def _image_plane_to_source_plane(self, image_pos_flat):
        """
        Image plane to source plane transformation.
        
        Parameters:
        -----------
        image_pos_flat : array
            Flattened image plane positions (x1, x2, ..., y1, y2, ...)
            
        Returns:
        --------
        source_x_sum : array
            Sum of source x positions
        source_y_sum : array
            Sum of source y positions
        """
        lens_params = self.lens_kwargs
        source_positions = []
        source_x = []
        source_y = []
        n_images = len(image_pos_flat)//2
        image_x = image_pos_flat[0:n_images]
        image_y = image_pos_flat[n_images:]
        for i in range(len(image_x)):
            source_ = self.lens_mass_model.ray_shooting(image_x[i], image_y[i], lens_params)
            source_positions.append(source_)
            source_x.append(source_[0])
            source_y.append(source_[1])
        source_x_ = jnp.array(source_x)
        source_y_ = jnp.array(source_y)
        source_x_sum = jnp.sum(source_x_)
        source_y_sum = jnp.sum(source_y_)

        return source_x_sum, source_y_sum

    def _transform_image_to_source_plane_keys(self, image_plane_ordered_keys):
        """
        Transform image plane ordered keys to source plane ordered keys.
        
        Replaces image position keys (image_x1, image_x2, ..., image_y1, image_y2, ...) 
        with source position keys (y1, y2).
        
        Parameters:
        -----------
        image_plane_ordered_keys : list
            List of parameter keys in image plane order
            
        Returns:
        --------
        source_plane_ordered_keys : list
            List of parameter keys in source plane order
        """
        #first extract the image keys
        image_keys = [key for key in image_plane_ordered_keys if key.startswith('image_x') or key.startswith('image_y')]
    
        ordered_keys = list(image_plane_ordered_keys)
        start = ordered_keys.index(image_keys[0])
        end = start + len(image_keys)
        new_keys = ordered_keys[:start] + ['y1','y2'] + ordered_keys[start+len(image_keys):]
            
        return new_keys, start, end

    def get_jacobian_matrix(self, image_plane_ordered_keys, image_pos_flat):
        """
        Get Jacobian matrix for image plane to source plane transformation.
        
        Parameters:
        -----------
        image_plane_ordered_keys : list
            List of parameter keys in image plane order
        image_pos_flat : array
            Flattened image plane positions (x1, x2, ..., y1, y2, ...)
            
        Returns:
        --------
        J : array
            Jacobian matrix
        """
        image_block = len(image_pos_flat)
        source_block = 2
        # print("image_plane_ordered_keys:", image_plane_ordered_keys)
        source_plane_ordered_keys, start, end = self._transform_image_to_source_plane_keys(image_plane_ordered_keys)
        nparams_source = len(source_plane_ordered_keys)
        nparams_image = len(image_plane_ordered_keys)
        jac_yx = jax.jacfwd(self._image_plane_to_source_plane)(image_pos_flat)
        jac_yx_matrix = jnp.stack(jac_yx, axis=0)
        jac_xy_matrix = 1 / jac_yx_matrix.T
        # print("jac_xy_matrix:", jac_xy_matrix)
        # Vertically: Top, Middle, Bottom
        # Top block
        J_topleft = jnp.eye(start)
        J_topright = jnp.zeros((start, nparams_source-start))
        J_top = jnp.hstack([J_topleft, J_topright])

        # Middle block
        J_middleleft = jnp.zeros((image_block, start))
        J_middle = jac_xy_matrix
        J_middleright = jnp.zeros((image_block, nparams_source-start-source_block))
        J_middle = jnp.hstack([J_middleleft, J_middle, J_middleright])

        # Bottom block
        J_bottomleft = jnp.zeros((nparams_image-start-image_block, start))
        J_bottommiddle = jnp.zeros((nparams_image-start-image_block, source_block))
        J_bottomright = jnp.eye(nparams_image-start-image_block)
        J_bottom = jnp.hstack([J_bottomleft, J_bottommiddle, J_bottomright])


        J = jnp.vstack([J_top, J_middle, J_bottom])
        return J, source_plane_ordered_keys
    

    def _compute_fisher_matrix_source_plane(self, fisher_matrix_image_plane, image_plane_ordered_keys, image_pos_flat):
        J, source_plane_ordered_keys = self.get_jacobian_matrix(image_plane_ordered_keys, image_pos_flat)
        fisher_matrix_source_plane = J.T @ fisher_matrix_image_plane @ J
        return fisher_matrix_source_plane, source_plane_ordered_keys, J
    
    
    def _compute_covariance_matrix(self, fisher_matrix):
        return jnp.linalg.inv(fisher_matrix)

    def _get_ordered_best_fit_dict(self, best_fit_params, image_plane_ordered_keys):
        """
        Get ordered best fit dictionary.
        
        Parameters:
        -----------
        best_fit_params : dict
            Best fit parameters in constrained (physical) space
        image_plane_ordered_keys : list     
            List of parameter keys in image plane order
        source_plane_ordered_keys : list
            List of parameter keys in source plane order
            
        Returns:
        --------
        best_fit_dict : dict
            Ordered best fit dictionary
        """
        ordered_keys = list(image_plane_ordered_keys)  # preserves numpyro order

        # Validate keys
        missing = [k for k in ordered_keys if k not in best_fit_params]
        extra   = [k for k in best_fit_params if k not in ordered_keys]
        assert not missing, f"Missing in input_params: {missing}"
        # extra are ok, but ignored in the ordered view

        # Dict in numpyro order
        ordered_image_plane_dict = {k: best_fit_params[k] for k in ordered_keys}
        
        # print('image_plane_ordered_keys:', image_plane_ordered_keys)
        #source plane ordered keys
        source_plane_ordered_keys, start, end = self._transform_image_to_source_plane_keys(image_plane_ordered_keys)

        # Build the new dict following new_keys strictly
        y0_gw = self.gw_source_pos[0]
        y1_gw = self.gw_source_pos[1]
        ordered_source_plane_dict = {}
        for k in source_plane_ordered_keys:
            if k == 'y1':
                ordered_source_plane_dict['y1'] = y0_gw
            elif k == 'y2':
                ordered_source_plane_dict['y2'] = y1_gw
            else:
                ordered_source_plane_dict[k] = ordered_image_plane_dict[k]

        return OrderedDict(ordered_image_plane_dict), OrderedDict(ordered_source_plane_dict)

    def _sample_from_covariance_matrix(self, covariance_matrix, ordered_dict, n_samples):
        mean, unravel_fn = jax.flatten_util.ravel_pytree(ordered_dict)
        fim_samples = jax.vmap(unravel_fn)(np.random.multivariate_normal(mean, covariance_matrix, size=n_samples))
        return fim_samples, mean

    def _convert_image_plane_samples_to_source_plane_samples(self, image_plane_samples, image_plane_ordered_keys):
        """
        Convert image plane samples to source plane samples.
        """
        source_plane_ordered_keys, start, end = self._transform_image_to_source_plane_keys(image_plane_ordered_keys)
        y1_samples = []
        y2_samples = []
        image_keys = [image_plane_ordered_keys[i] for i in range(start, end)]
        n_images = len(image_keys)//2
        n_samples = len(image_plane_samples['image_x1'])
        for i in range(n_images):
            image_xi = image_plane_samples[f'image_x{i+1}']
            image_yi = image_plane_samples[f'image_y{i+1}']
            source_ = self.lens_mass_model.ray_shooting(image_xi, image_yi, self.lens_kwargs)
            y1_samples.append(source_[0])
            y2_samples.append(source_[1])
        y1_samples = np.array(y1_samples)
        y2_samples = np.array(y2_samples)
        y1_flattened = y1_samples.flatten()
        y2_flattened = y2_samples.flatten()

        index = np.arange(len(y1_flattened))
        chosen_index = np.random.choice(index, size=n_samples, replace=False)

        y1_samples = y1_flattened[chosen_index]
        y2_samples = y2_flattened[chosen_index]

        source_plane_samples = {}
        for k in source_plane_ordered_keys:
            if k == 'y1':
                source_plane_samples['y1'] = y1_samples
            elif k == 'y2':
                source_plane_samples['y2'] = y2_samples
            else:
                source_plane_samples[k] = image_plane_samples[k]

        return OrderedDict(source_plane_samples)
