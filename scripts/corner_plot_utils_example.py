"""
Example usage of corner_plot_utils module

This file demonstrates how to use the corner plot utilities for creating
professional corner plots with legends, parameter ranges, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from corner_plot_utils import (
    add_corner_legend,
    set_corner_axis_ranges,
    create_corner_ranges,
    add_truth_lines,
    plot_grouped_corner,
    plot_comparison_corner,
    create_default_param_groups
)

# Example 1: Simple corner plot with legend
def example_simple_corner():
    """Create a simple corner plot with a legend."""
    import corner
    
    # Generate some sample data
    samples = {
        'param1': np.random.normal(0, 1, 1000),
        'param2': np.random.normal(0, 1, 1000),
        'param3': np.random.normal(0, 1, 1000),
    }
    
    # Create corner plot
    samples_array = np.column_stack([samples[k] for k in ['param1', 'param2', 'param3']])
    fig = corner.corner(samples_array, labels=['param1', 'param2', 'param3'], color='blue')
    
    # Add legend
    add_corner_legend(fig, labels=['HMC Samples'], colors=['blue'])
    
    plt.show()


# Example 2: Comparison plot (HMC vs Fisher)
def example_comparison_corner():
    """Create a comparison corner plot with two datasets."""
    import corner
    
    # Generate sample data for two methods
    hmc_samples = {
        'theta_E': np.random.normal(2.0, 0.1, 1000),
        'e1': np.random.normal(0.1, 0.05, 1000),
    }
    
    fisher_samples = {
        'theta_E': np.random.normal(2.0, 0.15, 1000),
        'e1': np.random.normal(0.1, 0.08, 1000),
    }
    
    # Create first plot
    params = ['theta_E', 'e1']
    hmc_array = np.column_stack([hmc_samples[p] for p in params])
    fig = corner.corner(hmc_array, labels=params, color='#3B5BA7')
    
    # Overlay second dataset
    fisher_array = np.column_stack([fisher_samples[p] for p in params])
    _ = corner.corner(fisher_array, labels=params, color='#D2691E', fig=fig)
    
    # Add legend
    add_corner_legend(fig, labels=['HMC-EM', 'Fisher-EM'], 
                     colors=['#3B5BA7', '#D2691E'])
    
    plt.show()


# Example 3: Grouped plots with parameter ranges
def example_grouped_plots():
    """Create grouped corner plots with parameter ranges."""
    
    # Sample data
    samples = {
        'lens_theta_E': np.random.normal(2.0, 0.1, 1000),
        'lens_e1': np.random.normal(0.1, 0.05, 1000),
        'lens_e2': np.random.normal(0.05, 0.05, 1000),
        'source_amp': np.random.normal(4.0, 0.5, 1000),
        'source_R_sersic': np.random.normal(0.5, 0.1, 1000),
    }
    
    # Define parameter groups
    param_groups = {
        'lens_mass': ['lens_theta_E', 'lens_e1', 'lens_e2'],
        'source_light': ['source_amp', 'source_R_sersic'],
    }
    
    # Define truth values
    truths_dict = {
        'lens_mass': {'lens_theta_E': 2.0, 'lens_e1': 0.1, 'lens_e2': 0.05},
        'source_light': {'source_amp': 4.0, 'source_R_sersic': 0.5},
    }
    
    # Define parameter ranges
    param_ranges = {
        'lens_theta_E': (1.5, 2.5),
        'lens_e1': (-0.3, 0.3),
        'lens_e2': (-0.3, 0.3),
        'source_amp': (2.0, 6.0),
        'source_R_sersic': (0.2, 0.8),
    }
    
    # Create grouped plots
    figures = plot_grouped_corner(
        samples_dict=samples,
        param_groups=param_groups,
        truths_dict=truths_dict,
        param_ranges=param_ranges,
        color='#2c3e50',
        show_titles=True,
    )
    
    plt.show()


# Example 4: Comparison with grouped plots
def example_comparison_grouped():
    """Create grouped comparison plots (HMC vs Fisher)."""
    
    # Sample data for two methods
    hmc_samples = {
        'lens_theta_E': np.random.normal(2.0, 0.1, 1000),
        'lens_e1': np.random.normal(0.1, 0.05, 1000),
        'source_amp': np.random.normal(4.0, 0.5, 1000),
    }
    
    fisher_samples = {
        'lens_theta_E': np.random.normal(2.0, 0.15, 1000),
        'lens_e1': np.random.normal(0.1, 0.08, 1000),
        'source_amp': np.random.normal(4.0, 0.7, 1000),
    }
    
    # Define parameter groups
    param_groups = {
        'lens_mass': ['lens_theta_E', 'lens_e1'],
        'source_light': ['source_amp'],
    }
    
    # Define truth values
    truths_dict = {
        'lens_mass': {'lens_theta_E': 2.0, 'lens_e1': 0.1},
        'source_light': {'source_amp': 4.0},
    }
    
    # Create comparison plots
    figures = plot_comparison_corner(
        samples_dict1=hmc_samples,
        samples_dict2=fisher_samples,
        param_groups=param_groups,
        labels=('HMC-EM', 'Fisher-EM'),
        colors=('#3B5BA7', '#D2691E'),
        truths_dict=truths_dict,
    )
    
    plt.show()


if __name__ == '__main__':
    print("Corner plot utilities examples")
    print("Uncomment the example you want to run:")
    # example_simple_corner()
    # example_comparison_corner()
    # example_grouped_plots()
    # example_comparison_grouped()

