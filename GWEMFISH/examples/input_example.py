#!/usr/bin/env python3
"""
Example of using GWEMFISH input functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gwemfish.input.lens_input import LensInput
from gwemfish.input.parameter_input import ParameterInput
from gwemfish.models.lens_models import SupportedLensModels


def example_sie_shear_lens():
    """Example: SIE + SHEAR lens model."""
    print("Example: SIE + SHEAR Lens Model")
    print("=" * 40)
    
    # Create lens input
    lens_input = LensInput()
    
    # Add SIE lens model
    lens_input.add_lens_model(
        model_type="SIE",
        parameters={
            'theta_E': 5.0,      # Einstein radius
            'e1': 0.137,         # Ellipticity component 1
            'e2': 0.039          # Ellipticity component 2
        },
        fixed_parameters={
            'center_x': 0.0,     # Fixed at origin
            'center_y': 0.0
        }
    )
    
    # Add external shear
    lens_input.add_lens_model(
        model_type="SHEAR",
        parameters={
            'gamma1': -3.09e-5,  # Shear component 1
            'gamma2': 9.51e-5    # Shear component 2
        },
        fixed_parameters={
            'ra_0': 0.0,         # Shear center
            'dec_0': 0.0
        }
    )
    
    # Set source
    lens_input.set_source(
        model_type="SersicElliptic",
        parameters={
            'amp': 4.0,          # Amplitude
            'R_sersic': 0.5,     # Sersic radius
            'n_sersic': 2.0,     # Sersic index
            'e1': 0.05,          # Source ellipticity
            'e2': 0.05
        },
        fixed_parameters={
            'center_x': 0.05,    # Source position
            'center_y': 0.1
        },
        position=(0.3, 0.2),     # Source plane position
        redshift=2.0
    )
    
    # Set lens light
    lens_input.set_lens_light(
        model_type="SersicElliptic",
        parameters={
            'amp': 8.0,
            'R_sersic': 1.0,
            'n_sersic': 3.0,
            'e1': 0.137,         # Aligned with lens mass
            'e2': 0.039
        },
        fixed_parameters={
            'center_x': 0.0,
            'center_y': 0.0
        }
    )
    
    # Set cosmology
    lens_input.set_cosmology(
        H0=67.3,
        Om0=0.316,
        use_as_parameters=True  # Treat as free parameters
    )
    
    # Set observation parameters
    lens_input.set_observation(
        pixel_scale=0.08,        # arcsec/pixel
        image_size=(200, 200),   # pixels
        psf_fwhm=0.3,           # arcsec
        background_rms=0.01
    )
    
    # Set image positions (computed from source position)
    image_positions = [
        (0.94, 5.33),           # Image 1
        (2.90, -4.22),          # Image 2
        (5.02, -0.44),          # Image 3
        (-4.26, -1.10)          # Image 4
    ]
    lens_input.set_image_positions(image_positions)
    
    # Print summary
    lens_input.print_summary()
    
    return lens_input


def example_parameter_setup():
    """Example: Parameter input setup."""
    print("\n\nExample: Parameter Input Setup")
    print("=" * 40)
    
    # Create parameter input
    param_input = ParameterInput()
    
    # Add cosmology parameters
    param_input.add_cosmology_parameters(
        H0=67.3,
        Om0=0.316,
        use_as_parameters=True
    )
    
    # Add lens parameters
    param_input.add_lens_parameters(
        model_type="SIE",
        parameters={
            'theta_E': 5.0,
            'e1': 0.137,
            'e2': 0.039
        },
        fixed_parameters={
            'center_x': 0.0,
            'center_y': 0.0
        },
        model_index=0
    )
    
    param_input.add_lens_parameters(
        model_type="SHEAR",
        parameters={
            'gamma1': -3.09e-5,
            'gamma2': 9.51e-5
        },
        fixed_parameters={
            'ra_0': 0.0,
            'dec_0': 0.0
        },
        model_index=1
    )
    
    # Add image position parameters
    image_positions = [
        (0.94, 5.33),
        (2.90, -4.22),
        (5.02, -0.44),
        (-4.26, -1.10)
    ]
    param_input.add_image_position_parameters(image_positions)
    
    # Add source parameters
    param_input.add_source_parameters(
        model_type="SersicElliptic",
        parameters={
            'amp': 4.0,
            'R_sersic': 0.5,
            'n_sersic': 2.0,
            'e1': 0.05,
            'e2': 0.05
        },
        fixed_parameters={
            'center_x': 0.05,
            'center_y': 0.1
        }
    )
    
    # Add lens light parameters
    param_input.add_lens_light_parameters(
        model_type="SersicElliptic",
        parameters={
            'amp': 8.0,
            'R_sersic': 1.0,
            'n_sersic': 3.0,
            'e1': 0.137,
            'e2': 0.039
        },
        fixed_parameters={
            'center_x': 0.0,
            'center_y': 0.0
        }
    )
    
    # Add redshift parameters
    param_input.add_redshift_parameters(
        z_lens=0.5,
        z_source=2.0
    )
    
    # Add noise parameter
    param_input.add_noise_parameter(background_rms=0.01)
    
    # Print summary
    param_input.print_summary()
    
    return param_input


def example_lens_model_info():
    """Example: Lens model information."""
    print("\n\nExample: Lens Model Information")
    print("=" * 40)
    
    # Print all supported models
    print("All supported lens models:")
    models = SupportedLensModels.get_supported_models()
    for i, model in enumerate(models):
        print(f"  {i+1:2d}. {model}")
    
    # Get detailed info for SIE
    print(f"\nDetailed information for SIE:")
    SupportedLensModels.print_model_info('SIE')
    
    # Test parameter validation
    print(f"\nParameter validation example:")
    valid_params = {
        'theta_E': 5.0,
        'e1': 0.137,
        'e2': 0.039,
        'center_x': 0.0,
        'center_y': 0.0
    }
    
    try:
        SupportedLensModels.validate_parameters('SIE', valid_params)
        print("✓ Valid SIE parameters accepted")
    except ValueError as e:
        print(f"✗ Valid SIE parameters rejected: {e}")
    
    # Test invalid parameters
    invalid_params = {
        'theta_E': 5.0,
        'invalid_param': 1.0
    }
    
    try:
        SupportedLensModels.validate_parameters('SIE', invalid_params)
        print("✗ Invalid SIE parameters accepted (should be rejected)")
    except ValueError as e:
        print(f"✓ Invalid SIE parameters correctly rejected: {e}")


def main():
    """Run all examples."""
    print("GWEMFISH Input Examples")
    print("=" * 50)
    
    # Run examples
    lens_input = example_sie_shear_lens()
    param_input = example_parameter_setup()
    example_lens_model_info()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nNext steps:")
    print("1. Use lens_input to define your lens model")
    print("2. Use param_input to set up parameter bounds and priors")
    print("3. Validate your input with lens_input.validate_input()")
    print("4. Extract parameters for analysis")


if __name__ == "__main__":
    main()
