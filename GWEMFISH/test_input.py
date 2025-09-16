#!/usr/bin/env python3
"""
Test script for GWEMFISH input functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gwemfish'))

def test_lens_input():
    """Test lens input functionality."""
    print("Testing lens input...")
    
    try:
        from gwemfish.input.lens_input import LensInput
        
        # Create lens input
        lens_input = LensInput()
        
        # Add lens models
        lens_input.add_lens_model(
            model_type="SIE",
            parameters={
                'theta_E': 5.0,
                'e1': 0.137,
                'e2': 0.039
            },
            fixed_parameters={
                'center_x': 0.0,
                'center_y': 0.0
            }
        )
        
        lens_input.add_lens_model(
            model_type="SHEAR",
            parameters={
                'gamma1': -3.09e-5,
                'gamma2': 9.51e-5
            },
            fixed_parameters={
                'ra_0': 0.0,
                'dec_0': 0.0
            }
        )
        
        # Set source
        lens_input.set_source(
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
            },
            position=(0.3, 0.2),
            redshift=2.0
        )
        
        # Set lens light
        lens_input.set_lens_light(
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
        
        # Set cosmology
        lens_input.set_cosmology(
            H0=67.3,
            Om0=0.316,
            use_as_parameters=True
        )
        
        # Set observation
        lens_input.set_observation(
            pixel_scale=0.08,
            image_size=(200, 200),
            psf_fwhm=0.3,
            background_rms=0.01
        )
        
        # Set image positions (mock)
        image_positions = [
            (0.94, 5.33),
            (2.90, -4.22),
            (5.02, -0.44),
            (-4.26, -1.10)
        ]
        lens_input.set_image_positions(image_positions)
        
        # Validate input
        lens_input.validate_input()
        print("✓ Lens input validation successful")
        
        # Test parameter extraction
        param_names = lens_input.get_all_parameter_names()
        print(f"✓ Parameter names extracted: {len(param_names)} parameters")
        
        fixed_params = lens_input.get_fixed_parameters()
        print(f"✓ Fixed parameters extracted: {len(fixed_params)} parameters")
        
        # Test model info
        lens_models = lens_input.get_lens_model_list()
        print(f"✓ Lens models: {lens_models}")
        
        lens_kwargs = lens_input.get_lens_kwargs()
        print(f"✓ Lens kwargs: {len(lens_kwargs)} models")
        
        source_kwargs = lens_input.get_source_kwargs()
        print(f"✓ Source kwargs: {len(source_kwargs)} parameters")
        
        # Print summary
        lens_input.print_summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Lens input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_input():
    """Test parameter input functionality."""
    print("\nTesting parameter input...")
    
    try:
        from gwemfish.input.parameter_input import ParameterInput
        
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
        
        # Test parameter extraction
        all_values = param_input.get_parameter_values()
        print(f"✓ All parameter values: {len(all_values)} parameters")
        
        free_values = param_input.get_free_parameter_values()
        print(f"✓ Free parameter values: {len(free_values)} parameters")
        
        fixed_values = param_input.get_fixed_parameter_values()
        print(f"✓ Fixed parameter values: {len(fixed_values)} parameters")
        
        bounds = param_input.get_parameter_bounds()
        print(f"✓ Parameter bounds: {len(bounds)} parameters")
        
        priors = param_input.get_parameter_priors()
        print(f"✓ Parameter priors: {len(priors)} parameters")
        
        # Test parameter groups
        cosmology_params = param_input.get_parameters_by_group('cosmology')
        print(f"✓ Cosmology parameters: {cosmology_params}")
        
        lens_params = param_input.get_parameters_by_group('lens')
        print(f"✓ Lens parameters: {len(lens_params)} parameters")
        
        # Print summary
        param_input.print_summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lens_models():
    """Test lens models functionality."""
    print("\nTesting lens models...")
    
    try:
        from gwemfish.models.lens_models import SupportedLensModels
        
        # Test getting supported models
        models = SupportedLensModels.get_supported_models()
        print(f"✓ Supported models: {len(models)} models")
        print(f"  Sample models: {models[:5]}...")
        
        # Test getting model info
        sie_info = SupportedLensModels.get_model_info('SIE')
        print(f"✓ SIE model info: {sie_info.name} - {sie_info.description}")
        
        # Test getting model parameters
        sie_params = SupportedLensModels.get_model_parameters('SIE')
        print(f"✓ SIE parameters: {sie_params}")
        
        # Test parameter validation
        valid_params = {
            'theta_E': 5.0,
            'e1': 0.137,
            'e2': 0.039,
            'center_x': 0.0,
            'center_y': 0.0
        }
        SupportedLensModels.validate_parameters('SIE', valid_params)
        print("✓ SIE parameter validation successful")
        
        # Test invalid parameters
        try:
            invalid_params = {
                'theta_E': 5.0,
                'invalid_param': 1.0
            }
            SupportedLensModels.validate_parameters('SIE', invalid_params)
            print("✗ Should have failed with invalid parameter")
            return False
        except ValueError:
            print("✓ Invalid parameter correctly rejected")
        
        # Test missing required parameters
        try:
            incomplete_params = {
                'e1': 0.137,
                'e2': 0.039
            }
            SupportedLensModels.validate_parameters('SIE', incomplete_params)
            print("✗ Should have failed with missing required parameter")
            return False
        except ValueError:
            print("✓ Missing required parameter correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Lens models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("GWEMFISH Input Section Test")
    print("=" * 50)
    
    tests = [
        test_lens_input,
        test_parameter_input,
        test_lens_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All input tests passed! Input section is working correctly.")
        print("\nFeatures tested:")
        print("  ✓ Lens model input handling")
        print("  ✓ Parameter input and validation")
        print("  ✓ Supported lens models registry")
        print("  ✓ Parameter bounds and priors")
        print("  ✓ Fixed vs free parameter handling")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
