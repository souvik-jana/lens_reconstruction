# GWEMFISH Input Section

This document describes the input functionality of the GWEMFISH package for defining lens models and parameters.

## Overview

The input section provides a clean, user-friendly interface for:
- Defining lens models and their parameters
- Setting up parameter bounds and priors
- Managing fixed vs free parameters
- Validating input configurations

## Components

### 1. LensInput (`gwemfish.input.lens_input`)

Main class for defining lens models and their configuration.

**Key Features:**
- Support for multiple lens models (SIE, SHEAR, NFW, etc.)
- Source and lens light model definition
- Cosmology parameter handling
- Observation parameter setup
- Image position management

**Example Usage:**
```python
from gwemfish.input.lens_input import LensInput

# Create lens input
lens_input = LensInput()

# Add SIE lens model
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

# Add external shear
lens_input.add_lens_model(
    model_type="SHEAR",
    parameters={
        'gamma1': -3.09e-5,
        'gamma2': 9.51e-5
    }
)

# Set source
lens_input.set_source(
    model_type="SersicElliptic",
    parameters={
        'amp': 4.0,
        'R_sersic': 0.5,
        'n_sersic': 2.0
    },
    position=(0.3, 0.2),
    redshift=2.0
)

# Set cosmology
lens_input.set_cosmology(
    H0=67.3,
    Om0=0.316,
    use_as_parameters=True
)

# Set observation parameters
lens_input.set_observation(
    pixel_scale=0.08,
    image_size=(200, 200),
    psf_fwhm=0.3
)

# Set image positions
image_positions = [(0.94, 5.33), (2.90, -4.22), (5.02, -0.44), (-4.26, -1.10)]
lens_input.set_image_positions(image_positions)

# Validate input
lens_input.validate_input()
```

### 2. ParameterInput (`gwemfish.input.parameter_input`)

Handles parameter definitions, bounds, and priors.

**Key Features:**
- Parameter bounds and prior distributions
- Fixed vs free parameter management
- Parameter grouping (cosmology, lens, source, etc.)
- Automatic parameter validation

**Example Usage:**
```python
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

# Add image position parameters
image_positions = [(0.94, 5.33), (2.90, -4.22), (5.02, -0.44), (-4.26, -1.10)]
param_input.add_image_position_parameters(image_positions)

# Add source parameters
param_input.add_source_parameters(
    model_type="SersicElliptic",
    parameters={
        'amp': 4.0,
        'R_sersic': 0.5,
        'n_sersic': 2.0
    }
)

# Add redshift parameters
param_input.add_redshift_parameters(
    z_lens=0.5,
    z_source=2.0
)

# Add noise parameter
param_input.add_noise_parameter(background_rms=0.01)
```

### 3. SupportedLensModels (`gwemfish.models.lens_models`)

Registry of supported lens models and their parameter requirements.

**Key Features:**
- 27 supported lens models
- Parameter validation
- Model information lookup
- Parameter requirement checking

**Example Usage:**
```python
from gwemfish.models.lens_models import SupportedLensModels

# Get all supported models
models = SupportedLensModels.get_supported_models()
print(f"Supported models: {models}")

# Get model information
sie_info = SupportedLensModels.get_model_info('SIE')
print(f"SIE parameters: {sie_info.parameters}")

# Validate parameters
valid_params = {
    'theta_E': 5.0,
    'e1': 0.137,
    'e2': 0.039,
    'center_x': 0.0,
    'center_y': 0.0
}
SupportedLensModels.validate_parameters('SIE', valid_params)
```

## Supported Lens Models

The package supports 27 different lens models:

1. **SIE** - Singular Isothermal Ellipsoid
2. **SIS** - Singular Isothermal Sphere
3. **NFW** - Navarro-Frenk-White profile
4. **SHEAR** - External shear
5. **CONVERGENCE** - External convergence
6. **POINT_MASS** - Point mass
7. **SPEMD** - Singular Power-law Elliptical Mass Distribution
8. **SPEP** - Singular Power-law Elliptical Potential
9. **CHAMELEON** - Chameleon profile
10. **DPL** - Dual Pseudo Isothermal Elliptical
11. **GAUSSIAN** - Gaussian profile
12. **HERNQUIST** - Hernquist profile
13. **JAFFE** - Jaffe profile
14. **MULTI_GAUSSIAN** - Multi-Gaussian profile
15. **PJAFFE** - Projected Jaffe profile
16. **PJAFFE_ELLIPSE** - Projected Jaffe Elliptical profile
17. **POWER_LAW** - Power-law profile
18. **POWER_LAW_ELLIPSE** - Power-law Elliptical profile
19. **POWER_LAW_ELLIPSE_CORE** - Power-law Elliptical Core profile
20. **SERSIC** - Sersic profile
21. **SERSIC_ELLIPSE** - Sersic Elliptical profile
22. **SERSIC_ELLIPSE_CORE** - Sersic Elliptical Core profile
23. **SPP** - Singular Power-law Potential
24. **TNFW** - Truncated Navarro-Frenk-White profile
25. **TNFW_ELLIPSE** - Truncated Navarro-Frenk-White Elliptical profile
26. **TNFW_ELLIPSE_CORE** - Truncated Navarro-Frenk-White Elliptical Core profile
27. **UNIFORM** - Uniform convergence

## Parameter Types

### Cosmology Parameters
- `H0`: Hubble constant (km/s/Mpc)
- `Om0`: Matter density parameter
- `Ode0`: Dark energy density parameter (optional, defaults to 1-Om0)

### Lens Parameters
- `theta_E`: Einstein radius
- `e1`, `e2`: Ellipticity components
- `center_x`, `center_y`: Center coordinates
- `gamma1`, `gamma2`: Shear components
- `Rs`, `alpha_Rs`: NFW scale radius and strength
- And many more depending on the model

### Source Parameters
- `amp`: Amplitude
- `R_sersic`: Sersic radius
- `n_sersic`: Sersic index
- `e1`, `e2`: Ellipticity components
- `center_x`, `center_y`: Center coordinates

### Image Position Parameters
- `image_x1`, `image_y1`: First image position
- `image_x2`, `image_y2`: Second image position
- And so on for each image

### Redshift Parameters
- `zl`: Lens redshift
- `zs`: Source redshift

### Noise Parameters
- `noise_sigma_bkg`: Background noise RMS

## Running Tests

To test the input functionality:

```bash
cd GWEMFISH
python test_input.py
```

To run the examples:

```bash
python examples/input_example.py
```

## Next Steps

1. **Define your lens model** using `LensInput`
2. **Set up parameters** using `ParameterInput`
3. **Validate your input** with `lens_input.validate_input()`
4. **Extract parameters** for analysis
5. **Move to the next section** (data simulation, likelihood computation, etc.)

The input section provides a solid foundation for the rest of the GWEMFISH pipeline. Once you have your lens model and parameters defined, you can proceed to the data simulation and likelihood computation sections.


