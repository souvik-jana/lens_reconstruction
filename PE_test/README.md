# Gravitational Lensing Parameter Estimation Analysis

This folder contains a complete Python implementation of gravitational lensing parameter estimation analysis, converted from the `herculens__Starting_guide.ipynb` notebook.

## Overview

The analysis performs a comprehensive parameter estimation study including:

1. **Simulation** of an HST-like observation of a strong lens
2. **Optimization** to find best-fit parameters using BFGS
3. **Fisher Information Matrix (FIM)** estimation for parameter uncertainties
4. **Stochastic Variational Inference (SVI)** for analytical posterior approximation
5. **Hamiltonian Monte Carlo (HMC)** sampling for full posterior exploration
6. **Comparison** of all methods with corner plots

## Files

- `herculens_pe_analysis.py` - Main analysis script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Output

The script automatically creates an `output/` directory containing:

### Plots (PNG format, 300 DPI)
- `01_simulated_observation.png` - Clean vs noisy lens images
- `02_initial_guess.png` - Initial model vs data comparison
- `03_optimization_history.png` - Loss function convergence
- `04_best_fit_model.png` - Best-fit model summary
- `05_svi_convergence.png` - SVI ELBO convergence
- `06_prior_distribution.png` - Prior parameter distribution
- `07_covariance_matrix.png` - Fisher matrix covariance
- `08_hmc_mass_matrix.png` - HMC adapted mass matrix
- `09_hmc_diagnostics.png` - HMC energy and momentum
- `10_posterior_comparison.png` - Corner plot comparing all methods

### Data (Pickle format)
- `samples_all.pkl` - Complete dataset containing:
  - All sample arrays (prior, FIM, SVI, HMC)
  - Input and best-fit parameters
  - Fisher matrix and covariance matrix
  - SVI results and HMC diagnostics
  - Runtime information

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   ```bash
   python herculens_pe_analysis.py
   ```

3. **View results:**
   - Check the `output/` directory for all generated plots and data
   - Load the pickle file to access all samples and results programmatically

## Key Features

- **Non-interactive mode**: Uses `matplotlib.use('Agg')` for headless operation
- **Automatic saving**: All plots and data are automatically saved
- **Progress tracking**: Shows progress bars for long-running operations
- **Comprehensive output**: Saves all intermediate results and diagnostics
- **Reproducible**: Fixed random seed for consistent results

## Parameters Analyzed

The analysis estimates 22 parameters:

### Lens Mass (7 parameters)
- `lens_theta_E`: Einstein radius
- `lens_e1`, `lens_e2`: Ellipticity components
- `lens_center_x`, `lens_center_y`: Lens position
- `lens_gamma1`, `lens_gamma2`: External shear components

### Source Light (7 parameters)
- `source_amp`: Amplitude
- `source_R_sersic`: Sersic scale radius
- `source_n`: Sersic index
- `source_e1`, `source_e2`: Ellipticity
- `source_center_x`, `source_center_y`: Position

### Lens Light (7 parameters)
- `light_amp`: Amplitude
- `light_R_sersic`: Sersic scale radius
- `light_n`: Sersic index
- `light_e1`, `light_e2`: Ellipticity
- `light_center_x`, `light_center_y`: Position

### Noise (1 parameter)
- `noise_sigma_bkg`: Background noise level

## Methods Comparison

1. **Fisher Information Matrix**: Fast, analytical approximation
2. **Stochastic Variational Inference**: Efficient, learns correlations
3. **Hamiltonian Monte Carlo**: Most accurate, full posterior sampling

## Runtime

Typical runtime on modern hardware:
- Optimization: ~20-30 seconds
- SVI: ~15-25 minutes
- HMC adaptation: ~3-5 minutes
- HMC sampling: ~5-10 minutes
- **Total**: ~25-40 minutes

## Notes

- The script uses double precision (`jax_enable_x64=True`) for numerical stability
- All random operations use a fixed seed for reproducibility
- The analysis assumes a simple SIE+shear lens model with Sersic light profiles
- Results can be loaded and analyzed further using the saved pickle file
