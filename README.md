# Herculens Project - Gravitational Lensing Parameter Estimation

This repository contains code for gravitational lensing parameter estimation using HMC (Hamiltonian Monte Carlo) methods, combining electromagnetic (EM) and gravitational wave (GW) observations.

## Directory Structure

```
herculens_project/
├── notebooks/
│   ├── EM_GW_PE_MCMC_HMC.ipynb         # Main analysis notebook
│   └── EM_GW_PE_MCMC_HMC_copy.ipynb    # Backup copy
├── scripts/
│   ├── lensimage_gw.py                  # GW lensing helper functions
│   ├── jaxcosmo.py                      # JAX-based cosmology functions
│   ├── fisher.py                        # Fisher matrix analysis
│   ├── corner_plot_utils.py              # Corner plot utilities (main)
│   ├── corner_plot_utils_example.py      # Usage examples for corner plots
│   └── corner_plot.py                   # Legacy corner plot utilities
├── data/
│   ├── samples_PE_EM.pkl                # Posterior samples (EM only)
│   ├── samples_PE_EM_GW.pkl            # Posterior samples (EM + GW)
│   ├── samples_fisher_EM.pkl            # Fisher matrix samples (EM)
│   ├── samples_fisher_EM_GW.pkl        # Fisher matrix samples (EM + GW)
│   ├── truths_PE_EM.pkl                 # True parameter values
│   ├── truths_PE_EM.csv                 # True parameter values (CSV)
│   ├── truths_PE_EM_GW.pkl             # True parameter values (EM + GW)
│   └── truths_PE_EM_GW.csv             # True parameter values (EM + GW, CSV)
└── plots/
    └── [generated corner plots]          # Output plots saved here
```

## Main Notebook

The main notebook `EM_GW_PE_MCMC_HMC.ipynb` performs:
- Gravitational lensing parameter estimation using HMC
- Combined electromagnetic (EM) and gravitational wave (GW) analysis
- Fisher matrix computation and comparison
- Posterior distribution visualization using utility functions

## Dependencies

The notebook requires:
- `herculens` - Gravitational lensing library
- `jaxtronomy` - JAX-based astronomy tools
- `numpyro` - Probabilistic programming framework
- `jax` - JAX numerical computing library
- `astropy` - Astronomy library
- `corner` - Corner plot visualization
- `matplotlib` - Plotting library

## Usage

1. Ensure all dependencies are installed
2. Open `notebooks/EM_GW_PE_MCMC_HMC.ipynb` in Jupyter
3. The notebook automatically adds the `scripts/` directory to the Python path
4. Data files are loaded from the `data/` directory
5. Generated plots are automatically saved to the `plots/` directory

## Scripts

- **lensimage_gw.py**: Provides `LensImageGW` class for computing GW lensing observables
- **jaxcosmo.py**: Provides `JAXCosmology` class for cosmological distance calculations
- **fisher.py**: Provides `FisherMatrix` class for Fisher information matrix analysis
- **corner_plot_utils.py**: Comprehensive utilities for creating corner plots with:
  - Multiple dataset comparisons (e.g., HMC vs Fisher)
  - Custom legends with colored patches
  - Parameter range settings
  - Grouped parameter plots
  - Truth value overlays
  - Automatic plot saving
  - See `corner_plot_utils_example.py` for usage examples
- **corner_plot.py**: Legacy corner plot utilities (deprecated, use `corner_plot_utils.py`)

## Data Files

All data files (samples and truths) are stored in the `data/` directory. The notebook loads these files for analysis and visualization.

## Output

Generated corner plots are automatically saved to the `plots/` directory with descriptive filenames based on parameter groups (e.g., `corner_PE_EM_GW_lens_mass.pdf`, `corner_fisher_EM_GW_source_light.pdf`).

