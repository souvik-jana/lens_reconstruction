# Herculens Project - Clean Repository

This repository contains the cleaned-up code for gravitational lensing parameter estimation using HMC (Hamiltonian Monte Carlo) methods.

## Directory Structure

```
herculens_project/
├── notebooks/
│   └── EM_GW_PE_MCMC_HMC_copy3.ipynb    # Main analysis notebook
├── scripts/
│   ├── lensimage_gw.py                  # GW lensing helper functions
│   ├── jaxcosmo.py                      # JAX-based cosmology functions
│   ├── fisher.py                        # Fisher matrix analysis
│   └── corner_plot.py                   # Corner plot utilities
└── data/
    ├── samples_PE_EM.pkl                # Posterior samples (EM only)
    ├── samples_PE_EM_GW.pkl            # Posterior samples (EM + GW)
    ├── samples_fisher_EM.pkl            # Fisher matrix samples (EM)
    ├── samples_fisher_EM_GW.pkl        # Fisher matrix samples (EM + GW)
    ├── truths_PE_EM.pkl                 # True parameter values
    ├── truths_PE_EM.csv                 # True parameter values (CSV)
    ├── truths_PE_EM_GW.pkl              # True parameter values (EM + GW)
    └── truths_PE_EM_GW.csv              # True parameter values (EM + GW, CSV)
```

## Main Notebook

The main notebook `EM_GW_PE_MCMC_HMC_copy3.ipynb` performs:
- Gravitational lensing parameter estimation using HMC
- Combined electromagnetic (EM) and gravitational wave (GW) analysis
- Fisher matrix computation and comparison
- Posterior distribution visualization

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
2. Open `notebooks/EM_GW_PE_MCMC_HMC_copy3.ipynb` in Jupyter
3. The notebook automatically adds the `scripts/` directory to the Python path
4. Data files are loaded from the `data/` directory

## Scripts

- **lensimage_gw.py**: Provides `LensImageGW` class for computing GW lensing observables
- **jaxcosmo.py**: Provides `JAXCosmology` class for cosmological distance calculations
- **fisher.py**: Provides `FisherMatrix` class for Fisher information matrix analysis
- **corner_plot.py**: Utilities for creating corner plots (currently imported but not actively used)

## Data Files

All data files (samples and truths) are stored in the `data/` directory. The notebook loads these files for analysis and visualization.

