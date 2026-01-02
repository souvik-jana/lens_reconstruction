# Herculens Project - Parametric Lens Reconstruction for Strongly Lensed EM+GW Systems

This repository contains code for parametric lens reconstruction of strongly lensed electromagnetic (EM) and gravitational wave (GW) systems, including Bayesian parameter estimation using Hamiltonian Monte Carlo (HMC), likelihood derivative approximations, and Fisher matrix analysis.

## Main Package: GWEMFISH

The **GWEMFISH** package (`gwemfish/`) contains the finalized scripts for parametric lens reconstruction of strongly lensed electromagnetic (EM) and gravitational wave (GW) systems.

GWEMFISH provides:
- **Bayesian Parameter Estimation**: Posterior inference using Hamiltonian Monte Carlo (HMC) with NUTS sampler
- **Likelihood Derivative Approximations**: Computation of gradients and Hessians (and higher order matrices if needed) for approximate posterior estimation
- **Fisher Matrix Analysis**: Approximate posterior estimation using Fisher information matrix

See [`gwemfish/README.md`](gwemfish/README.md) for package documentation and usage.

## Directory Structure

```
herculens_project/
├── gwemfish/                    # Main package (GWEMFISH)
│   ├── README.md               # Package documentation
│   ├── __init__.py
│   ├── config.py
│   ├── data_sim.py
│   ├── fisher.py
│   ├── inference.py
│   ├── jax_config.py
│   ├── jaxcosmo.py
│   ├── lens_setup.py
│   ├── lensimage_gw.py
│   ├── prob_model.py
│   └── corner_plot_utils.py
│
├── examples/                    # User-facing examples
│   ├── notebooks/              # Example notebooks
│   │   └── example_notebook.ipynb
│   └── scripts/                # Example scripts
│       └── example_usage.py
│
├── notebooks/                   # Development notebooks (work-in-progress)
│   ├── README.md               # See this for development notebook info
│   ├── EM_GW_PE_MCMC_HMC.ipynb
│   └── EM_GW_PE_MCMC_HMC_copy.ipynb
│
├── scripts/                     # Standalone utilities
│   ├── lensimage_gw.py
│   ├── jaxcosmo.py
│   ├── fisher.py
│   └── corner_plot_utils.py
│
├── data/                        # Output data (samples, truths)
│   ├── samples_PE_EM.pkl
│   ├── samples_PE_EM_GW.pkl
│   ├── samples_fisher_EM_GW.pkl
│   └── truths_PE_EM_GW.pkl
│
└── plots/                       # Generated plots
    └── [corner plots and figures]
```

## Quick Start

### Using the Package

```python
from gwemfish import setup_jax, setup_lens, simulate_em, simulate_gw, ProbModel, run_mcmc

# Setup and run inference
setup_jax(ncpus=8, enable_x64=True, platform='cpu')
# ... see examples/ for complete usage
```

### Examples

- **Jupyter Notebook**: See `examples/notebooks/example_notebook.ipynb`
- **Python Script**: See `examples/scripts/example_usage.py`

### Development Notebooks

Development notebooks in `notebooks/` are work-in-progress and not intended for distribution. See [`notebooks/README.md`](notebooks/README.md) for details.

## Dependencies

- `jax` / `jaxlib` - Numerical computing
- `numpyro` - Probabilistic programming framework
- `herculens` - Gravitational lensing library
- `jaxtronomy` - JAX-based astronomy tools
- `matplotlib` - Plotting library
- `corner` - Corner plot visualization
- `numpy`, `scipy` - Scientific computing

## Installation

```bash
# Install core dependencies
pip install jax jaxlib numpyro herculens matplotlib numpy scipy corner

# Install gwemfish package (development mode)
cd /path/to/herculens_project
pip install -e .
```

## Features

- **Parametric Lens Reconstruction**: Joint modeling of lens mass, source light, and lens light profiles
- **Joint EM+GW Parameter Estimation**: Bayesian inference combining lensed galaxy images and gravitational wave time delays
- **Hamiltonian Monte Carlo**: NUTS (No-U-Turn Sampler) for efficient posterior sampling
- **Likelihood Derivatives**: Automatic computation of gradients and Hessians for optimization
- **Fisher Matrix Analysis**: Fast approximate posterior estimation using Fisher information matrix
- **Flexible Lens Models**: Support for various lens mass models (EPL, SHEAR, etc.)
- **Source Plane Inference**: Option to sample source positions and solve for images

## Documentation

- **Package Documentation**: [`gwemfish/README.md`](gwemfish/README.md)
- **Development Notebooks**: [`notebooks/README.md`](notebooks/README.md)
- **Examples**: See `examples/` directory

## Output

- **Data**: Saved samples and truths are stored in `data/`
- **Plots**: Generated corner plots and figures are saved to `plots/`
