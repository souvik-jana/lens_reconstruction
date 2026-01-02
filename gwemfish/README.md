# GWEMFISH

**Gravitational Wave + Electromagnetic Fisher Information**

Parametric lens reconstruction pipeline for strongly lensed EM+GW systems.

## Description

GWEMFISH is a Python package for parametric lens reconstruction of strongly lensed electromagnetic (EM) and gravitational wave (GW) systems.

The package provides:
- **Parametric Lens Modeling**: Joint reconstruction of lens mass, source light, and lens light profiles
- **Bayesian Parameter Estimation**: Posterior inference using Hamiltonian Monte Carlo (HMC) with NUTS sampler
- **Likelihood Derivative Approximations**: Computation of gradients and Hessians (and higher order matrices if needed) for approximate posterior estimation
- **Fisher Matrix Analysis**: Approximate posterior estimation using Fisher information matrix

GWEMFISH uses:
- **Electromagnetic (EM) observations**: Lensed galaxy images
- **Gravitational Wave (GW) observations**: Time delays and observed luminosity distance
- **Joint Inference**: Parameter estimation from both EM and GW data

## Installation

```bash
# Install dependencies
pip install jax jaxlib numpyro herculens matplotlib numpy scipy

# Install gwemfish (if using as package)
cd /path/to/herculens_project
pip install -e .
```

## Quick Start

```python
from gwemfish import setup_jax, setup_lens, simulate_em, simulate_gw, ProbModel, run_mcmc

# Setup JAX first
setup_jax(ncpus=8, enable_x64=True, platform='cpu')

# Setup lens and simulate data
kwargs_lens, x_image_true, y_image_true, lens_mass_model = setup_lens(...)
em_obs, lens_image = simulate_em(...)
x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(...)

# Run inference
model = ProbModel(n_images=4, gw_observations=gw_obs, em_observations=em_obs, ...)
samples, summary, extra_fields, mcmc = run_mcmc(model.model, num_warmup=6000, num_samples=12000)
```

## Package Structure

- `config.py` - Configuration constants and default parameters
- `data_sim.py` - Data simulation functions (EM and GW)
- `fisher.py` - Fisher matrix computation
- `inference.py` - MCMC inference functions
- `jax_config.py` - JAX configuration utilities
- `jaxcosmo.py` - JAX-based cosmology calculations
- `lens_setup.py` - Lens model setup and image position solving
- `lensimage_gw.py` - GW lensing calculations
- `prob_model.py` - Probabilistic models (ProbModel, ProbModelSourcePlane, ProbModelFisher)
- `corner_plot_utils.py` - Plotting utilities for corner plots

## Examples

See `../examples/` for complete usage examples:
- `examples/notebooks/example_notebook.ipynb` - Jupyter notebook example
- `examples/scripts/example_usage.py` - Python script example

## Documentation

For detailed documentation, see the main project README at `../README.md`.

## Dependencies

- `jax` / `jaxlib` - Numerical computing
- `numpyro` - Probabilistic programming
- `herculens` - Gravitational lensing library
- `matplotlib` - Plotting
- `numpy`, `scipy` - Scientific computing

## License

See `../LICENSE` (if applicable)

