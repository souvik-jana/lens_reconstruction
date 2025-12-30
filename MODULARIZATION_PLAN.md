# Minimalistic Modularization Plan

## Structure

```
scripts/
├── lens_setup.py          # Mock lens model + image positions
├── data_sim.py            # EM + GW data simulation  
├── prob_model.py          # ProbModel class
├── inference.py           # MCMC inference
├── fisher.py              # Fisher matrix + banana model
└── [existing files...]
```

## Modules

### 1. `lens_setup.py`
```python
def setup_lens(zl, zs, phi, q, gamma, theta_E, source_pos):
    """Returns: kwargs_lens, x_image_true, y_image_true, lens_mass_model"""
```

### 2. `data_sim.py`
```python
def simulate_em(lens_params, source_params, light_params, npix, pix_scl, seed):
    """Returns: em_obs dict"""
    
def simulate_gw(image_pos, lens_params, cosmology, zl, zs):
    """Returns: gw_obs dict, data_GW dict"""
```

### 3. `prob_model.py`
```python
class ProbModel(hcl.NumpyroModel):
    def __init__(self, n_images, gw_obs, em_obs, lens_image, lens_gw, 
                 input_params, x_image_true, y_image_true):
    def model(self):
```

### 4. `inference.py`
```python
def run_mcmc(model, num_warmup=6500, num_samples=14500):
    """Returns: samples, summary, extra, mcmc"""
```

### 5. `fisher.py`
```python
def compute_fisher(model, u0, keys_to_include):
    """Returns: approx_logp function"""
    
def banana_model(keys_to_include, x_image_true, y_image_true, 
                 approx_logp, input_params):
    """Returns: banana model function"""
```

## Notebook Flow

```python
# Setup
from scripts.lens_setup import setup_lens
from scripts.data_sim import simulate_em, simulate_gw
from scripts.prob_model import ProbModel
from scripts.inference import run_mcmc
from scripts.fisher import compute_fisher, banana_model

# 1. Lens
kwargs_lens, x_img, y_img, mass_model = setup_lens(...)

# 2. Data
em_obs = simulate_em(...)
gw_obs, data_GW = simulate_gw(...)

# 3. Model
gw_model = ProbModel(..., gw_obs, em_obs, ...)

# 4. PE
samples, _, _, _ = run_mcmc(gw_model.model)

# 5. Fisher
approx_logp = compute_fisher(gw_model.model, u0, keys_to_include)
banana = banana_model(keys_to_include, x_img, y_img, approx_logp, input_params)

# 6. Fisher PE
samples_approx, _, _, _ = run_mcmc(banana)
```

