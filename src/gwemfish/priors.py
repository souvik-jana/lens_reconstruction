"""
Default prior registries and default image-position geometry.

This module exists to keep `prob_model.py` focused on likelihood/model logic,
while all default priors and geometry live in one place.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .ellipticity_reparam import default_qphi_priors


def _make_default_priors_em_gw(pix_scl=0.4):
    """Return the default prior registry for EM+GW (joint) models."""
    return {
        # GW
        "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
        "dL": lambda: numpyro.sample("dL", dist.Uniform(10.0, 50000.0)),
        # Source light (SersicElliptic)
        "source_amp": lambda: numpyro.sample("source_amp", dist.LogUniform(1.0e-6, 1.0e6)),
        "source_R_sersic": lambda: numpyro.sample("source_R_sersic", dist.Uniform(0.0, 30.0)),
        "source_n": lambda: numpyro.sample("source_n", dist.Uniform(0.8, 5.0)),
        "source_e1": lambda: numpyro.sample("source_e1", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        "source_e2": lambda: numpyro.sample("source_e2", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        # Source center — fixed by default, sample by overriding in priors=
        "source_center_x": lambda: numpyro.sample("source_center_x", dist.Uniform(low=-20.0, high=20.0)),
        "source_center_y": lambda: numpyro.sample("source_center_y", dist.Uniform(low=-20.0, high=20.0)),
        # Lens light (SersicElliptic) priors
        "light_center_x": lambda: numpyro.sample("light_center_x", dist.Normal(0.0, 0.3)),
        "light_center_y": lambda: numpyro.sample("light_center_y", dist.Normal(0.0, 0.3)),
        "light_R_sersic": lambda: numpyro.sample("light_R_sersic", dist.Uniform(0.0, 30.0)),
        "light_e1": lambda: numpyro.sample("light_e1", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        "light_e2": lambda: numpyro.sample("light_e2", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        "light_amp": lambda: numpyro.sample("light_amp", dist.LogUniform(1.0e-6, 1.0e6)),
        "light_n": lambda: numpyro.sample("light_n", dist.Uniform(0.8, 5.0)),
        # Lens mass
        "lens_theta_E": lambda: numpyro.sample("lens_theta_E", dist.Uniform(0.0, 8.0)),
        "lens_e1": lambda: numpyro.sample("lens_e1", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        "lens_e2": lambda: numpyro.sample("lens_e2", dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0)),
        # Alternative (q, phi) parametrization -- only used when
        # lens_mass_parametrization="q_phi"; present unconditionally so Fisher /
        # deriv-approx (which look these keys up generically) always find them.
        **default_qphi_priors("lens"),
        "lens_gamma": lambda: numpyro.sample("lens_gamma", dist.Uniform(1.5, 3.0)),
        # External shear components (gamma1/gamma2)
        "lens_gamma1": lambda: numpyro.sample("lens_gamma1", dist.Uniform(-0.3, 0.3)),
        "lens_gamma2": lambda: numpyro.sample("lens_gamma2", dist.Uniform(-0.3, 0.3)),
        # Lens center — sample by overriding in priors= (default is Normal(0,0.1))
        "lens_center_x": lambda: numpyro.sample("lens_center_x", dist.Normal(0.0, 0.1)),
        "lens_center_y": lambda: numpyro.sample("lens_center_y", dist.Normal(0.0, 0.1)),
        # Noise
        "noise_sigma_bkg": lambda: numpyro.sample("noise_sigma_bkg", dist.LogUniform(1.0e-6, 1.0e6)),
    }


DEFAULT_PRIORS_EM_GW = _make_default_priors_em_gw()

# EM-only defaults: same as EM+GW priors, but with GW/cosmology parameters removed.
DEFAULT_PRIORS_EM_ONLY = {k: v for k, v in DEFAULT_PRIORS_EM_GW.items() if k not in ("T_star", "dL")}

DEFAULT_PRIORS_GW_ONLY = {
    "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
    "dL": lambda: numpyro.sample("dL", dist.Uniform(0.00001, 50000.0)),
    "lens_theta_E": lambda: numpyro.sample("lens_theta_E", dist.Uniform(0.1, 10.0)),
    "lens_e1": lambda: numpyro.sample("lens_e1", dist.TruncatedNormal(0.0, 0.6, low=-1.0, high=1.0)),
    "lens_e2": lambda: numpyro.sample("lens_e2", dist.TruncatedNormal(0.0, 0.6, low=-1.0, high=1.0)),
    # Alternative (q, phi) parametrization -- see note in _make_default_priors_em_gw.
    **default_qphi_priors("lens"),
    "lens_gamma": lambda: numpyro.sample("lens_gamma", dist.Uniform(1.1, 3.0)),
    # External shear components (gamma1/gamma2)
    "lens_gamma1": lambda: numpyro.sample("lens_gamma1", dist.Uniform(-0.3, 0.3)),
    "lens_gamma2": lambda: numpyro.sample("lens_gamma2", dist.Uniform(-0.3, 0.3)),
    # Lens center — default is Normal(0,0.1)
    "lens_center_x": lambda: numpyro.sample("lens_center_x", dist.Normal(0.0, 0.1)),
    "lens_center_y": lambda: numpyro.sample("lens_center_y", dist.Normal(0.0, 0.1)),
}

DEFAULT_IMAGE_POSITION_PRIORS_EM = {
    # Default fallback: sample all image positions from a global box.
    # If per-image geometry is provided (x_image_true/delx and y_image_true/dely),
    # the model will instead sample per-image centered boxes.
    "minx": -20.0,
    "maxx": 20.0,
    "miny": -20.0,
    "maxy": 20.0,
}

DEFAULT_IMAGE_POSITION_PRIORS_GW = {
    "minx": -20.0,
    "maxx": 20.0,
    "miny": -20.0,
    "maxy": 20.0,
}

# Bounds for Nautilus source-plane sampling.
# Values are (lo, hi) tuples used to build scipy distributions in nautilus source/image builders.
# Override via cfg['gw']['source_plane_bounds'].
DEFAULT_PRIORS_GW_SOURCE_PLANE = {
    "lens_theta_E":  (0.1,  10.0),
    "lens_e1":       (-1.0,  1.0),   # TruncatedNormal in practice; these are hard bounds
    "lens_e2":       (-1.0,  1.0),
    "lens_gamma":    (1.1,   3.0),
    "lens_gamma1":   (-0.3,  0.3),
    "lens_gamma2":   (-0.3,  0.3),
    "lens_center_x": (-1.0,  1.0),   # Normal(0,0.1) clipped; these are hard bounds
    "lens_center_y": (-1.0,  1.0),
    "T_star":        (1e1,   1e12),
    "dL":            (10.0,  50000.0),
    "y0gw":          (-1.0,  1.0),   # uniform source-plane position (arcsec)
    "y1gw":          (-1.0,  1.0),
}

