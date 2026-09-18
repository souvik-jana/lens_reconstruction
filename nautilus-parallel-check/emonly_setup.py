"""Shared EM-only setup: tutorial config, priors, and the fisher_h0 prior box.

Copied from tutorial/tutorial_em_only.py. EM-only uses the methods "fisher" and
"deriv-approx" (no "-source" suffix -- there is no lens equation to solve, so
there is no source-plane variant), and fixes lens1_ra_0 / lens1_dec_0 to truth.
"""

import numpy as np
import numpyro.distributions as dist

from gwemfish import (
    make_default_cfg,
    prune_gw_images,
    setup_em_observation,
    setup_gw_observation,
)

SIGMA_SPAN = 3.5


def emonly_cfg(out_dir, nautilus=None):
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["gw"]["n_images"] = 2
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["source_plane"]["n_images"] = 2
    cfg["gw"]["source_pos"] = (0.2, 0.01)
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["nautilus"] = nautilus or {"n_live": 2000, "n_eff": 5000,
                                   "n_like_max": 500_000, "resume": False,
                                   "prior_check": True, "verbose": True}
    cfg["output"]["output_dir"] = out_dir
    return cfg


def emonly_priors(truth):
    """The tutorial's PRIORS block: lens1 shear centre pinned to truth."""
    return {
        "lens1_ra_0": float(truth["lens1_ra_0"]),
        "lens1_dec_0": float(truth["lens1_dec_0"]),
    }


def build_emonly_ctx(out_dir, nautilus=None):
    cfg = emonly_cfg(out_dir, nautilus)
    ctx = setup_em_observation(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    ctx = prune_gw_images(ctx, n_keep=2)
    ctx["cfg"]["priors"] = dict(emonly_priors(ctx["truth_params"]))
    return ctx


def fisher_sigmas(ctx):
    """Per-parameter 1-sigma widths from the Fisher matrix.

    Returns (keys, u0, sigmas). sigmas can contain NaN or non-positive entries
    when the Fisher matrix is ill-conditioned -- with 23 free EM parameters that
    is a real possibility, which is why this is checked before it is used to
    build a prior box.
    """
    keys = list(ctx["likelihood"]["keys_to_include"])
    u0 = np.asarray(ctx["likelihood"]["u0"])
    H0 = np.asarray(ctx["fisher"]["H0"])
    cov = np.linalg.pinv(-H0)
    return keys, u0, np.sqrt(np.diag(cov))


def fisher_h0_priors(ctx, priors, span=SIGMA_SPAN):
    """Uniform box of +-span sigma around the Fisher peak, skipping any parameter
    whose sigma is not a usable positive number."""
    keys, u0, sigmas = fisher_sigmas(ctx)
    out = dict(priors)
    skipped = []
    for i, key in enumerate(keys):
        sig = float(sigmas[i])
        if not np.isfinite(sig) or sig <= 0:
            skipped.append((key, sig))
            continue
        lo, hi = float(u0[i]) - span * sig, float(u0[i]) + span * sig
        out[key] = dist.Uniform(lo, hi)
    return out, skipped
