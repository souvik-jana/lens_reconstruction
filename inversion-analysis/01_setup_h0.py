"""
Copy of tutorial/tutorial_em_gw.py setup, fisher-source only, for inversion analysis.

For each (parametrization, sigma_dL_eff): simulate EM+GW, run fisher-source, save
FM = -H0, u0, keys and the pipeline's own fisher-source samples to
outputs/fisher_h0_{parametrization}_sigma{sigma}.npz.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib
import numpy as np
import numpyro.distributions as dist

matplotlib.use("Agg")

from gwemfish import make_default_cfg, run_inference, setup_em_observation, setup_gw_observation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

SIGMAS = [0.05, 0.1, 0.5]
PARAMETRIZATIONS = ["e1e2", "q_phi"]
RUNS = [(p, s) for p in PARAMETRIZATIONS for s in SIGMAS]

for parametrization, sigma_dl in RUNS:
    tag = f"{parametrization}_sigma{sigma_dl}"
    run_dir = os.path.join(OUT, f"run_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    CFG = make_default_cfg()
    CFG["use_parameter_layout"] = True
    CFG["lens_mass_parametrization"] = parametrization
    CFG["gw"]["n_images"] = 4
    CFG["gw"]["source_box_half_width"] = 0.8
    CFG["source_plane"]["n_images"] = 4
    CFG["gw"]["source_pos"] = (0.02, 0.00001)
    CFG["gw"]["error_scales"]["sigma_td"] = 0.001
    CFG["gw"]["error_scales"]["sigma_dL_eff"] = sigma_dl
    CFG["output"]["output_dir"] = run_dir

    ctx = setup_em_observation(cfg=CFG)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    truth_params = ctx["truth_params"]

    PRIORS = {
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        "y0gw": dist.Uniform(-0.8, 0.8),
        "y1gw": dist.Uniform(-0.8, 0.8),
    }
    ctx["cfg"]["priors"] = dict(PRIORS)

    print(f"\n--- EM+GW fisher-source, {tag} ---\n")
    samples, truths = run_inference(
        ctx,
        mode="EM+GW",
        method="fisher-source",
        cfg={"priors": PRIORS, "output": {"output_dir": run_dir, "json_tag": "fisher_source"}},
    )

    keys = list(ctx["likelihood"]["keys_to_include"])
    np.savez(
        os.path.join(OUT, f"fisher_h0_{tag}.npz"),
        FM=-np.asarray(ctx["fisher"]["H0"]),
        u0=np.asarray(ctx["likelihood"]["u0"]),
        keys=np.array(keys),
        pipeline_samples=np.stack([np.asarray(samples[k]) for k in keys], axis=1),
    )
    print(f"saved fisher_h0_{tag}.npz  ({len(keys)} params)")
