"""Which fisher_h0 prior is doing the damage in the tutorial GW-only config?

nautilus reached n_eff 69 after 181,700 calls there. The priors are built from the
Fisher covariance at +/- SPAN sigma, so a parameter whose sigma is large compared
to its truth gets a prior box the sampler has to search almost blind.
"""

import numpy as np
from common import save_json
from t9_tutorial_gw_only_timed import tutorial_cfg

import numpyro.distributions as dist
from gwemfish import run_inference, setup_em_observation, setup_gw_observation
from gwemfish.fisher import invert_fisher_matrix

SPANS = (3.5, 2.0, 1.0)

ctx = setup_em_observation(cfg=tutorial_cfg("outputs/t11", "outputs/t11/ckpt.hdf5"))
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
truth = ctx["truth_params"]

ctx["cfg"]["priors"] = {
    "lens1_gamma1": float(truth["lens1_gamma1"]),
    "lens1_gamma2": float(truth["lens1_gamma2"]),
    "lens1_ra_0": float(truth["lens1_ra_0"]),
    "lens1_dec_0": float(truth["lens1_dec_0"]),
    "lens0_theta_E": float(truth["lens0_theta_E"]),
    "lens0_center_x": float(truth["lens0_center_x"]),
    "lens0_center_y": float(truth["lens0_center_y"]),
    "lens0_e2": float(truth["lens0_e2"]),
    "lens0_phi": float(truth["lens0_phi"]),
    "lens0_gamma": float(truth["lens0_gamma"]),
    "y0gw": dist.Uniform(-0.6, 0.6),
    "y1gw": dist.Uniform(-0.6, 0.6),
}
ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (-0.6, 0.6), "y1gw": (-0.6, 0.6)}

run_inference(ctx, mode="GW-only", method="fisher-source",
              cfg={"priors": ctx["cfg"]["priors"],
                   "output": {"output_dir": "outputs/t11"}})

keys = list(ctx["likelihood"]["keys_to_include"])
u0 = np.asarray(ctx["likelihood"]["u0"])
H0 = np.asarray(ctx["fisher"]["H0"])
sigmas = np.sqrt(np.diag(np.asarray(invert_fisher_matrix(-H0))))

print(f"\n{'param':<16} {'truth':>14} {'sigma':>12} {'sigma/|truth|':>13}   "
      + "  ".join(f"{'width@' + str(s):>22}" for s in SPANS))
rows = {}
for i, key in enumerate(keys):
    mu, sig = float(u0[i]), float(sigmas[i])
    rel = abs(sig / mu) if mu else float("inf")
    boxes = {s: (mu - s * sig, mu + s * sig) for s in SPANS}
    rows[key] = {"truth": mu, "sigma": sig, "rel": rel,
                 "boxes": {str(s): boxes[s] for s in SPANS}}
    print(f"{key:<16} {mu:>14.6g} {sig:>12.4g} {rel:>13.3g}   "
          + "  ".join(f"[{boxes[s][0]:>9.4g},{boxes[s][1]:>9.4g}]" for s in SPANS))

print("\nParameters whose 3.5-sigma box spans more than its own truth value:")
for key, r in rows.items():
    if r["rel"] * 3.5 > 1.0:
        print(f"  {key}: box is {r['rel'] * 7:.1f}x the truth value wide")

save_json("t11_prior_widths.json", rows)
