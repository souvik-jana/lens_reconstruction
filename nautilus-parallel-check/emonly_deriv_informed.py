"""EM-only deriv-approx with informed=True, against the plain-NUTS run.

The first EM-only deriv-approx diverged (r_hat ~1e15, sigmas inflated up to 1e6x)
because cfg["inference"]["informed"] defaults to None, which the dispatch at
simple_pipeline.py:2229 reads as False -- so it ran plain NUTS with an identity
mass matrix over parameters spanning three orders of magnitude (light0_amp ~ 8,
noise_sigma_bkg ~ 0.01).

informed=True uses the Fisher Hessian as the mass matrix, which is the
preconditioning that scale disparity needs. This reruns it and compares both
against Fisher, which is the reference for what the widths should be.

    python emonly_deriv_informed.py
"""

import json
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from gwemfish import run_inference

from emonly_setup import build_emonly_ctx, emonly_priors, fisher_sigmas

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_only")
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    ctx = build_emonly_ctx(OUT_DIR)
    PRIORS = emonly_priors(ctx["truth_params"])

    print("\n--- EM-only: deriv-approx with informed=True ---", flush=True)
    samples, _ = run_inference(
        ctx, mode="EM-only", method="deriv-approx",
        cfg={"priors": PRIORS,
             "inference": {"informed": True},
             "output": {"output_dir": OUT_DIR,
                        "json_tag": "deriv_approx_informed"}},
    )
    samples = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(os.path.join(OUT_DIR, "EM-only_deriv-approx-informed.npz"), **samples)

    keys, _, sigmas = fisher_sigmas(ctx)
    sigma_ref = dict(zip(keys, sigmas))

    plain_path = os.path.join(OUT_DIR, "EM-only_deriv-approx.npz")
    plain = dict(np.load(plain_path)) if os.path.exists(plain_path) else {}

    print(f"\n{'=' * 92}\nSIGMA vs FISHER  (ratio; 1.0 = agrees)\n{'=' * 92}")
    print(f"  {'param':<22} {'fisher':>13} {'informed':>13} {'ratio':>9}"
          f" {'plain':>13} {'ratio':>11}")
    worst_inf = worst_plain = 0.0
    for p in sorted(samples):
        ref = sigma_ref.get(p)
        if not ref or not np.isfinite(ref) or ref <= 0:
            continue
        s_inf = float(np.std(samples[p]))
        r_inf = s_inf / ref
        worst_inf = max(worst_inf, r_inf, 1 / r_inf if r_inf > 0 else 0)
        row = f"  {p:<22} {ref:>13.6g} {s_inf:>13.6g} {r_inf:>9.3f}"
        if p in plain:
            s_pl = float(np.std(plain[p]))
            r_pl = s_pl / ref
            worst_plain = max(worst_plain, r_pl, 1 / r_pl if r_pl > 0 else 0)
            row += f" {s_pl:>13.6g} {r_pl:>11.4g}"
        print(row)

    print(f"\n  worst deviation from Fisher -- informed: {worst_inf:.3g}x"
          f"   plain: {worst_plain:.3g}x")
    print(f"\n  {'informed=True FIXES the divergence.' if worst_inf < 10 else 'informed=True did NOT fix it -- investigate further.'}")

    json.dump({"worst_informed": worst_inf, "worst_plain": worst_plain},
              open(os.path.join(OUT_DIR, "deriv_informed_check.json"), "w"), indent=2)
