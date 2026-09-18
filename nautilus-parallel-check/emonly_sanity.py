"""EM-only sanity gate: run fisher and deriv-approx, then inspect the sigmas.

This runs before any nautilus work on purpose. EM-only has 23 free parameters,
and the fisher_h0 prior box is built from sqrt(diag(inv(Fisher))). If the Fisher
matrix is ill-conditioned, some of those sigmas come back NaN or non-positive,
and a prior box built from them would be silently garbage -- an infinite or empty
interval that nautilus would then spend hours sampling.

So: check first, decide after.

    python emonly_sanity.py
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

from emonly_setup import SIGMA_SPAN, build_emonly_ctx, emonly_priors, fisher_sigmas

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "em_only")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["fisher", "deriv-approx"]

if __name__ == "__main__":
    ctx = build_emonly_ctx(OUT_DIR)
    truth = ctx["truth_params"]
    PRIORS = emonly_priors(truth)
    print(f"EM-only system built. Fixed to truth: {list(PRIORS)}")

    samples_by_method = {}
    for method in METHODS:
        print(f"\n--- EM-only: {method} ---", flush=True)
        # informed=True for deriv-approx: cfg["inference"]["informed"] defaults to
        # None, which simple_pipeline.py:2229 reads as False, giving plain NUTS
        # with an identity mass matrix. EM-only spans three orders of magnitude
        # (light0_amp ~ 8, noise_sigma_bkg ~ 0.01), so plain NUTS stalls -- first
        # run here came back with r_hat ~1e15. The Fisher Hessian mass matrix is
        # the preconditioning that scale disparity needs.
        samples, _ = run_inference(
            ctx, mode="EM-only", method=method,
            cfg={"priors": PRIORS,
                 "inference": {"informed": True} if method == "deriv-approx" else {},
                 "output": {"output_dir": OUT_DIR,
                            "json_tag": method.replace("-", "_")}},
        )
        samples_by_method[method] = {k: np.asarray(v) for k, v in samples.items()}
        np.savez(os.path.join(OUT_DIR, f"EM-only_{method}.npz"),
                 **samples_by_method[method])

    keys, u0, sigmas = fisher_sigmas(ctx)

    print(f"\n{'=' * 86}\nFISHER SIGMAS  ({len(keys)} free parameters)\n{'=' * 86}")
    print(f"  {'param':<22} {'u0':>14} {'sigma':>14} {'span/|u0|':>12}  status")
    bad = []
    for i, k in enumerate(keys):
        sig = float(sigmas[i])
        centre = float(u0[i])
        if not np.isfinite(sig):
            status, rel = "NaN/inf", float("nan")
            bad.append((k, sig))
        elif sig <= 0:
            status, rel = "NON-POSITIVE", float("nan")
            bad.append((k, sig))
        else:
            rel = SIGMA_SPAN * sig / abs(centre) if centre != 0 else float("inf")
            status = "ok" if rel < 1.0 else "WIDE (box exceeds |u0|)"
        print(f"  {k:<22} {centre:>14.6g} {sig:>14.6g} {rel:>12.3g}  {status}")

    print(f"\n{'=' * 86}\nPOSTERIOR SIGMAS FROM EACH METHOD\n{'=' * 86}")
    shared = [p for p in samples_by_method[METHODS[0]]
              if all(p in s for s in samples_by_method.values())]
    print(f"  {'param':<22}" + "".join(f"{m:>18}" for m in METHODS) + "   nan?")
    nan_in_samples = []
    for p in shared:
        cells, flag = [], ""
        for m in METHODS:
            v = samples_by_method[m][p]
            sd = float(np.std(v))
            cells.append(sd)
            if not np.isfinite(sd) or not np.all(np.isfinite(v)):
                flag = "  <-- NaN"
                nan_in_samples.append((m, p))
        print(f"  {p:<22}" + "".join(f"{c:>18.6g}" for c in cells) + flag)

    print(f"\n{'=' * 86}\nVERDICT\n{'=' * 86}")
    ok = True
    if bad:
        ok = False
        print(f"  {len(bad)} Fisher sigma(s) unusable -- a {SIGMA_SPAN}-sigma prior "
              f"box cannot be built for these:")
        for k, s in bad:
            print(f"    {k}: sigma={s}")
    else:
        print(f"  All {len(keys)} Fisher sigmas finite and positive.")

    if nan_in_samples:
        ok = False
        print(f"  NaN in posterior samples: {nan_in_samples}")
    else:
        print("  No NaN in any posterior samples from either method.")

    # A NaN check alone is not enough. A diverged chain returns perfectly finite
    # numbers that are simply wrong -- and a sigma inflated by six orders of
    # magnitude is more dangerous than a NaN because nothing downstream flags it.
    # Fisher's sigmas are the reference: any method disagreeing by more than 10x
    # has not converged, whatever its samples look like.
    sigma_ref = dict(zip(keys, sigmas))
    diverged = {}
    for m in METHODS:
        offenders = []
        for p in shared:
            ref = sigma_ref.get(p)
            if ref and np.isfinite(ref) and ref > 0:
                ratio = float(np.std(samples_by_method[m][p])) / ref
                if ratio > 10 or ratio < 0.1:
                    offenders.append((p, ratio))
        if offenders:
            diverged[m] = offenders

    if diverged:
        ok = False
        for m, offenders in diverged.items():
            worst = max(offenders, key=lambda kv: kv[1])
            print(f"  {m}: NOT CONVERGED -- {len(offenders)}/{len(shared)} "
                  f"parameters disagree with Fisher by >10x "
                  f"(worst: {worst[0]} at {worst[1]:.3g}x)")
    else:
        print("  All posterior widths within 10x of Fisher -- no sign of divergence.")

    json.dump({"keys": keys, "u0": [float(x) for x in u0],
               "sigmas": [float(x) for x in sigmas],
               "bad": [[k, float(s)] for k, s in bad]},
              open(os.path.join(OUT_DIR, "fisher_sigmas.json"), "w"), indent=2)

    # Two separate questions, and they have different answers here: the prior box
    # only needs the Fisher sigmas, so a broken deriv-approx does not block it.
    box_ok = not bad
    print(f"\n  fisher_h0 prior box buildable : "
          f"{'YES' if box_ok else 'NO -- unusable Fisher sigmas'}")
    print(f"  all methods usable for comparison: "
          f"{'YES' if ok else 'NO -- see divergence above'}")
