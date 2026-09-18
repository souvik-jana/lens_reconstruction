"""Prove the prototype cannot disturb fisher / deriv-approx / hmc.

The strong argument is structural: no file under ``src/gwemfish`` is edited, and
the prototype patches nothing at runtime -- it defines its own compiled copies and
leaves ``_gw_loglike_from_images``, ``_solve_images``, ``solve_and_select`` and the
whole of ``fisher.py`` / ``differentiable_solver.py`` untouched, so the autodiff
and the differentiable lens-equation solver behave exactly as before.

This script is the runtime confirmation of that argument: it runs the two
gradient-based source-plane methods, then imports and exercises the prototype,
then runs them again in the same process, and compares. Any monkeypatch, global
jax config change or mutated shared object would show up as a difference.

    python check_isolation.py
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from gwemfish import run_inference

from tutorial_cfg import build_ctx, cfg_for_run

MODE = "GW-only"
METHODS = ["fisher-source", "deriv-approx-source"]


def summarize(samples):
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in sorted(samples.items())}


def run_methods(ctx, cfg, tag):
    out = {}
    for method in METHODS:
        samples, _ = run_inference(
            ctx, mode=MODE, method=method,
            cfg={**cfg, "output": {"output_dir": os.path.join("outputs", "proto"),
                                   "json_tag": f"isolation_{tag}_{method}"}},
        )
        out[method] = summarize(samples)
    return out


ctx = build_ctx(MODE)
cfg = cfg_for_run(ctx, MODE)

print("=== pass 1: gradient methods, prototype not yet imported ===")
before = run_methods(ctx, cfg, "before")

print("\n=== exercising the prototype in the same process ===")
from proto.jit_likelihood import build_problem

prior, loglike, _ = build_problem(ctx, cfg, MODE, jit=True)
rng = np.random.default_rng(0)
for u in rng.uniform(size=(20, prior.dimensionality())):
    loglike(prior.unit_to_dictionary(u))
print("  prototype evaluated 20 points")

print("\n=== pass 2: same gradient methods, same process ===")
after = run_methods(ctx, cfg, "after")

print(f"\n{'=' * 60}\nISOLATION\n{'=' * 60}")
ok_all = True
for method in METHODS:
    worst = 0.0
    for key, (mu_b, sd_b) in before[method].items():
        mu_a, sd_a = after[method][key]
        worst = max(worst, abs(mu_a - mu_b), abs(sd_a - sd_b))
    ok = worst == 0.0
    ok_all &= ok
    print(f"  {method:<22} max |difference| = {worst:.3e}   "
          f"{'IDENTICAL' if ok else 'CHANGED'}")
print(f"\noverall: {'PASS' if ok_all else 'FAIL'}")
