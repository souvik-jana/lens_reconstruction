"""Cheap smoke test for the pool path: does pool=N start, build and finish?

Tiny budget on purpose -- this answers "does it work at all", not "is it fast".
Prints each worker's start method and build time, which is what tells you the
workers really are separate spawned processes and not a silent serial fallback.
"""

import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from proto.run_proto import run
from tutorial_cfg import build_ctx, cfg_for_run

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "GW-only"
    ctx = build_ctx(mode)
    cfg = cfg_for_run(ctx, mode)

    for label, opts in [("serial jit", dict(jit=True, pool=None)),
                        ("pool=4 jit", dict(jit=True, pool=4))]:
        print(f"\n--- {mode} / {label} ---", flush=True)
        t0 = time.perf_counter()
        samples, diag = run(ctx, cfg, mode, n_live=100, n_eff=100, n_like_max=800, **opts)
        print(f"  {label}: {time.perf_counter() - t0:.1f}s  n_like={diag['n_like']}  "
              f"log_z={diag['log_z']:.4f}  samples={diag['n_posterior_samples']}")
