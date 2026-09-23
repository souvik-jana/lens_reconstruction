"""Shared setup for the nautilus-fix analysis tests.

Reads (never edits) the committed prototype in ``nautilus-parallel-check/`` and
``src/gwemfish``. Everything written by these scripts lands in
``nautilus-fix-analysis/outputs/``.
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROTO_DIR = os.path.join(REPO, "nautilus-parallel-check")
OUT = os.path.join(HERE, "outputs")

sys.path.insert(0, PROTO_DIR)

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax._src.interpreters.pxla as pxla

COMPILES = [0]
_ORIGINAL_FROM_HLO = pxla.UnloadedMeshExecutable.from_hlo


def counting_from_hlo(*args, **kwargs):
    COMPILES[0] += 1
    return _ORIGINAL_FROM_HLO(*args, **kwargs)


pxla.UnloadedMeshExecutable.from_hlo = counting_from_hlo


def compiles_per_call(loglike, points, n=10):
    for p in points[:3]:
        loglike(p)
    COMPILES[0] = 0
    for p in points[:n]:
        loglike(p)
    return COMPILES[0] / min(n, len(points))


def ms_per_call(loglike, points):
    import time

    for p in points[:3]:
        loglike(p)
    t0 = time.perf_counter()
    for p in points:
        loglike(p)
    return (time.perf_counter() - t0) / len(points) * 1000


def save_json(name, obj):
    import json

    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"\nwrote {path}")
