"""Does the q/phi parametrization flow through every mode and method?

q_phi has only ever been run GW-only, only on source-plane methods, and always
with phi fixed -- while cfg_reference.py:597 claims it "works uniformly" across
every method. This runs the whole grid on one 4-image system.

The assertions are deliberately split:

  A. flow      -- q/phi reached the sampler and converted correctly. MUST pass.
                  Independent of convergence, which is why the budgets can be tiny.
  B. recovery  -- truth within N sigma. REPORTED, not asserted: nautilus at
                  n_eff=100 has not converged and a wide posterior there is
                  expected, not a failure.

    python qphi_coverage_matrix.py            # all 19 cells
    python qphi_coverage_matrix.py EM-only    # one mode
"""

import os
import sys
import time
import traceback

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from gwemfish import run_inference

from qphi_setup import (
    build_ctx,
    check_four_images,
    priors_for,
    source_plane_bounds,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "qphi_coverage")
os.makedirs(OUT_DIR, exist_ok=True)

PREFIX = "lens0"
TOL_IDENTITY = 1e-12

# Layouts where lens0_phi is a free parameter. This changes what a correct run
# looks like: add_qphi_columns_to_samples backfills e1/e2 only when BOTH q and
# phi are in the samples dict (ellipticity_reparam.py:167-191), so a fixed-phi
# run legitimately produces neither phi nor e1/e2 columns. Asserting the same
# thing for both would fail correct code.
PHI_FREE_LAYOUTS = ("EM-only", "EM+GW")

# (label, mode, layout, method)
CELLS = [
    ("GW-only-A", "GW-only", "GW-only-A", "fisher-source"),
    ("GW-only-A", "GW-only", "GW-only-A", "deriv-approx-source"),
    ("GW-only-A", "GW-only", "GW-only-A", "hmc-informed-source"),
    ("GW-only-A", "GW-only", "GW-only-A", "nautilus-source"),
    ("GW-only-A", "GW-only", "GW-only-A", "nautilus-image"),

    ("GW-only-B", "GW-only", "GW-only-B", "fisher-source"),
    ("GW-only-B", "GW-only", "GW-only-B", "deriv-approx-source"),
    ("GW-only-B", "GW-only", "GW-only-B", "hmc-informed-source"),
    ("GW-only-B", "GW-only", "GW-only-B", "nautilus-source"),
    ("GW-only-B", "GW-only", "GW-only-B", "nautilus-image"),

    ("EM-only", "EM-only", "EM-only", "fisher"),
    ("EM-only", "EM-only", "EM-only", "deriv-approx"),
    ("EM-only", "EM-only", "EM-only", "hmc-informed"),
    ("EM-only", "EM-only", "EM-only", "nautilus-source"),

    ("EM+GW", "EM+GW", "EM+GW", "fisher-source"),
    ("EM+GW", "EM+GW", "EM+GW", "deriv-approx-source"),
    ("EM+GW", "EM+GW", "EM+GW", "hmc-informed-source"),
    ("EM+GW", "EM+GW", "EM+GW", "nautilus-source"),
    ("EM+GW", "EM+GW", "EM+GW", "nautilus-image"),
]

# informed=True for every NUTS-based method. The default
# (cfg["inference"]["informed"]=None) is read as False by simple_pipeline.py:2229
# and gives plain NUTS, which diverged on EM-only at r_hat ~1e15.
NEEDS_INFORMED = ("deriv-approx", "deriv-approx-source")


def qphi_identity_error(samples):
    """max |e1 - (1-q)/(1+q)cos(2phi)| over the posterior.

    This is the strong form of the flow check: if e1/e2 had been sampled as free
    parameters from their own prior -- the failure qphi_derived_keys() exists to
    prevent -- they would not satisfy this identity at all.
    """
    need = [f"{PREFIX}_{s}" for s in ("q", "phi", "e1", "e2")]
    if not all(k in samples for k in need):
        return None
    q = np.asarray(samples[f"{PREFIX}_q"], dtype=float)
    phi = np.asarray(samples[f"{PREFIX}_phi"], dtype=float)
    a = (1.0 - q) / (1.0 + q)
    de1 = np.abs(np.asarray(samples[f"{PREFIX}_e1"], dtype=float) - a * np.cos(2 * phi))
    de2 = np.abs(np.asarray(samples[f"{PREFIX}_e2"], dtype=float) - a * np.sin(2 * phi))
    return float(max(de1.max(), de2.max()))


def recovery_sigma(samples, truths, key):
    if key not in samples or key not in truths:
        return None
    v = np.asarray(samples[key], dtype=float)
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd == 0:
        return None
    return float((np.median(v) - float(truths[key])) / sd)


def run_cell(ctx, label, mode, layout, method):
    priors = priors_for(ctx, layout)
    cfg = {"priors": priors,
           "output": {"output_dir": OUT_DIR,
                      "json_tag": f"{label}_{method}".replace("-", "_")}}
    if method in NEEDS_INFORMED:
        cfg["inference"] = {"informed": True}
    if mode != "EM-only":
        cfg["gw"] = {"source_plane_bounds": source_plane_bounds(ctx)}

    t0 = time.perf_counter()
    samples, truths = run_inference(ctx, mode=mode, method=method, cfg=cfg)
    wall = time.perf_counter() - t0

    samples = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(os.path.join(OUT_DIR, f"{label}_{method}.npz".replace("/", "_")), **samples)

    phi_free = layout in PHI_FREE_LAYOUTS
    has_q = f"{PREFIX}_q" in samples
    has_phi = f"{PREFIX}_phi" in samples
    has_e = all(f"{PREFIX}_{k}" in samples for k in ("e1", "e2"))
    ident = qphi_identity_error(samples)
    q = np.asarray(samples.get(f"{PREFIX}_q", []), dtype=float)
    q_in_range = bool(q.size and (q.min() >= 0.01) and (q.max() <= 1.0))
    e2 = np.asarray(samples.get(f"{PREFIX}_e2", []), dtype=float)

    # PASS means "matches what a correct run produces for THIS layout".
    if phi_free:
        flow = has_q and has_phi and has_e and ident is not None and ident < TOL_IDENTITY
    else:
        flow = has_q and not has_phi and not has_e

    return {
        "label": label, "method": method, "wall": wall, "n": len(q) if q.size else 0,
        "phi_free": phi_free, "flow": flow,
        "has_q": has_q, "has_phi": has_phi, "has_e": has_e,
        "identity": ident,
        "q_in_range": q_in_range,
        "q_min": float(q.min()) if q.size else None,
        "q_max": float(q.max()) if q.size else None,
        "e2_nonzero": bool(e2.size and float(np.std(e2)) > 0),
        "bias_q": recovery_sigma(samples, truths, f"{PREFIX}_q"),
        "bias_phi": recovery_sigma(samples, truths, f"{PREFIX}_phi"),
        "error": None,
    }


if __name__ == "__main__":
    argv = sys.argv[1:]
    # --neff N raises the nautilus budget. At n_eff=100 nautilus' equal-weight
    # thinning (floor(w/w_max)) leaves a single sample, which is enough for the
    # deterministic flow assertions but makes the recovery column meaningless.
    neff = None
    if "--neff" in argv:
        i = argv.index("--neff")
        neff = int(argv[i + 1])
        del argv[i:i + 2]
        from qphi_setup import BUDGETS
        BUDGETS["nautilus"]["n_eff"] = neff
        BUDGETS["nautilus"]["n_live"] = max(200, neff // 4)
        # cap the call budget: this is a plumbing check, not a production run
        BUDGETS["nautilus"]["n_like_max"] = max(20_000, neff * 50)
        print(f"nautilus budget overridden: n_eff={neff}, "
              f"n_live={BUDGETS['nautilus']['n_live']}, "
              f"n_like_max={BUDGETS['nautilus']['n_like_max']}")
    wanted = argv

    print("=" * 78)
    print("STEP 0 -- pre-flight")
    print("=" * 78)
    ctx = build_ctx("q_phi")
    check_four_images(ctx)
    truth = ctx["truth_params"]
    print(f"\n  lens0_q   truth = {float(truth['lens0_q']):.6f}")
    print(f"  lens0_phi truth = {float(truth['lens0_phi']):+.6f} rad "
          f"({np.degrees(float(truth['lens0_phi'])):+.2f} deg)")

    cells = [c for c in CELLS
             if not wanted or c[0] in wanted or c[1] in wanted or c[3] in wanted]
    print(f"\n{len(cells)} cells to run\n")

    results = []
    for label, mode, layout, method in cells:
        print("=" * 78)
        print(f"{label}  /  {method}")
        print("=" * 78, flush=True)
        try:
            r = run_cell(ctx, label, mode, layout, method)
        except Exception as exc:
            traceback.print_exc()
            r = {"label": label, "method": method, "wall": 0.0, "n": 0,
                 "phi_free": layout in PHI_FREE_LAYOUTS, "flow": False,
                 "has_q": False, "has_phi": False, "has_e": False, "identity": None,
                 "q_in_range": False, "q_min": None, "q_max": None,
                 "e2_nonzero": False, "bias_q": None, "bias_phi": None,
                 "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"}
        results.append(r)
        print(f"\n  -> flow {'PASS' if r['flow'] else 'FAIL'}   "
              f"phi={'free' if r['phi_free'] else 'fixed'}   "
              f"identity={r['identity']}   wall={r['wall']:.1f}s\n", flush=True)

    # ---- matrix ----------------------------------------------------------
    def fmt(v, spec=".2f"):
        return "-" if v is None else format(v, spec)

    lines = ["# q/phi coverage matrix", "",
             "Flow assertions must pass. Recovery is informational -- nautilus at",
             "n_eff=100 has not converged, so a wide posterior there is expected.", "",
             "phi is free only in EM-only and EM+GW. Where it is fixed, e1/e2 are",
             "correctly absent -- add_qphi_columns_to_samples backfills only when both",
             "q and phi are sampled.", "",
             "| cell | method | phi | flow | q | phi col | e1/e2 | identity | q in [0.01,1] | bias_q | bias_phi | wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n_pass = 0
    for r in results:
        flow = r["flow"]
        n_pass += flow
        note = f" {r['error']}" if r["error"] else ""
        lines.append(
            f"| {r['label']} | `{r['method']}` | {'free' if r['phi_free'] else 'fixed'} | "
            f"{'PASS' if flow else '**FAIL**'}{note} | "
            f"{'y' if r['has_q'] else 'n'} | {'y' if r['has_phi'] else 'n'} | "
            f"{'y' if r['has_e'] else 'n'} | "
            f"{fmt(r['identity'], '.1e')} | {'y' if r['q_in_range'] else 'n'} | "
            f"{fmt(r['bias_q'])} | {fmt(r['bias_phi'])} | {r['wall']:.0f}s |")
    lines += ["", f"**{n_pass}/{len(results)} cells pass the flow assertions.**"]

    path = os.path.join(OUT_DIR, "matrix.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
