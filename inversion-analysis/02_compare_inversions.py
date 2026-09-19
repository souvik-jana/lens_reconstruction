"""
Compare the Fisher inversions used across gwemfish on the saved H0s.

Methods (each mirrors one site in src/):
  raw_inv      np.linalg.inv(FM)                      inference.py:104, cfg_reference.py:839, tutorial
  raw_clamp    eigh(FM), clamp at 1e-6*lam_max        inference.py:95 (regularize=True)
  jacobi_clip  simple_pipeline._fisher_covariance     fisher / fisher-source (current)
  jacobi       scale by 1/sqrt|FM_ii|, inv, unscale   proposed (no clip, no post-process)

Reference: 60-digit mpmath inverse of the same FM.
"""

import os
import warnings

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scienceplots  # noqa: F401

from gwemfish.simple_pipeline import _fisher_covariance

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
TAGS = [f"{p}_sigma{s}" for p in ["e1e2", "q_phi"] for s in [0.05, 0.1, 0.5]]
N_SAMPLES = 20000
METHODS = ["raw_inv", "raw_clamp", "jacobi_clip", "jacobi"]


def jacobi_scale(FM):
    return 1.0 / np.sqrt(np.abs(np.diag(FM)))


def raw_inv(FM):
    return np.linalg.inv(FM)


def raw_clamp(FM):
    vals, vecs = np.linalg.eigh(FM)
    vals = np.maximum(vals, vals.max() * 1e-6)
    return (vecs * (1.0 / vals)) @ vecs.T


def jacobi_clip(FM):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov = np.asarray(_fisher_covariance(FM, None))
    fired = any("not positive definite" in str(w.message) for w in caught)
    return cov, fired


def jacobi(FM):
    s = jacobi_scale(FM)
    return np.linalg.inv(FM * s[:, None] * s[None, :]) * s[:, None] * s[None, :]


def mp_inverse(FM):
    mpmath.mp.dps = 60
    return np.array(mpmath.inverse(mpmath.matrix(FM.tolist())).tolist(), dtype=float)


def is_cholesky_ok(cov):
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


def nan_rows(u0, cov, seed):
    draws = jax.random.multivariate_normal(jax.random.PRNGKey(seed), u0, cov, shape=(N_SAMPLES,))
    return int(np.sum(~np.all(np.isfinite(np.asarray(draws)), axis=1)))


def fmt(x):
    return f"{x:.2e}"


report = []
for tag in TAGS:
    d = np.load(os.path.join(OUT, f"fisher_h0_{tag}.npz"))
    FM, u0, keys = d["FM"], d["u0"], [str(k) for k in d["keys"]]
    n = len(keys)
    s = jacobi_scale(FM)
    FM_s = FM * s[:, None] * s[None, :]
    I = np.eye(n)

    cov_ref = mp_inverse(FM)
    sig_ref = np.sqrt(np.diag(cov_ref))

    covs = {}
    clip_fired = False
    for m in METHODS:
        if m == "jacobi_clip":
            covs[m], clip_fired = jacobi_clip(FM)
        else:
            covs[m] = globals()[m](FM)

    report.append(f"\n## {tag}\n")
    report.append(f"- n params {n}; cond(FM) raw {np.linalg.cond(FM):.2e}, "
                  f"Jacobi-scaled {np.linalg.cond(FM_s):.2e}")
    report.append(f"- eig(FM_s) range [{np.linalg.eigvalsh(FM_s).min():.3e}, "
                  f"{np.linalg.eigvalsh(FM_s).max():.3e}]")
    report.append(f"- jacobi_clip warning/clip fired: {clip_fired}")
    report.append(f"- pipeline fisher-source sample std / ref sigma: "
                  f"max|ratio-1| = {np.max(np.abs(d['pipeline_samples'].std(0) / sig_ref - 1)):.3f}\n")
    report.append("| method | max abs(FM cov - I) phys | max abs(FM_s cov_s - I) scaled | "
                  "min eig cov_s | Cholesky ok | NaN rows /20k | max abs(sigma/sigma_ref - 1) | worst param |")
    report.append("|---|---|---|---|---|---|---|---|")

    stats = {}
    for m in METHODS:
        cov = covs[m]
        cov_s = cov / (s[:, None] * s[None, :])
        res_phys = np.max(np.abs(FM @ cov - I))
        res_scaled = np.abs(FM_s @ cov_s - I)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        rel = np.abs(sig / sig_ref - 1)
        worst = keys[int(np.argmax(rel))]
        stats[m] = dict(sig=sig, res_scaled=res_scaled)
        report.append(
            f"| {m} | {fmt(res_phys)} | {fmt(res_scaled.max())} | "
            f"{np.linalg.eigvalsh(0.5 * (cov_s + cov_s.T)).min():.3e} | {is_cholesky_ok(cov)} | "
            f"{nan_rows(u0, cov, 0)} | {fmt(rel.max())} | {worst} |"
        )

    report.append("\n| param | sigma_ref | " + " | ".join(METHODS) + " |")
    report.append("|---|---|" + "---|" * len(METHODS))
    for i, k in enumerate(keys):
        report.append(f"| {k} | {sig_ref[i]:.3e} | "
                      + " | ".join(f"{stats[m]['sig'][i] / sig_ref[i]:.4f}" for m in METHODS) + " |")

    fig, ax = plt.subplots(figsize=(7, 3))
    for m in METHODS:
        ax.plot(range(n), np.abs(stats[m]["sig"] / sig_ref - 1) + 1e-17, "o-", ms=2, label=m)
    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(keys, rotation=90, fontsize=5)
    ax.set_ylabel(r"$|\sigma/\sigma_{\rm ref}-1|$")
    ax.set_title(tag)
    ax.legend(fontsize=5)
    fig.savefig(os.path.join(OUT, f"sigma_error_{tag}.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(METHODS), figsize=(10, 2.8))
    for ax, m in zip(axes, METHODS):
        im = ax.imshow(np.log10(stats[m]["res_scaled"] + 1e-18), vmin=-17, vmax=0, cmap="viridis")
        ax.set_title(m, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, label=r"$\log_{10}|F_s C_s - I|$")
    fig.savefig(os.path.join(OUT, f"residual_{tag}.png"), dpi=200)
    plt.close(fig)

text = "\n".join(report)
print(text)
with open(os.path.join(OUT, "02_results.md"), "w") as f:
    f.write(text + "\n")
