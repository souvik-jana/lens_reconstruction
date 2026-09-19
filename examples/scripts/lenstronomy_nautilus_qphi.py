"""Standalone lenstronomy GW-only, q/phi parametrization: q free, phi FIXED to
truth (0.0) -- true apples-to-apples with gwemfish's q_phi deriv-approx-source /
fisher-source runs, unlike lenstronomy_infer_reparam.py (in lensing-degeneracies),
which samples e1 directly. Uniform(e1) and Uniform(q) are genuinely different
priors (nonlinear transform between them), even though the two scripts' e1/q
numeric ranges correspond to the same physical region -- this script closes that
gap by sampling q itself.

Physics, error model, solver settings, and all other bounds/truths are copied
verbatim from lenstronomy_infer_reparam.py so the only difference is the free
parameter: e1 -> q (phi fixed).

No gwemfish -- same lean, fast architecture as the original script (see
gwemfish-infer skill for why gwemfish's nautilus-source path is ~10-50x slower
per likelihood call: JAX<->host callback overhead + differentiable-solver
machinery that nested sampling never uses).
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from nautilus import Prior, Sampler
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util.param_util import phi_q2_ellipticity

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===================== SWITCHES =====================
USE_TD21 = True
USE_TD32 = True
USE_TD43 = True
USE_DLEFF1 = True
USE_DLEFF2 = True
USE_DLEFF3 = True
USE_DLEFF4 = True

SAMPLER = "emcee"  # "nautilus", "emcee", "pocomc", or "nessai" -- emcee/pocomc/nessai write to
                    # separate *_emcee.*/*_pocomc.*/*_nessai.* files, never touch nautilus's
                    # checkpoint/samples.npz

RESUME = True   # continue from the existing checkpoint (n_eff already ~6635)
NAUTILUS_NLIVE = 1000
NAUTILUS_NEFF = 7000   # lowered from 20000 -- stop near where the checkpoint already is

EMCEE_NWALKERS = 256
EMCEE_NSTEPS = 4000
EMCEE_RESUME = True  # continue from the existing HDFBackend file (same label) if it has progress
EMCEE_INIT_JITTER = 0.02  # fraction of each param's prior width, for initial walker ball
EMCEE_INIT = "fisher_hessian"  # "jitter" (small ball around TRUTHS) or "fisher_hessian" (MAP + Fisher
                       # covariance from gwemfish's fisher-source run on this same system)
EMCEE_RUN_LABEL = f"w{EMCEE_NWALKERS}_s{EMCEE_NSTEPS}_{EMCEE_INIT}"  # tags output filenames so
                                                                     # different configs don't
                                                                     # overwrite each other

POCOMC_NPARTICLES = 4000
POCOMC_NTOTAL = 8192  # None -> pocomc's own default (4096); lower for a quick smoke test
POCOMC_INIT = "jitter"  # "jitter" (small ball around fisher-source's u0, no covariance shape
                        # used) or "fisher_hessian" (u0 + full Fisher covariance, same as before)
POCOMC_INIT_JITTER = 0.02  # fraction of each param's prior width, for "jitter" init
POCOMC_PRECONDITION = False  # False disables the normalizing-flow preconditioning (diagnostic)
POCOMC_RUN_LABEL = f"n{POCOMC_NPARTICLES}_{POCOMC_INIT}_precond{POCOMC_PRECONDITION}"  # tags
                                                                                        # output
                                                                                        # filenames

NESSAI_NLIVE = 1000
NESSAI_STOPPING = 0.1  # nessai's own native evidence-based stopping criterion (its default)
NESSAI_NPOOL = 6  # None -> no multiprocessing; int -> that many worker processes (forces the
                     # `fork` start method, since this script has no __main__ guard and macOS
                     # defaults to `spawn`, which would unsafely re-run the whole module per worker)
NESSAI_INS = False  # True -> nessai's importance-nested-sampler mode (algorithmically closer to
                    # nautilus; different kwargs than classic mode, see FlowSampler call below)
NESSAI_RUN_LABEL = f"nlive{NESSAI_NLIVE}" + ("_ins" if NESSAI_INS else "") + (f"_pool{NESSAI_NPOOL}" if NESSAI_NPOOL else "")

# gwemfish fisher-source output for this exact truth system (theta_E=1, gamma=2, q=0.6, phi=0,
# GW source (0.02, 0.001)) -- reused as a MAP + Hessian-informed init for emcee/pocomc so their
# walkers/particles start from a Gaussian approximation of the posterior instead of a small ball
# around the truth or the raw prior.
FISHER_SOURCE_JSON = os.path.join(
    REPO_ROOT,
    "examples/outputs/outputs_gw_only_deriv_approx_source_vs_nautilus_source_qphi",
    "fisher_source/pipeline_outputs_fisher_source.json",
)


def load_fisher_map_and_cov():
    with open(FISHER_SOURCE_JSON) as f:
        d = json.load(f)
    fisher = d["fisher"]
    keys_fisher = fisher["keys"]  # ["lens0_gamma", "lens0_q", "T_star", "dL", "y0gw", "y1gw"]
    rename = {"lens0_gamma": "gamma", "lens0_q": "q"}
    keys_renamed = [rename.get(k, k) for k in keys_fisher]
    order = [keys_renamed.index(p) for p in PARAMS]
    u0_all = np.asarray(fisher["u0"])
    H0_all = np.asarray(fisher["H0"])
    u0 = u0_all[order]
    H0 = H0_all[np.ix_(order, order)]
    cov = np.linalg.inv(-H0)
    cov = 0.5 * (cov + cov.T)  # symmetrize away float roundoff from the matrix inverse
    return u0, cov


def sample_walkers_from_map_cov(u0, cov, n_walkers, rng):
    accepted = []
    while len(accepted) < n_walkers:
        batch = rng.multivariate_normal(u0, cov, size=n_walkers * 4)
        for row in batch:
            if all(BOUNDS[key][0] <= row[j] <= BOUNDS[key][1] for j, key in enumerate(PARAMS)):
                accepted.append(row)
                if len(accepted) == n_walkers:
                    break
    return np.array(accepted)


def sample_walkers_jitter(centre, n_walkers, jitter_frac, rng):
    # small isotropic ball around `centre` (a dict keyed by PARAMS) -- uses only the starting
    # point, no covariance/correlation shape at all.
    p0 = np.zeros((n_walkers, len(PARAMS)))
    for j, key in enumerate(PARAMS):
        lo, hi = BOUNDS[key]
        width = hi - lo
        vals = centre[key] + jitter_frac * width * rng.standard_normal(n_walkers)
        p0[:, j] = np.clip(vals, lo + 1e-9 * width, hi - 1e-9 * width)
    return p0

# ===================== TRUTH (identical to lenstronomy_infer_reparam.py) =====================
THETA_E = 1.0
GAMMA = 2.0
Q_TRUTH = 0.6
PHI_TRUTH = 0.0
E1_TRUTH, E2_TRUTH = phi_q2_ellipticity(PHI_TRUTH, Q_TRUTH)
Y0_TRUTH, Y1_TRUTH = 0.02, 0.001
T_STAR_TRUTH = 14792489.407213762
DL_TRUTH = 11213.719423532619

FRAC_TD = 0.002
FRAC_DLEFF = 0.1

SOLVER = dict(min_distance=0.02, search_window=5.0, precision_limit=1e-10, num_iter_max=200)

OBS_TD = ["td21", "td32", "td43"]
OBS_DLEFF = ["dL_eff1", "dL_eff2", "dL_eff3", "dL_eff4"]
OBS_NAMES = OBS_TD + OBS_DLEFF
PARAMS = ["gamma", "q", "T_star", "dL", "y0gw", "y1gw"]  # phi fixed, not a free param
CONTOUR_LEVELS = [0.95]
CORNER_QUANTILES = [0.16, 0.5, 0.84]

OUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_lenstronomy_nautilus_qphi")
os.makedirs(OUT_DIR, exist_ok=True)


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(to_serializable(data), f, indent=2)


def kwargs_epl(q, gamma):
    # phi fixed to PHI_TRUTH (0.0) -- only q varies here.
    e1, e2 = phi_q2_ellipticity(PHI_TRUTH, q)
    return [{"theta_E": THETA_E, "gamma": float(gamma), "e1": float(e1), "e2": float(e2),
             "center_x": 0.0, "center_y": 0.0}]


LENS = LensModel(lens_model_list=["EPL"])
SOLVER_ENGINE = LensEquationSolver(LENS)


def observables(t_star, d_l, q, gamma, y0, y1):
    kw = kwargs_epl(q, gamma)
    x, y = SOLVER_ENGINE.image_position_from_source(float(y0), float(y1), kw, **SOLVER)
    if len(x) != 4:
        return None
    phi = np.asarray(LENS.fermat_potential(x, y, kw, float(y0), float(y1)))
    mu = np.asarray(LENS.magnification(x, y, kw))
    idx = np.argsort(phi)
    phi, mu = phi[idx], mu[idx]
    td = float(t_star) * np.diff(phi)
    dleff = float(d_l) / np.sqrt(np.abs(mu))
    return {
        "td21": float(td[0]), "td32": float(td[1]), "td43": float(td[2]),
        "dL_eff1": float(dleff[0]), "dL_eff2": float(dleff[1]),
        "dL_eff3": float(dleff[2]), "dL_eff4": float(dleff[3]),
    }


TRUTHS = {
    "gamma": GAMMA, "q": Q_TRUTH, "T_star": T_STAR_TRUTH, "dL": DL_TRUTH,
    "y0gw": Y0_TRUTH, "y1gw": Y1_TRUTH,
}
OBS = observables(T_STAR_TRUTH, DL_TRUTH, Q_TRUTH, GAMMA, Y0_TRUTH, Y1_TRUTH)
if OBS is None:
    raise SystemExit("truth centre source is not a 4-image quad")

SIGMA = {
    "td21": FRAC_TD * abs(OBS["td21"]),
    "td32": FRAC_TD * abs(OBS["td32"]),
    "td43": FRAC_TD * abs(OBS["td43"]),
    "dL_eff1": FRAC_DLEFF * abs(OBS["dL_eff1"]),
    "dL_eff2": FRAC_DLEFF * abs(OBS["dL_eff2"]),
    "dL_eff3": FRAC_DLEFF * abs(OBS["dL_eff3"]),
    "dL_eff4": FRAC_DLEFF * abs(OBS["dL_eff4"]),
}

USE = {
    "td21": USE_TD21, "td32": USE_TD32, "td43": USE_TD43,
    "dL_eff1": USE_DLEFF1, "dL_eff2": USE_DLEFF2,
    "dL_eff3": USE_DLEFF3, "dL_eff4": USE_DLEFF4,
}
ACTIVE = [k for k in OBS_NAMES if USE[k]]

# q bounds derived from lenstronomy_infer_reparam.py's e1 bounds (0.04, 0.48) at
# phi=0 -- SAME physical region, but sampled here as Uniform(q) directly rather
# than Uniform(e1). gamma/T_star/dL/y0gw/y1gw bounds copied verbatim.
E1_BOUNDS_ORIGINAL = (0.04, 0.48)
Q_BOUNDS = (
    (1 - E1_BOUNDS_ORIGINAL[1]) / (1 + E1_BOUNDS_ORIGINAL[1]),
    (1 - E1_BOUNDS_ORIGINAL[0]) / (1 + E1_BOUNDS_ORIGINAL[0]),
)
BOUNDS = {
    "gamma": (0.8, 2.8),
    "q": Q_BOUNDS,
    "T_star": (0.2 * T_STAR_TRUTH, 2.0 * T_STAR_TRUTH),
    "dL": (0.5 * DL_TRUTH, 1.89 * DL_TRUTH),
    "y0gw": (0.00375, 0.045),
    "y1gw": (1e-5, 0.003),
}

print("=" * 72)
print("  lenstronomy_nautilus_qphi  centre/full  nautilus (q free, phi fixed)")
print(f"  free={PARAMS}  active={ACTIVE}")
print("  priors [left, truth, right]:")
for key in PARAMS:
    lo, hi = BOUNDS[key]
    print(f"    {key:>7}: [{lo:.6g}, {TRUTHS[key]:.6g}, {hi:.6g}]")
print("=" * 72)


def loglike(gamma, q, t_star, d_l, y0, y1):
    m = observables(t_star, d_l, q, gamma, y0, y1)
    if m is None:
        return -1e300
    chi2 = sum(((m[k] - OBS[k]) / SIGMA[k]) ** 2 for k in ACTIVE)
    return -0.5 * chi2


def loglike_dict(d):
    return loglike(d["gamma"], d["q"], d["T_star"], d["dL"], d["y0gw"], d["y1gw"])


save_json(
    {
        "source": "centre",
        "obs_tag": "full",
        "theta_E": THETA_E,
        "phi_fixed": PHI_TRUTH,
        "active_observables": ACTIVE,
        "frac_td": FRAC_TD,
        "frac_dL_eff": FRAC_DLEFF,
        "bounds": BOUNDS,
        "truths": TRUTHS,
        "observed": OBS,
        "sigma": SIGMA,
    },
    os.path.join(OUT_DIR, "cfg.json"),
)
save_json(TRUTHS, os.path.join(OUT_DIR, "truths.json"))

def log_prior(theta):
    for key, val in zip(PARAMS, theta):
        lo, hi = BOUNDS[key]
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def log_prob_emcee(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    d = dict(zip(PARAMS, theta))
    ll = loglike_dict(d)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


if SAMPLER == "nautilus":
    print("\n--- nautilus ---")
    prior = Prior()
    for key in PARAMS:
        prior.add_parameter(key, dist=BOUNDS[key])
    sampler = Sampler(
        prior, loglike_dict, n_live=NAUTILUS_NLIVE,
        filepath=os.path.join(OUT_DIR, "nautilus.hdf5"), resume=RESUME)
    t0 = time.time()
    sampler.run(n_eff=NAUTILUS_NEFF, verbose=True)
    elapsed = time.time() - t0
    points, log_w, _ = sampler.posterior(return_as_dict=True)
    weights = np.exp(log_w - np.max(log_w))
    samp = {k: np.asarray(points[k]) for k in PARAMS}
    np.savez(os.path.join(OUT_DIR, "samples.npz"), weights=weights, **samp)

    stats = {}
    for k, v in samp.items():
        m = float(np.average(v, weights=weights))
        s = float(np.sqrt(np.average((v - m) ** 2, weights=weights)))
        stats[k] = {"mean": m, "std": s}
        print(f"  {k}: mean={m:.6g}  std={s:.6g}")
    print(f"  runtime={elapsed:.0f}s  logZ={float(sampler.log_z):.4f}")

    # native convergence/health diagnostics nautilus tracks -- f_live (evidence fraction still
    # in the live set, its own stopping-criterion check) and eta (sampling efficiency)
    diagnostics_nautilus = {
        "n_eff_achieved": float(sampler.n_eff),
        "f_live": float(sampler.f_live),
        "asymptotic_sampling_efficiency": float(sampler.asymptotic_sampling_efficiency),
    }
    print(f"  nautilus native diagnostics: {diagnostics_nautilus}")

    save_json(
        {
            "method": "lenstronomy_nautilus_qphi",
            "n_live": NAUTILUS_NLIVE,
            "n_eff": NAUTILUS_NEFF,
            "runtime_seconds": round(elapsed, 1),
            "log_z": float(sampler.log_z),
            "diagnostics": diagnostics_nautilus,
            "stats": stats,
            "priors": BOUNDS,
        },
        os.path.join(OUT_DIR, "run_meta.json"),
    )

    samples_path = os.path.join(OUT_DIR, "samples.npz")
    meta_path = os.path.join(OUT_DIR, "run_meta.json")
    corner_path = os.path.join(OUT_DIR, "corner_centre_full_lenstronomy_nautilus_qphi.png")
    corner_contours_path = os.path.join(OUT_DIR, "corner_contours_centre_full_lenstronomy_nautilus_qphi.png")

elif SAMPLER == "emcee":
    import emcee

    print("\n--- emcee ---")
    ndim = len(PARAMS)
    rng = np.random.default_rng(0)

    if EMCEE_INIT == "jitter":
        p0 = sample_walkers_jitter(TRUTHS, EMCEE_NWALKERS, EMCEE_INIT_JITTER, rng)
    elif EMCEE_INIT == "fisher_hessian":
        u0_fisher, cov_fisher = load_fisher_map_and_cov()
        p0 = sample_walkers_from_map_cov(u0_fisher, cov_fisher, EMCEE_NWALKERS, rng)
        print(f"  init from fisher-source MAP+Hessian: u0={dict(zip(PARAMS, u0_fisher))}")
    else:
        raise ValueError(f"unknown EMCEE_INIT: {EMCEE_INIT!r}")

    # persist the full (steps, walkers, params) chain to disk via emcee's own backend --
    # without this, only the flattened post-processed samples survive, and native convergence
    # diagnostics that need per-walker structure (r-hat) can't be computed after the fact.
    backend_path = os.path.join(OUT_DIR, f"emcee_backend_{EMCEE_RUN_LABEL}.h5")
    backend = emcee.backends.HDFBackend(backend_path)
    if EMCEE_RESUME and os.path.exists(backend_path) and backend.iteration > 0:
        print(f"  resuming from existing backend: {backend.iteration} iterations already done")
        p0 = None  # None -> continue from the backend's last stored state
        n_steps_remaining = max(0, EMCEE_NSTEPS - backend.iteration)
    else:
        backend.reset(EMCEE_NWALKERS, ndim)
        n_steps_remaining = EMCEE_NSTEPS

    t0 = time.time()
    es = emcee.EnsembleSampler(EMCEE_NWALKERS, ndim, log_prob_emcee, backend=backend)
    if n_steps_remaining > 0:
        es.run_mcmc(p0, n_steps_remaining, progress=True)
    elapsed = time.time() - t0

    try:
        tau = es.get_autocorr_time(quiet=True)
        burnin = int(3 * np.max(tau))
        thin = max(1, int(np.max(tau) / 2))
    except Exception:
        tau = None
        burnin = EMCEE_NSTEPS // 3
        thin = 1

    chain = es.get_chain(discard=burnin, thin=thin, flat=True)
    accept_frac = float(np.mean(es.acceptance_fraction))
    print(f"  acceptance fraction: {accept_frac:.3f}")
    print(f"  autocorr time: {tau}")
    print(f"  burnin={burnin} thin={thin} n_samples={chain.shape[0]}")

    # native convergence diagnostics (arviz's own rhat/ess, not hand-rolled) computed on the
    # full per-walker chain (post burn-in, unthinned so rhat/ess see the real sample count)
    import arviz as az

    idata = az.from_emcee(es, var_names=PARAMS)
    idata_post_burnin = idata.sel(draw=slice(burnin, None))
    rhat = az.rhat(idata_post_burnin)
    ess_bulk = az.ess(idata_post_burnin, method="bulk")
    rhat_dict = {k: float(rhat[k].values) for k in PARAMS}
    ess_dict = {k: float(ess_bulk[k].values) for k in PARAMS}
    print(f"  r-hat (arviz, per param): {rhat_dict}")
    print(f"  ess_bulk (arviz, per param): {ess_dict}")

    samp = {k: chain[:, j] for j, k in enumerate(PARAMS)}
    weights = np.ones(chain.shape[0])
    np.savez(os.path.join(OUT_DIR, f"samples_emcee_{EMCEE_RUN_LABEL}.npz"), weights=weights, **samp)

    stats = {}
    for k, v in samp.items():
        m = float(np.mean(v))
        s = float(np.std(v))
        stats[k] = {"mean": m, "std": s}
        print(f"  {k}: mean={m:.6g}  std={s:.6g}")
    print(f"  runtime={elapsed:.0f}s")

    save_json(
        {
            "method": "lenstronomy_emcee_qphi",
            "nwalkers": EMCEE_NWALKERS,
            "nsteps": EMCEE_NSTEPS,
            "burnin": burnin,
            "thin": thin,
            "acceptance_fraction": accept_frac,
            "autocorr_time": None if tau is None else tau.tolist(),
            "rhat": rhat_dict,
            "ess_bulk": ess_dict,
            "runtime_seconds": round(elapsed, 1),
            "stats": stats,
            "priors": BOUNDS,
        },
        os.path.join(OUT_DIR, f"run_meta_emcee_{EMCEE_RUN_LABEL}.json"),
    )

    samples_path = os.path.join(OUT_DIR, f"samples_emcee_{EMCEE_RUN_LABEL}.npz")
    meta_path = os.path.join(OUT_DIR, f"run_meta_emcee_{EMCEE_RUN_LABEL}.json")
    corner_path = os.path.join(OUT_DIR, f"corner_centre_full_lenstronomy_emcee_{EMCEE_RUN_LABEL}_qphi.png")
    corner_contours_path = os.path.join(OUT_DIR, f"corner_contours_centre_full_lenstronomy_emcee_{EMCEE_RUN_LABEL}_qphi.png")

elif SAMPLER == "pocomc":
    import pocomc
    from scipy.stats import uniform as scipy_uniform

    print("\n--- pocomc ---")

    def log_likelihood_pocomc(theta):
        return loglike_dict(dict(zip(PARAMS, theta)))

    prior_pocomc = pocomc.Prior([scipy_uniform(BOUNDS[k][0], BOUNDS[k][1] - BOUNDS[k][0]) for k in PARAMS])

    pc_sampler = pocomc.Sampler(
        prior_pocomc, log_likelihood_pocomc, n_dim=len(PARAMS),
        n_effective=POCOMC_NPARTICLES, n_active=POCOMC_NPARTICLES // 2,
        precondition=POCOMC_PRECONDITION,
        output_dir=os.path.join(OUT_DIR, f"pocomc_state_{POCOMC_RUN_LABEL}"), output_label="pmc",
    )

    # seed the initial particle cloud instead of a raw prior draw. This only changes where the
    # SMC run *starts*; prior_pocomc.logpdf (the real, flat prior over BOUNDS, same range as
    # nautilus's own Prior) is untouched, so the target posterior is unaffected.
    rng = np.random.default_rng(0)
    if POCOMC_INIT == "jitter":
        u0_fisher, _ = load_fisher_map_and_cov()
        centre = dict(zip(PARAMS, u0_fisher))
        pc_sampler.prior_samples = sample_walkers_jitter(centre, pc_sampler.n_prior, POCOMC_INIT_JITTER, rng)
        print(f"  seeded {pc_sampler.n_prior} initial particles: small jitter around fisher-source u0 (no Hessian shape)")
    elif POCOMC_INIT == "fisher_hessian":
        u0_fisher, cov_fisher = load_fisher_map_and_cov()
        pc_sampler.prior_samples = sample_walkers_from_map_cov(u0_fisher, cov_fisher, pc_sampler.n_prior, rng)
        print(f"  seeded {pc_sampler.n_prior} initial particles from fisher-source MAP+Hessian")
    else:
        raise ValueError(f"unknown POCOMC_INIT: {POCOMC_INIT!r}")
    pc_sampler.scaler.fit(pc_sampler.prior_samples)  # normally done inside run(), skipped since
                                                       # we pre-set prior_samples ourselves

    t0 = time.time()
    if POCOMC_NTOTAL is None:
        pc_sampler.run()
    else:
        pc_sampler.run(n_total=POCOMC_NTOTAL)
    elapsed = time.time() - t0

    samples, weights, logl, logp = pc_sampler.posterior()
    print(f"  n_samples={samples.shape[0]}  runtime={elapsed:.0f}s")

    # pocomc's own native per-particle diagnostics (not hand-rolled) -- ess/accept/beta/logz
    # tracked internally throughout the run; summarize rather than dropping them.
    pc_results = pc_sampler.results
    logz_pocomc, logz_err_pocomc = pc_sampler.evidence()
    diagnostics_pocomc = {
        "ess_min": float(np.min(pc_results["ess"])),
        "ess_mean": float(np.mean(pc_results["ess"])),
        "ess_final": float(pc_results["ess"][-1]),
        "accept_mean": float(np.mean(pc_results["accept"])),
        "beta_final": float(pc_results["beta"][-1]),
        "logz_final": float(pc_results["logz"][-1]),
        "log_evidence": float(logz_pocomc),
        "log_evidence_error": float(logz_err_pocomc),
        "total_calls": int(pc_results["calls"][-1]),
    }
    print(f"  pocomc native diagnostics: {diagnostics_pocomc}")

    samp = {k: samples[:, j] for j, k in enumerate(PARAMS)}
    np.savez(os.path.join(OUT_DIR, f"samples_pocomc_{POCOMC_RUN_LABEL}.npz"), weights=weights, **samp)

    stats = {}
    for k, v in samp.items():
        m = float(np.average(v, weights=weights))
        s = float(np.sqrt(np.average((v - m) ** 2, weights=weights)))
        stats[k] = {"mean": m, "std": s}
        print(f"  {k}: mean={m:.6g}  std={s:.6g}")

    save_json(
        {
            "method": "lenstronomy_pocomc_qphi",
            "n_particles": POCOMC_NPARTICLES,
            "n_samples": int(samples.shape[0]),
            "diagnostics": diagnostics_pocomc,
            "runtime_seconds": round(elapsed, 1),
            "stats": stats,
            "priors": BOUNDS,
        },
        os.path.join(OUT_DIR, f"run_meta_pocomc_{POCOMC_RUN_LABEL}.json"),
    )

    samples_path = os.path.join(OUT_DIR, f"samples_pocomc_{POCOMC_RUN_LABEL}.npz")
    meta_path = os.path.join(OUT_DIR, f"run_meta_pocomc_{POCOMC_RUN_LABEL}.json")
    corner_path = os.path.join(OUT_DIR, f"corner_centre_full_lenstronomy_pocomc_{POCOMC_RUN_LABEL}_qphi.png")
    corner_contours_path = os.path.join(OUT_DIR, f"corner_contours_centre_full_lenstronomy_pocomc_{POCOMC_RUN_LABEL}_qphi.png")

elif SAMPLER == "nessai":
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model

    print("\n--- nessai ---")

    class LensModelNessai(Model):
        def __init__(self):
            self.names = list(PARAMS)
            self.bounds = {k: list(BOUNDS[k]) for k in PARAMS}

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x), dtype="float")
            for n in self.names:
                log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
            return log_p

        def log_likelihood(self, x):
            return loglike_dict({k: x[k] for k in self.names})

        def to_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = (x[n] - self.bounds[n][0]) / np.ptp(self.bounds[n])
            return x_out

        def from_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = np.ptp(self.bounds[n]) * x[n] + self.bounds[n][0]
            return x_out

    nessai_model = LensModelNessai()

    # classic mode takes nlive/stopping; importance_nested_sampler mode (ImportanceNestedSampler)
    # doesn't accept `stopping` -- its own stopping_criterion/tolerance defaults are used instead.
    nessai_kwargs = dict(nlive=NESSAI_NLIVE)
    if not NESSAI_INS:
        nessai_kwargs["stopping"] = NESSAI_STOPPING
    else:
        nessai_kwargs["min_samples"] = max(1, NESSAI_NLIVE // 2)  # must be < nlive
    if NESSAI_NPOOL is not None:
        import multiprocessing

        multiprocessing.set_start_method("fork", force=True)
        nessai_kwargs["n_pool"] = NESSAI_NPOOL

    t0 = time.time()
    nessai_sampler = FlowSampler(
        nessai_model,
        output=os.path.join(OUT_DIR, f"nessai_state_{NESSAI_RUN_LABEL}"),
        importance_nested_sampler=NESSAI_INS,
        resume=False, seed=0,
        **nessai_kwargs,
    )
    nessai_sampler.run()
    elapsed = time.time() - t0

    post = nessai_sampler.posterior_samples
    print(f"  n_samples={len(post)}  runtime={elapsed:.0f}s")
    print(f"  log_evidence={nessai_sampler.log_evidence:.4f} +/- {nessai_sampler.log_evidence_error:.4f}")

    # native diagnostics from the underlying (classic-mode) NestedSampler -- includes the
    # insertion-index KS test, the standard rigorous nested-sampling convergence check (p<0.05
    # on the final, whole-run test signals a real problem, not just low ESS/high autocorrelation).
    diagnostics_nessai = {}
    if hasattr(nessai_sampler.ns, "check_insertion_indices"):
        nessai_sampler.ns.check_insertion_indices(rolling=False)
        diagnostics_nessai["insertion_index_ks_pvalue"] = float(nessai_sampler.ns.final_p_value)
        diagnostics_nessai["insertion_index_ks_statistic"] = float(nessai_sampler.ns.final_ks_statistic)
        diagnostics_nessai["mean_acceptance"] = float(nessai_sampler.ns.mean_acceptance)
        diagnostics_nessai["information_nats"] = float(nessai_sampler.ns.information)
        diagnostics_nessai["posterior_ess"] = float(nessai_sampler.ns.posterior_effective_sample_size)
        diagnostics_nessai["total_likelihood_evaluations"] = int(nessai_sampler.ns.total_likelihood_evaluations)
    print(f"  nessai native diagnostics: {diagnostics_nessai}")

    samp = {k: np.asarray(post[k]) for k in PARAMS}
    weights = np.ones(len(post))  # nessai's posterior_samples are already equally-weighted
    np.savez(os.path.join(OUT_DIR, f"samples_nessai_{NESSAI_RUN_LABEL}.npz"), weights=weights, **samp)

    stats = {}
    for k, v in samp.items():
        m = float(np.mean(v))
        s = float(np.std(v))
        stats[k] = {"mean": m, "std": s}
        print(f"  {k}: mean={m:.6g}  std={s:.6g}")

    save_json(
        {
            "method": "lenstronomy_nessai_qphi",
            "nlive": NESSAI_NLIVE,
            "stopping": NESSAI_STOPPING,
            "n_samples": len(post),
            "log_evidence": float(nessai_sampler.log_evidence),
            "log_evidence_error": float(nessai_sampler.log_evidence_error),
            "diagnostics": diagnostics_nessai,
            "runtime_seconds": round(elapsed, 1),
            "stats": stats,
            "priors": BOUNDS,
        },
        os.path.join(OUT_DIR, f"run_meta_nessai_{NESSAI_RUN_LABEL}.json"),
    )

    samples_path = os.path.join(OUT_DIR, f"samples_nessai_{NESSAI_RUN_LABEL}.npz")
    meta_path = os.path.join(OUT_DIR, f"run_meta_nessai_{NESSAI_RUN_LABEL}.json")
    corner_path = os.path.join(OUT_DIR, f"corner_centre_full_lenstronomy_nessai_{NESSAI_RUN_LABEL}_qphi.png")
    corner_contours_path = os.path.join(OUT_DIR, f"corner_contours_centre_full_lenstronomy_nessai_{NESSAI_RUN_LABEL}_qphi.png")

else:
    raise ValueError(f"unknown SAMPLER: {SAMPLER!r}")

labels = PARAMS
truth_arr = [TRUTHS[k] for k in PARAMS]
arr = np.column_stack([samp[k] for k in PARAMS])

fig = corner.corner(
    arr, labels=labels, truths=truth_arr, weights=weights,
    quantiles=CORNER_QUANTILES, show_titles=True, title_fmt=".3f",
    hist_kwargs={"density": True},
)
fig.savefig(corner_path, dpi=200)
plt.close(fig)

fig = corner.corner(
    arr, labels=labels, truths=truth_arr, weights=weights,
    quantiles=CORNER_QUANTILES, show_titles=True, title_fmt=".3f",
    levels=CONTOUR_LEVELS, plot_datapoints=False, plot_density=False,
    fill_contours=False, no_fill_contours=True, hist_kwargs={"density": True},
)
fig.savefig(corner_contours_path, dpi=200)
plt.close(fig)

print(f"\nDone ({SAMPLER}) -> {samples_path}")
