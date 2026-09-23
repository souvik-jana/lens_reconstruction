"""GW-only geometry inference from Fermat / magnification *ratios* (lenstronomy).

Truth system matches tutorial/tutorial_gw_only_pool.py (EPL+SHEAR, q=0.5,
phi=30 deg, source ~centre). Observables cancel T_star and dL:

    r32_21, r43_21  from sorted Fermat increments
    mu21, mu31, mu41 from |mu| ratios

Fractional errors map from tutorial absolute scales with a factor of 2:
    FRAC_MU = 2 * sigma_dL_eff,  FRAC_TD = 2 * sigma_td

Sampler switch: nautilus | emcee | nessai (each with optional pool).

    uv run python examples/scripts/geometry_ratios_lenstronomy.py
"""

import copy
import json
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from nautilus import Prior, Sampler
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util.param_util import phi_q2_ellipticity

from gwemfish import (
    make_default_cfg,
    plot_system_observation,
    setup_em_observation,
    setup_gw_observation,
)
from gwemfish.corner_plot_utils import plot_custom_params

if "science" in plt.style.available:
    plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===================== SWITCHES =====================
USE_R32_21 = True
USE_R43_21 = True
USE_MU21 = True
USE_MU31 = True
USE_MU41 = True

SAMPLER = "nautilus"  # "nautilus" | "emcee" | "nessai"
RESUME = False

NAUTILUS_NLIVE = 1000
NAUTILUS_NEFF = 5000
NAUTILUS_POOL = 4  # None -> serial

EMCEE_NWALKERS = 32
EMCEE_NSTEPS = 4000
EMCEE_DISCARD = 1000
EMCEE_THIN = 5
EMCEE_POOL = 10 #None  # int -> ProcessPool workers; None -> serial
EMCEE_INIT_JITTER = 0.02

NESSAI_NLIVE = 2000
NESSAI_STOPPING = 0.2
NESSAI_NPOOL = 4  # None -> serial
NESSAI_INS = False

# ===================== TRUTH (tutorial_gw_only_pool) =====================
LENS_Q = 0.5
LENS_PHI_DEG = 30.0
Y0_TRUTH = 0.02
Y1_TRUTH = 0.00001
N_IMAGES = 4

SIGMA_TD = 0.001
SIGMA_DL_EFF = 0.1
FRAC_TD = [2 * SIGMA_TD, 2 * SIGMA_TD]
FRAC_MU = [2 * SIGMA_DL_EFF] * 3

SOLVER = dict(
    min_distance=0.05,
    search_window=5.0,
    precision_limit=1e-10,
    num_iter_max=200,
)

OBS_NAMES = ["r32_21", "r43_21", "mu21", "mu31", "mu41"]
PARAMS = ["q", "gamma", "y0gw", "y1gw"]
CONTOUR_LEVELS = [0.95]
CORNER_QUANTILES = [0.16, 0.5, 0.84]

OUT_DIR = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_lenstronomy", SAMPLER
)
os.makedirs(OUT_DIR, exist_ok=True)

# Match geometry_ratios_lenstronomy_absolute geometry priors
BOUNDS = {
    "q": (0.3, 0.75),
    "gamma": (1.2, 2.8),
    "y0gw": (-0.001, 0.05),
    "y1gw": (-0.008, 0.028),
}


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


def build_truth_cfg():
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens_mass_parametrization"] = "q_phi"
    cfg["lens"]["kwargs_lens"] = copy.deepcopy(cfg["lens"]["kwargs_lens"])
    e1, e2 = phi_q2_ellipticity(np.deg2rad(LENS_PHI_DEG), LENS_Q)
    cfg["lens"]["kwargs_lens"][0]["e1"] = float(e1)
    cfg["lens"]["kwargs_lens"][0]["e2"] = float(e2)
    cfg["gw"]["n_images"] = N_IMAGES
    cfg["gw"]["source_pos"] = (Y0_TRUTH, Y1_TRUTH)
    cfg["gw"]["source_box_half_width"] = 0.8
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["gw"]["error_scales"]["sigma_td"] = SIGMA_TD
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = SIGMA_DL_EFF
    cfg["em"]["kwargs_source"][0]["center_x"] = float(Y0_TRUTH)
    cfg["em"]["kwargs_source"][0]["center_y"] = float(Y1_TRUTH)
    cfg["output"]["output_dir"] = OUT_DIR
    return cfg


CFG_TRUTH = build_truth_cfg()
KW0 = CFG_TRUTH["lens"]["kwargs_lens"][0]
KW1 = CFG_TRUTH["lens"]["kwargs_lens"][1]
THETA_E = float(KW0["theta_E"])
GAMMA_TRUTH = float(KW0["gamma"])
PHI_RAD = float(np.deg2rad(LENS_PHI_DEG))
SHEAR_FIXED = {
    "gamma1": float(KW1["gamma1"]),
    "gamma2": float(KW1["gamma2"]),
    "ra_0": float(KW1["ra_0"]),
    "dec_0": float(KW1["dec_0"]),
}
CENTER_X = float(KW0["center_x"])
CENTER_Y = float(KW0["center_y"])

LENS = LensModel(lens_model_list=["EPL", "SHEAR"])
SOLVER_ENGINE = LensEquationSolver(LENS)


def as_float(x):
    return float(np.asarray(x).reshape(-1)[0])


def kwargs_lens(q, gamma):
    e1, e2 = phi_q2_ellipticity(PHI_RAD, as_float(q))
    return [
        {
            "theta_E": THETA_E,
            "gamma": as_float(gamma),
            "e1": as_float(e1),
            "e2": as_float(e2),
            "center_x": CENTER_X,
            "center_y": CENTER_Y,
        },
        dict(SHEAR_FIXED),
    ]


def sorted_images(q, gamma, y0, y1):
    kw = kwargs_lens(q, gamma)
    y0s, y1s = as_float(y0), as_float(y1)
    x, y = SOLVER_ENGINE.image_position_from_source(
        y0s, y1s, kw, **SOLVER
    )
    if len(x) != N_IMAGES:
        return None
    phi = np.asarray(LENS.fermat_potential(x, y, kw, y0s, y1s))
    mu = np.asarray(LENS.magnification(x, y, kw))
    idx = np.argsort(phi)
    return x[idx], y[idx], phi[idx], mu[idx]


def observables(q, gamma, y0, y1):
    sol = sorted_images(q, gamma, y0, y1)
    if sol is None:
        return None
    _, _, phi, mu = sol
    dphi = np.diff(phi)
    amu = np.abs(mu)
    return {
        "dphi21": float(dphi[0]),
        "dphi32": float(dphi[1]),
        "dphi43": float(dphi[2]),
        "mu": [float(m) for m in mu],
        "amu": [float(m) for m in amu],
        "r32_21": float(dphi[1] / dphi[0]),
        "r43_21": float(dphi[2] / dphi[0]),
        "mu21": float(amu[1] / amu[0]),
        "mu31": float(amu[2] / amu[0]),
        "mu41": float(amu[3] / amu[0]),
    }


def gwemfish_observables(ctx):
    dg = ctx["data_GW"]
    td = np.asarray(dg["time_delays_in_seconds"])
    mu = np.asarray(dg["mu"])
    amu = np.abs(mu)
    return {
        "td21": float(td[0]),
        "td32": float(td[1]),
        "td43": float(td[2]),
        "mu": [float(m) for m in mu],
        "amu": [float(m) for m in amu],
        "r32_21": float(td[1] / td[0]),
        "r43_21": float(td[2] / td[0]),
        "mu21": float(amu[1] / amu[0]),
        "mu31": float(amu[2] / amu[0]),
        "mu41": float(amu[3] / amu[0]),
    }


def print_observables(obs_l, obs_g):
    print("\n  cross-check lenstronomy vs gwemfish:")
    print(f"    dphi21: {obs_l['dphi21']:+.6e}  td21: {obs_g['td21']:+.6e} s")
    print(f"    dphi32: {obs_l['dphi32']:+.6e}  td32: {obs_g['td32']:+.6e} s")
    print(f"    dphi43: {obs_l['dphi43']:+.6e}  td43: {obs_g['td43']:+.6e} s")
    print(f"    |mu| ls: {obs_l['amu']}")
    print(f"    |mu| gw: {obs_g['amu']}")
    for k in OBS_NAMES:
        print(
            f"    {k:>7}: {obs_l[k]:+.6f}  {obs_g[k]:+.6f}  "
            f"(d={obs_l[k] - obs_g[k]:+.2e})"
        )


TRUTHS = {
    "q": LENS_Q,
    "gamma": GAMMA_TRUTH,
    "y0gw": Y0_TRUTH,
    "y1gw": Y1_TRUTH,
}

OBS = observables(LENS_Q, GAMMA_TRUTH, Y0_TRUTH, Y1_TRUTH)
if OBS is None:
    raise SystemExit(
        f"truth source ({Y0_TRUTH}, {Y1_TRUTH}) is not a {N_IMAGES}-image system"
    )

FRAC = dict(zip(OBS_NAMES, FRAC_TD + FRAC_MU))
SIGMA = {k: FRAC[k] * abs(OBS[k]) for k in OBS_NAMES}

USE = {
    "r32_21": USE_R32_21,
    "r43_21": USE_R43_21,
    "mu21": USE_MU21,
    "mu31": USE_MU31,
    "mu41": USE_MU41,
}
ACTIVE = [k for k in OBS_NAMES if USE[k]]
OBS_TAG = "full" if len(ACTIVE) == len(OBS_NAMES) else "_".join(ACTIVE)


def loglike(q, gamma, y0, y1):
    m = observables(q, gamma, y0, y1)
    if m is None:
        return -1e300
    chi2 = sum(((m[k] - OBS[k]) / SIGMA[k]) ** 2 for k in ACTIVE)
    return -0.5 * chi2


def loglike_dict(d):
    return loglike(d["q"], d["gamma"], d["y0gw"], d["y1gw"])


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
    ll = loglike(*theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def sample_walkers_jitter(centre, n_walkers, jitter_frac, rng):
    p0 = np.zeros((n_walkers, len(PARAMS)))
    for j, key in enumerate(PARAMS):
        lo, hi = BOUNDS[key]
        width = hi - lo
        vals = centre[key] + jitter_frac * width * rng.standard_normal(n_walkers)
        p0[:, j] = np.clip(vals, lo + 1e-9 * width, hi - 1e-9 * width)
    return p0


def plot_corners(samp, weights, tag):
    plot_custom_params(
        samp,
        PARAMS,
        truths=TRUTHS,
        weights=weights,
        show_titles=True,
        title_fmt=".4f",
        quantiles=CORNER_QUANTILES,
        hist_kwargs={"density": True},
        save_path=os.path.join(OUT_DIR, f"corner_{tag}.png"),
    )
    plt.close("all")
    plot_custom_params(
        samp,
        PARAMS,
        truths=TRUTHS,
        weights=weights,
        show_titles=True,
        title_fmt=".4f",
        quantiles=CORNER_QUANTILES,
        levels=CONTOUR_LEVELS,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        hist_kwargs={"density": True},
        save_path=os.path.join(OUT_DIR, f"corner_contours_{tag}.png"),
    )
    plt.close("all")


def weighted_stats(samp, weights):
    stats = {}
    for k, v in samp.items():
        m = float(np.average(v, weights=weights))
        s = float(np.sqrt(np.average((v - m) ** 2, weights=weights)))
        stats[k] = {"mean": m, "std": s}
        print(f"  {k}: mean={m:.6g}  std={s:.6g}")
    return stats


if __name__ == "__main__":
    print("=" * 72)
    print("  geometry_ratios_lenstronomy")
    print(
        f"  truth: theta_E={THETA_E} gamma={GAMMA_TRUTH} q={LENS_Q} "
        f"phi={LENS_PHI_DEG} deg  source=({Y0_TRUTH}, {Y1_TRUTH})"
    )
    print(f"  free={PARAMS}  active={ACTIVE}  sampler={SAMPLER}")
    print(f"  FRAC_TD={FRAC_TD}  FRAC_MU={FRAC_MU}")
    print("  priors [left, truth, right]:")
    for key in PARAMS:
        lo, hi = BOUNDS[key]
        print(f"    {key:>5}: [{lo:.6g}, {TRUTHS[key]:.6g}, {hi:.6g}]")
    print("=" * 72)

    ctx = setup_em_observation(cfg=CFG_TRUTH)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    print(f"Simulated {ctx['n_images']} GW images.")
    plot_system_observation(
        ctx,
        cfg={
            "output": {
                "output_dir": OUT_DIR,
                "save_system_plot_path": "system_observation.png",
            }
        },
    )
    plt.close("all")

    obs_gw = gwemfish_observables(ctx)
    print_observables(OBS, obs_gw)

    save_json(
        {
            "observed": OBS,
            "sigma": SIGMA,
            "frac": FRAC,
            "active": ACTIVE,
            "sigma_td": SIGMA_TD,
            "sigma_dL_eff": SIGMA_DL_EFF,
            "frac_td": FRAC_TD,
            "frac_mu": FRAC_MU,
            "gwemfish": obs_gw,
            "truths": TRUTHS,
            "bounds": BOUNDS,
            "theta_E": THETA_E,
            "phi_deg": LENS_PHI_DEG,
        },
        os.path.join(OUT_DIR, "observables.json"),
    )

    run_label = f"{SAMPLER}_{OBS_TAG}"
    samples_path = os.path.join(OUT_DIR, f"samples_{run_label}.npz")
    meta_path = os.path.join(OUT_DIR, f"run_meta_{run_label}.json")

    if SAMPLER == "nautilus":
        print("\n--- nautilus ---")
        prior = Prior()
        for key in PARAMS:
            prior.add_parameter(key, dist=BOUNDS[key])
        sampler = Sampler(
            prior,
            loglike_dict,
            n_live=NAUTILUS_NLIVE,
            filepath=os.path.join(OUT_DIR, f"nautilus_{OBS_TAG}.hdf5"),
            resume=RESUME,
            pool=NAUTILUS_POOL,
        )
        t0 = time.time()
        sampler.run(n_eff=NAUTILUS_NEFF, verbose=True)
        elapsed = time.time() - t0
        points, log_w, _ = sampler.posterior(return_as_dict=True)
        weights = np.exp(log_w - np.max(log_w))
        samp = {k: np.asarray(points[k]) for k in PARAMS}
        np.savez(samples_path, weights=weights, **samp)
        stats = weighted_stats(samp, weights)
        print(f"  runtime={elapsed:.0f}s  logZ={float(sampler.log_z):.4f}")
        save_json(
            {
                "method": "geometry_ratios_lenstronomy_nautilus",
                "n_live": NAUTILUS_NLIVE,
                "n_eff": NAUTILUS_NEFF,
                "pool": NAUTILUS_POOL,
                "runtime_seconds": round(elapsed, 1),
                "log_z": float(sampler.log_z),
                "n_eff_achieved": float(sampler.n_eff),
                "stats": stats,
                "priors": BOUNDS,
                "active": ACTIVE,
            },
            meta_path,
        )
        plot_corners(samp, weights, run_label)

    elif SAMPLER == "emcee":
        import emcee
        from multiprocessing import Pool

        print("\n--- emcee ---")
        ndim = len(PARAMS)
        rng = np.random.default_rng(0)
        p0 = sample_walkers_jitter(TRUTHS, EMCEE_NWALKERS, EMCEE_INIT_JITTER, rng)
        pool = Pool(EMCEE_POOL) if EMCEE_POOL else None
        t0 = time.time()
        es = emcee.EnsembleSampler(
            EMCEE_NWALKERS, ndim, log_prob_emcee, pool=pool
        )
        es.run_mcmc(p0, EMCEE_NSTEPS, progress=True)
        elapsed = time.time() - t0
        if pool is not None:
            pool.close()
            pool.join()
        chain = es.get_chain(
            discard=EMCEE_DISCARD, thin=EMCEE_THIN, flat=True
        )
        samp = {k: chain[:, j] for j, k in enumerate(PARAMS)}
        weights = np.ones(chain.shape[0])
        np.savez(samples_path, weights=weights, **samp)
        stats = weighted_stats(samp, weights)
        print(f"  acceptance={float(np.mean(es.acceptance_fraction)):.3f}")
        print(f"  runtime={elapsed:.0f}s")
        save_json(
            {
                "method": "geometry_ratios_lenstronomy_emcee",
                "nwalkers": EMCEE_NWALKERS,
                "nsteps": EMCEE_NSTEPS,
                "discard": EMCEE_DISCARD,
                "thin": EMCEE_THIN,
                "pool": EMCEE_POOL,
                "acceptance_fraction": float(np.mean(es.acceptance_fraction)),
                "runtime_seconds": round(elapsed, 1),
                "stats": stats,
                "priors": BOUNDS,
                "active": ACTIVE,
            },
            meta_path,
        )
        plot_corners(samp, weights, run_label)

    elif SAMPLER == "nessai":
        import logging

        from nessai.flowsampler import FlowSampler
        from nessai.model import Model

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        print("\n--- nessai ---")

        class RatioModel(Model):
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

        nessai_kwargs = dict(
            nlive=NESSAI_NLIVE,
            log_on_iteration=True,
        )
        if not NESSAI_INS:
            nessai_kwargs["stopping"] = NESSAI_STOPPING
        else:
            nessai_kwargs["min_samples"] = max(1, NESSAI_NLIVE // 2)
        if NESSAI_NPOOL is not None:
            import multiprocessing

            multiprocessing.set_start_method("fork", force=True)
            nessai_kwargs["n_pool"] = NESSAI_NPOOL

        t0 = time.time()
        nessai_sampler = FlowSampler(
            RatioModel(),
            output=os.path.join(OUT_DIR, f"nessai_state_{OBS_TAG}"),
            importance_nested_sampler=NESSAI_INS,
            resume=False,
            seed=0,
            **nessai_kwargs,
        )
        print("  nessai running (progress via logging INFO)...")
        nessai_sampler.run()
        elapsed = time.time() - t0
        post = nessai_sampler.posterior_samples
        samp = {k: np.asarray(post[k]) for k in PARAMS}
        weights = np.ones(len(post))
        np.savez(samples_path, weights=weights, **samp)
        stats = weighted_stats(samp, weights)
        print(f"  runtime={elapsed:.0f}s")
        print(
            f"  log_evidence={nessai_sampler.log_evidence:.4f} "
            f"+/- {nessai_sampler.log_evidence_error:.4f}"
        )
        save_json(
            {
                "method": "geometry_ratios_lenstronomy_nessai",
                "nlive": NESSAI_NLIVE,
                "stopping": NESSAI_STOPPING,
                "n_pool": NESSAI_NPOOL,
                "n_samples": len(post),
                "log_evidence": float(nessai_sampler.log_evidence),
                "log_evidence_error": float(nessai_sampler.log_evidence_error),
                "runtime_seconds": round(elapsed, 1),
                "stats": stats,
                "priors": BOUNDS,
                "active": ACTIVE,
            },
            meta_path,
        )
        plot_corners(samp, weights, run_label)

    else:
        raise ValueError(f"unknown SAMPLER: {SAMPLER!r}")

    print(f"\nDone ({SAMPLER}) -> {samples_path}")
    print(f"  meta -> {meta_path}")
