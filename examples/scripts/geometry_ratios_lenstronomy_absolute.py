"""geometry_ratios_lenstronomy_absolute: absolute TD + dL_eff with free T_star/dL.

Counterpart to geometry_ratios_gwemfish_absolute. Same truth system as
geometry_ratios_lenstronomy / tutorial_gw_only_pool (EPL+SHEAR, q=0.5,
phi=30 deg, source ~centre). No gwemfish imports.

Observables (phi-sorted, consecutive diffs — same convention as gwemfish):

    td_ij = T_star * diff(phi)
    dL_eff_i = dL / sqrt(|mu_i|)

Cosmology truths from astropy FlatLambdaCDM(H0=67.3, Om0=0.316), zl=0.5,
zs=2.0, using the gwemfish T_star definition (arcsec^2 factor).

Sampler switch: nautilus | emcee | nessai (each with optional pool).

    uv run python examples/scripts/geometry_ratios_lenstronomy_absolute.py
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from astropy.constants import c as C_LIGHT
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from nautilus import Prior, Sampler
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util.param_util import phi_q2_ellipticity

if "science" in plt.style.available:
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

SAMPLER = "nautilus"  # "nautilus" | "emcee" | "nessai"
RESUME = False

NAUTILUS_NLIVE = 5000
NAUTILUS_NEFF = 5000
NAUTILUS_POOL = 9

EMCEE_NWALKERS = 64
EMCEE_NSTEPS = 10000
EMCEE_DISCARD = 4000
EMCEE_THIN = 5
EMCEE_POOL = 10
EMCEE_INIT_JITTER = 0.02

NESSAI_NLIVE = 5000 #9000
NESSAI_STOPPING = 0.2
NESSAI_NPOOL = 10
NESSAI_INS = False

# ===================== TRUTH (tutorial / geometry_ratios) =====================
THETA_E = 2.0
GAMMA_TRUTH = 2.0
LENS_Q = 0.5
LENS_PHI_DEG = 30.0
CENTER_X = 0.0
CENTER_Y = 0.0
SHEAR_FIXED = {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0}
Y0_TRUTH = 0.02
Y1_TRUTH = 0.00001
N_IMAGES = 4

ZL = 0.5
ZS = 2.0
H0 = 67.3
OM0 = 0.316

SIGMA_TD = 0.001
SIGMA_DL_EFF = 0.1

SOLVER = dict(
    min_distance=0.05,
    search_window=5.0,
    precision_limit=1e-10,
    num_iter_max=200,
)

OBS_TD = ["td21", "td32", "td43"]
OBS_DLEFF = ["dL_eff1", "dL_eff2", "dL_eff3", "dL_eff4"]
OBS_NAMES = OBS_TD + OBS_DLEFF
PARAMS = ["q", "gamma", "T_star", "dL", "y0gw", "y1gw"]
CONTOUR_LEVELS = [0.95]
CORNER_QUANTILES = [0.16, 0.5, 0.84]

OUT_DIR = os.path.join(
    REPO_ROOT, "examples", "outputs", "geometry_ratios_lenstronomy_absolute", SAMPLER
)
os.makedirs(OUT_DIR, exist_ok=True)

# gwemfish LensImageGW constants
ARCSEC_TO_RAD = 4.84813681109536e-06
MPC_TO_M = 3.085677581491367e22


def cosmology_truths(zl, zs, h0, om0):
    cosmo = FlatLambdaCDM(H0=h0, Om0=om0)
    dd = cosmo.angular_diameter_distance(zl).to(u.Mpc).value
    ds = cosmo.angular_diameter_distance(zs).to(u.Mpc).value
    dds = cosmo.angular_diameter_distance_z1z2(zl, zs).to(u.Mpc).value
    d_dt = (1.0 + zl) * dd * ds / dds
    t_star = (d_dt * MPC_TO_M / C_LIGHT.value) * ARCSEC_TO_RAD ** 2
    d_l = cosmo.luminosity_distance(zs).to(u.Mpc).value
    return float(t_star), float(d_l), float(d_dt)


T_STAR_TRUTH, DL_TRUTH, D_DT_TRUTH = cosmology_truths(ZL, ZS, H0, OM0)

# Prior boxes: geometry match geometry_ratios; T_star/dL match gwemfish absolute
# BOUNDS = {
#     "q": (0.3, 0.75),
#     "gamma": (1.1, 3.0),
#     "T_star": (1e6, 1.88e7),
#     "dL": (10000.0, 29000.0),
#     "y0gw": (-0.001, 0.05),
#     "y1gw": (-0.008, 0.009),
# }

BOUNDS = {
    "q": (0.3, 0.75),
    "gamma": (1.2, 2.8),
    "T_star": (4e6, 1.6e7),
    "dL": (10000.0, 28000.0),
    "y0gw": (-0.001, 0.05),
    "y1gw": (-0.008, 0.028),
}

PHI_RAD = float(np.deg2rad(LENS_PHI_DEG))
LENS = LensModel(lens_model_list=["EPL", "SHEAR"])
SOLVER_ENGINE = LensEquationSolver(LENS)


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
    x, y = SOLVER_ENGINE.image_position_from_source(y0s, y1s, kw, **SOLVER)
    if len(x) != N_IMAGES:
        return None
    phi = np.asarray(LENS.fermat_potential(x, y, kw, y0s, y1s))
    mu = np.asarray(LENS.magnification(x, y, kw))
    idx = np.argsort(phi)
    return x[idx], y[idx], phi[idx], mu[idx]


def observables(q, gamma, t_star, d_l, y0, y1):
    sol = sorted_images(q, gamma, y0, y1)
    if sol is None:
        return None
    _, _, phi, mu = sol
    td = as_float(t_star) * np.diff(phi)
    dleff = as_float(d_l) / np.sqrt(np.abs(mu))
    return {
        "td21": float(td[0]),
        "td32": float(td[1]),
        "td43": float(td[2]),
        "dL_eff1": float(dleff[0]),
        "dL_eff2": float(dleff[1]),
        "dL_eff3": float(dleff[2]),
        "dL_eff4": float(dleff[3]),
        "mu": [float(m) for m in mu],
        "phi": [float(p) for p in phi],
    }


TRUTHS = {
    "q": LENS_Q,
    "gamma": GAMMA_TRUTH,
    "T_star": T_STAR_TRUTH,
    "dL": DL_TRUTH,
    "y0gw": Y0_TRUTH,
    "y1gw": Y1_TRUTH,
}

OBS = observables(
    LENS_Q, GAMMA_TRUTH, T_STAR_TRUTH, DL_TRUTH, Y0_TRUTH, Y1_TRUTH
)
if OBS is None:
    raise SystemExit(
        f"truth source ({Y0_TRUTH}, {Y1_TRUTH}) is not a {N_IMAGES}-image system"
    )

SIGMA = {
    "td21": SIGMA_TD * abs(OBS["td21"]),
    "td32": SIGMA_TD * abs(OBS["td32"]),
    "td43": SIGMA_TD * abs(OBS["td43"]),
    "dL_eff1": SIGMA_DL_EFF * abs(OBS["dL_eff1"]),
    "dL_eff2": SIGMA_DL_EFF * abs(OBS["dL_eff2"]),
    "dL_eff3": SIGMA_DL_EFF * abs(OBS["dL_eff3"]),
    "dL_eff4": SIGMA_DL_EFF * abs(OBS["dL_eff4"]),
}

USE = {
    "td21": USE_TD21,
    "td32": USE_TD32,
    "td43": USE_TD43,
    "dL_eff1": USE_DLEFF1,
    "dL_eff2": USE_DLEFF2,
    "dL_eff3": USE_DLEFF3,
    "dL_eff4": USE_DLEFF4,
}
ACTIVE = [k for k in OBS_NAMES if USE[k]]
OBS_TAG = "full" if len(ACTIVE) == len(OBS_NAMES) else "_".join(ACTIVE)


def loglike(q, gamma, t_star, d_l, y0, y1):
    m = observables(q, gamma, t_star, d_l, y0, y1)
    if m is None:
        return -1e300
    chi2 = sum(((m[k] - OBS[k]) / SIGMA[k]) ** 2 for k in ACTIVE)
    return -0.5 * chi2


def loglike_dict(d):
    return loglike(
        d["q"], d["gamma"], d["T_star"], d["dL"], d["y0gw"], d["y1gw"]
    )


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
    arr = np.column_stack([samp[k] for k in PARAMS])
    truths = [TRUTHS[k] for k in PARAMS]
    fig = corner.corner(
        arr,
        weights=weights,
        labels=PARAMS,
        truths=truths,
        quantiles=CORNER_QUANTILES,
        show_titles=True,
        title_fmt=".4f",
        hist_kwargs={"density": True},
    )
    fig.savefig(os.path.join(OUT_DIR, f"corner_{tag}.png"), dpi=150)
    plt.close(fig)
    fig = corner.corner(
        arr,
        weights=weights,
        labels=PARAMS,
        truths=truths,
        quantiles=CORNER_QUANTILES,
        show_titles=True,
        title_fmt=".4f",
        levels=CONTOUR_LEVELS,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        hist_kwargs={"density": True},
    )
    fig.savefig(os.path.join(OUT_DIR, f"corner_contours_{tag}.png"), dpi=150)
    plt.close(fig)


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
    print("  geometry_ratios_lenstronomy_absolute (TD + dL_eff, free T_star/dL)")
    print(
        f"  truth: theta_E={THETA_E} gamma={GAMMA_TRUTH} q={LENS_Q} "
        f"phi={LENS_PHI_DEG} deg  source=({Y0_TRUTH}, {Y1_TRUTH})"
    )
    print(f"  cosmology: H0={H0} Om0={OM0} zl={ZL} zs={ZS}")
    print(
        f"  T_star={T_STAR_TRUTH:.6g} s  dL={DL_TRUTH:.6g} Mpc  "
        f"D_dt={D_DT_TRUTH:.6g} Mpc"
    )
    print(f"  free={PARAMS}  active={ACTIVE}  sampler={SAMPLER}")
    print(f"  SIGMA_TD={SIGMA_TD}  SIGMA_DL_EFF={SIGMA_DL_EFF}")
    print("  simulated observables:")
    for k in OBS_NAMES:
        print(f"    {k:>8}: {OBS[k]:+.6g}  (sigma={SIGMA[k]:.6g})")
    print("  priors [left, truth, right]:")
    for key in PARAMS:
        lo, hi = BOUNDS[key]
        print(f"    {key:>7}: [{lo:.6g}, {TRUTHS[key]:.6g}, {hi:.6g}]")
    print("=" * 72)

    save_json(
        {
            "observed": {k: OBS[k] for k in OBS_NAMES},
            "mu": OBS["mu"],
            "phi": OBS["phi"],
            "sigma": SIGMA,
            "active": ACTIVE,
            "sigma_td": SIGMA_TD,
            "sigma_dL_eff": SIGMA_DL_EFF,
            "truths": TRUTHS,
            "bounds": BOUNDS,
            "theta_E": THETA_E,
            "phi_deg": LENS_PHI_DEG,
            "cosmology": {
                "H0": H0,
                "Om0": OM0,
                "zl": ZL,
                "zs": ZS,
                "D_dt_Mpc": D_DT_TRUTH,
            },
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
                "method": "geometry_ratios_lenstronomy_absolute_nautilus",
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
                "method": "geometry_ratios_lenstronomy_absolute_emcee",
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

        class AbsoluteModel(Model):
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
                    x_out[n] = (x[n] - self.bounds[n][0]) / np.ptp(
                        self.bounds[n]
                    )
                return x_out

            def from_unit_hypercube(self, x):
                x_out = x.copy()
                for n in self.names:
                    x_out[n] = (
                        np.ptp(self.bounds[n]) * x[n] + self.bounds[n][0]
                    )
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
            AbsoluteModel(),
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
                "method": "geometry_ratios_lenstronomy_absolute_nessai",
                "nlive": NESSAI_NLIVE,
                "stopping": NESSAI_STOPPING,
                "n_pool": NESSAI_NPOOL,
                "n_samples": len(post),
                "log_evidence": float(nessai_sampler.log_evidence),
                "log_evidence_error": float(
                    nessai_sampler.log_evidence_error
                ),
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
