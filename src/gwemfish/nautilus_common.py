"""
Shared Nautilus helpers: scipy priors, EM-only problem builder, sampler runner.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import scipy.stats as sps

from .ellipticity_reparam import (
    expand_qphi_params,
    mass_qphi_prefixes_from_entries,
    swap_ellipticity_for_qphi,
    validate_parametrization,
    warn_if_ellipticity_prior_keys_unused,
)
from .priors import DEFAULT_PRIORS_GW_SOURCE_PLANE


EM_EXTRA_DEFAULT_DISTS = {
    "source_amp":      lambda b: sps.loguniform(1e-6, 1e6),
    "source_R_sersic": lambda b: sps.uniform(0.0, 30.0),
    "source_n":        lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "source_e1":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "source_e2":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_amp":       lambda b: sps.loguniform(1e-6, 1e6),
    "light_R_sersic":  lambda b: sps.uniform(0.0, 30.0),
    "light_n":         lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "light_e1":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_e2":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_center_x":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "light_center_y":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "noise_sigma_bkg": lambda b: sps.loguniform(1e-6, 1e6),
}


def _tnorm(lo, hi, loc=0.0, scale=0.3):
    a, b = (lo - loc) / scale, (hi - loc) / scale
    return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)


def _numpyro_dist_to_scipy(d):
    try:
        type_name = type(d).__name__

        if type_name == "Uniform":
            lo, hi = float(d.low), float(d.high)
            return sps.uniform(lo, hi - lo)

        if type_name == "Normal":
            return sps.norm(loc=float(d.loc), scale=float(d.scale))

        if type_name in ("TwoSidedTruncatedDistribution",
                         "LeftTruncatedDistribution",
                         "RightTruncatedDistribution"):
            lo = float(d.low) if hasattr(d, "low") and d.low is not None else -np.inf
            hi = float(d.high) if hasattr(d, "high") and d.high is not None else np.inf
            base = d.base_dist
            loc, scale = float(base.loc), float(base.scale)
            a = (lo - loc) / scale
            b = (hi - loc) / scale
            return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)

        if type_name == "LogUniform":
            lo, hi = float(d.low), float(d.high)
            return sps.loguniform(lo, hi)

        warnings.warn(
            f"Unsupported numpyro distribution type '{type_name}' for scipy conversion; "
            "using default prior."
        )
    except Exception as exc:
        warnings.warn(f"Error converting numpyro distribution to scipy: {exc}; using default prior.")
    return None


def _extract_dist_from_callable(callable_prior):
    try:
        import jax
        import numpyro
        seeded = numpyro.handlers.seed(callable_prior, jax.random.PRNGKey(0))
        tr = numpyro.handlers.trace(seeded).get_trace()
        if not tr:
            return None
        site = list(tr.values())[0]
        return site.get("fn")
    except Exception:
        return None


def _numpyro_to_spec(d):
    type_name = type(d).__name__
    if type_name == "Delta":
        return "fixed", float(np.asarray(d.v).reshape(-1)[0])
    try:
        if int(np.prod(d.batch_shape + d.event_shape)) > 1:
            return "skip", None
    except Exception:
        pass
    scipy_dist = _numpyro_dist_to_scipy(d)
    if scipy_dist is None:
        return "skip", None
    return "dist", scipy_dist


def layout_defaults_from_registry(entries, registry):
    default_dists = {}
    registry_fixed = {}
    for e in entries:
        d = _extract_dist_from_callable(registry[e.flat_key])
        if d is None:
            warnings.warn(f"Could not extract default prior for '{e.flat_key}'; skipping.")
            continue
        kind, val = _numpyro_to_spec(d)
        if kind == "fixed":
            registry_fixed[e.flat_key] = val
        elif kind == "dist":
            default_dists[e.flat_key] = val
        else:
            warnings.warn(
                f"Default prior for '{e.flat_key}' is array-valued or unsupported; "
                "skipping (fix it via cfg['priors'] if needed)."
            )
    return default_dists, registry_fixed


def parse_cfg_priors(cfg_priors, default_dists, bounds):
    import numpyro.distributions as npdist

    scipy_overrides = {}
    fixed_params = {}

    for name, value in (cfg_priors or {}).items():
        if name not in default_dists:
            continue

        if hasattr(value, "rvs") and hasattr(value, "ppf"):
            scipy_overrides[name] = value
            continue

        if isinstance(value, npdist.Distribution):
            scipy_dist = _numpyro_dist_to_scipy(value)
            if scipy_dist is not None:
                scipy_overrides[name] = scipy_dist
            continue

        if callable(value):
            npdist_extracted = _extract_dist_from_callable(value)
            if npdist_extracted is not None:
                scipy_dist = _numpyro_dist_to_scipy(npdist_extracted)
                if scipy_dist is not None:
                    scipy_overrides[name] = scipy_dist
                    continue
            warnings.warn(
                f"Could not convert callable prior for '{name}' to scipy distribution; "
                "using default prior."
            )
            continue

        try:
            fixed_params[name] = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            warnings.warn(f"Cannot interpret prior value for '{name}'; using default prior.")

    return scipy_overrides, fixed_params


def build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params):
    import nautilus

    prior = nautilus.Prior()
    for name, dist_spec in default_dists.items():
        if name in fixed_params:
            continue
        if name in scipy_overrides:
            dist = scipy_overrides[name]
        elif hasattr(dist_spec, "rvs"):
            dist = dist_spec
        else:
            dist = dist_spec(bounds.get(name, (0.0, 1.0)))
        prior.add_parameter(name, dist=dist)
    return prior


def probmodel_log_likelihood(probmodel, params, rng_key=0):
    """HMC-equivalent likelihood: joint log density minus prior sample sites."""
    import jax
    from numpyro.handlers import seed, substitute, trace
    from numpyro.infer.util import log_density

    seeded = seed(probmodel.model, jax.random.PRNGKey(int(rng_key)))
    log_joint, _ = log_density(seeded, (), {}, params)
    tr = trace(substitute(seeded, data=params)).get_trace()
    log_prior = 0.0
    for site_name, site in tr.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            log_prior += jnp.sum(site["fn"].log_prob(site["value"]))
    return float(log_joint - log_prior)


def build_em_only_nautilus_problem(ctx, cfg):
    """EM-only Nautilus problem: flex-layout scipy priors + pixel Gaussian likelihood."""
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    if not bool(cfg_full.get("use_parameter_layout")):
        raise ValueError(
            "build_em_only_nautilus_problem requires use_parameter_layout=True "
            "(flex lens0_*/source0_*/light0_* names)."
        )

    lens_image = ctx["lens_image"]
    noise = ctx["noise_inf"]
    em_obs = ctx["em_obs"]
    truth_params = ctx.get("truth_params", {})
    em_data = jnp.array(em_obs["data"])

    from .nautilus_jit_cores import check_jittable, em_loglike_core

    jit = bool((cfg_full.get("nautilus") or {}).get("jit", True))
    if jit:
        check_jittable(cfg_full)
        em_compiled = em_loglike_core(lens_image, noise, em_obs["data"])

    from .parameter_layout import (
        build_parameter_layout, build_priors_registry, unpack_to_kwargs,
    )
    from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

    em_sec = cfg_full.get("em") or {}
    ks_tmpl = em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
    kll_tmpl = em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=ks_tmpl,
        kwargs_lens_light=kll_tmpl,
    )
    registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
    default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
    default_dists["noise_sigma_bkg"] = EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None)

    n_mass = len(lens_image.MassModel.func_list)
    n_source = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)

    parametrization = validate_parametrization(cfg_full.get("lens_mass_parametrization", "e1e2"))
    qphi_prefixes = mass_qphi_prefixes_from_entries(entries, parametrization)
    if qphi_prefixes:
        swap_ellipticity_for_qphi(default_dists, qphi_prefixes)

    bounds = DEFAULT_PRIORS_GW_SOURCE_PLANE
    cfg_priors = cfg_full.get("priors", {})
    warn_if_ellipticity_prior_keys_unused(
        cfg_priors, parametrization,
        qphi_prefixes or mass_qphi_prefixes_from_entries(entries, "q_phi"),
    )
    scipy_overrides, cfg_fixed = parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}
        full = expand_qphi_params(full, qphi_prefixes)
        kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
            full, entries, n_mass=n_mass,
            n_source=n_source, n_lens_light=n_lens_light,
        )
        if jit:
            return float(em_compiled(kwargs_lens, kwargs_source, kwargs_lens_light,
                                     full["noise_sigma_bkg"]))

        sigma_bkg = float(full["noise_sigma_bkg"])
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        return float(
            jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                             + jnp.log(2 * jnp.pi * model_var)))
        )

    print(f"Warming up EM-only log_likelihood (jit={jit}; compiles once)...")
    try:
        lv = log_likelihood(dict(truth_params))
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def build_nautilus_problem(ctx, cfg, mode, method):
    """(prior, log_likelihood, param_names) for any nautilus method and mode.

    One dispatch, used both by the pipeline in the parent process and by each
    pool worker, so a worker cannot build a different problem than the run it
    joined.
    """
    if method == "nautilus-image":
        from .nautilus_image_inference import build_image_plane_problem

        return build_image_plane_problem(ctx, mode, cfg)

    from .nautilus_source_inference import (
        build_em_gw_source_plane_problem,
        build_gw_source_plane_problem,
    )

    if mode == "GW-only":
        return build_gw_source_plane_problem(ctx, cfg)
    if mode == "EM+GW":
        return build_em_gw_source_plane_problem(ctx, cfg)
    if mode == "EM-only":
        return build_em_only_nautilus_problem(ctx, cfg)
    raise ValueError(
        f"{method} supports mode 'GW-only', 'EM+GW', and 'EM-only' only, got {mode!r}"
    )


# Written into ctx by a previous fisher / deriv-approx run and not picklable: both
# hold jitted closures (simple_pipeline.py:2097-2098 and :2114). Nautilus reads
# neither, so a pool run strips them from its copy rather than failing at pool
# startup, which is a place nobody would connect to the cause.
UNPICKLABLE_CTX_KEYS = ("fisher", "likelihood")

def sanitize_priors_for_pool(priors):
    """Make cfg['priors'] shippable, or say exactly which entry is not.

    ``parse_cfg_priors`` accepts callables, and a lambda cannot cross a process
    boundary. It can be replaced by the distribution it produces, which is what
    the eager path would have extracted anyway, so a callable prior costs a pool
    run nothing.
    """
    import pickle

    if not priors:
        return priors

    clean = {}
    unshippable = []
    for name, value in priors.items():
        try:
            pickle.dumps(value)
            clean[name] = value
            continue
        except Exception:
            pass

        dist = _extract_dist_from_callable(value) if callable(value) else None
        if dist is None:
            unshippable.append(name)
            continue
        try:
            pickle.dumps(dist)
            clean[name] = dist
        except Exception:
            unshippable.append(name)

    if unshippable:
        raise TypeError(
            f"cannot use cfg['nautilus']['pool']: these priors cannot be sent to a "
            f"worker process and are not convertible to a distribution: "
            f"{unshippable}. Use a numpyro distribution, a scipy distribution or a "
            "fixed float for those keys, or run without a pool."
        )
    return clean


def prepare_for_pool(ctx, cfg):
    """Copies of ctx and cfg that survive the trip to a worker.

    The caller's ctx is never mutated: a later method in the same script still
    needs ctx['fisher'].
    """
    ctx_clean = {k: v for k, v in ctx.items() if k not in UNPICKLABLE_CTX_KEYS}
    if isinstance(ctx_clean.get("cfg"), dict):
        ctx_cfg = dict(ctx_clean["cfg"])
        ctx_cfg["priors"] = sanitize_priors_for_pool(ctx_cfg.get("priors"))
        ctx_clean["cfg"] = ctx_cfg

    cfg_clean = dict(cfg)
    cfg_clean["priors"] = sanitize_priors_for_pool(cfg_clean.get("priors"))
    return ctx_clean, cfg_clean


def init_pool_worker():
    """Keep one worker to one core's worth of linear algebra.

    Four workers each starting as many BLAS threads as there are cores turns a
    speed-up into a slowdown. Nautilus applies this around its own neural-network
    sections only, not around likelihood calls.
    """
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


class PicklableLikelihood:
    """Ships the simulated system to a worker, which rebuilds the closure once.

    The likelihood itself cannot be sent: every builder returns a function defined
    inside another function, and pickle resolves functions by name.

        AttributeError: Can't get local object
                        'build_gw_source_plane_problem.<locals>.log_likelihood'

    The whole ctx, by contrast, is ~56 KB of plain data and ships instantly, so
    each worker gets the identical simulated system -- same noise realisation,
    same observed data -- and rebuilds only the cheap part.
    """

    def __init__(self, ctx, cfg, mode, method):
        self.ctx, self.cfg = prepare_for_pool(ctx, cfg)
        self.mode = mode
        self.method = method
        self.log_likelihood = None

    def __getstate__(self):
        # The built closure is exactly the thing that cannot be pickled; each
        # worker makes its own.
        return {**self.__dict__, "log_likelihood": None}

    def __call__(self, params):
        if self.log_likelihood is None:
            import os
            import time

            # Nautilus unpickles this object once per worker and keeps it
            # (nautilus/pool.py:59-61 passes it as the pool initializer), so the
            # rebuild below happens once per worker, not once per point.
            init_pool_worker()
            t0 = time.perf_counter()
            _, self.log_likelihood, _ = build_nautilus_problem(
                self.ctx, self.cfg, self.mode, self.method)
            print(f"    [pid {os.getpid()}] built {self.method} {self.mode} "
                  f"likelihood in {time.perf_counter() - t0:.1f}s", flush=True)
        return self.log_likelihood(params)


def check_pool_callable_from_here():
    """Catch an unguarded caller script before multiprocessing's wall of text.

    Workers are spawned, so each one re-imports the script that started the run.
    If that script does its work at module level, the worker re-runs it, reaches
    the same pooled ``run_inference`` call, and tries to create a pool of its own.
    Python then raises ``RuntimeError: An attempt has been made to start a new
    process before the current process has finished its bootstrapping phase``
    from deep inside ``multiprocessing.spawn``, once per worker, which says
    nothing about gwemfish.

    Reaching this function from anywhere but the main process means exactly that
    has happened.
    """
    import multiprocessing as mp

    if mp.parent_process() is None:
        return
    raise RuntimeError(
        "cfg['nautilus']['pool'] was reached inside a worker process, which means "
        "the script that started this run does its work at module level. Pool "
        "workers re-import that script, so it runs again in each of them and each "
        "tries to start its own pool.\n"
        "Fix: put the script's body under a guard, leaving imports and function "
        "definitions above it:\n\n"
        "    if __name__ == '__main__':\n"
        "        ctx = setup_em_observation(cfg=CFG)\n"
        "        ...\n"
        "        run_inference(ctx, mode=..., method='nautilus-source', cfg=...)\n\n"
        "Runs without a pool do not need the guard."
    )


def force_spawn_start_method():
    """Workers must be fresh processes, not clones of this one.

    ``nautilus/pool.py:3`` imports ``Pool`` plainly, so it takes the platform
    default -- ``fork`` on Linux. Forking a process whose JAX threads are running
    leaves the child holding locks owned by threads that do not exist in it, and
    it hangs with no error (observed). macOS already defaults to spawn.

    This is a process-wide setting, so it is flipped only when a pool is actually
    requested, and it is announced.
    """
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) == "spawn":
        return
    mp.set_start_method("spawn", force=True)
    print("  nautilus pool: multiprocessing start method set to 'spawn' "
          "(process-wide). Workers re-import your script, so its body must sit "
          "under `if __name__ == \"__main__\":`.")


def _prior_fingerprint(prior):
    """JSON-serializable fingerprint of a nautilus.Prior: per-parameter ppf quantiles."""
    qs = (0.001, 0.25, 0.5, 0.75, 0.999)
    params = {}
    for name, dist in zip(prior.keys, prior.dists):
        try:
            params[str(name)] = [float(v) for v in np.atleast_1d(dist.ppf(qs))]
        except Exception:
            params[str(name)] = None
    return {"quantiles": list(qs), "params": params}


def _check_checkpoint_priors(prior, filepath, resume):
    """Guard against resuming a nautilus checkpoint under different priors.

    Nautilus stores sample points in the unit cube and maps them through the
    *current* prior at ``posterior()`` time, so resuming a checkpoint whose
    priors differ from the current ones silently returns a wrong posterior
    (stored points get stretched onto the new intervals). A fingerprint
    sidecar (``<filepath>.priors.json``) is written on every run and compared
    here on resume; on mismatch a ``ValueError`` is raised.
    """
    import json
    import os

    sidecar = str(filepath) + ".priors.json"
    fp_new = _prior_fingerprint(prior)

    if resume and os.path.exists(filepath):
        if not os.path.exists(sidecar):
            warnings.warn(
                f"Resuming nautilus checkpoint '{filepath}' without a prior "
                f"fingerprint sidecar ('{sidecar}'); cannot verify that the current "
                "priors match the ones the checkpoint was built with. If they "
                "differ, the returned posterior will be silently wrong."
            )
        else:
            with open(sidecar) as f:
                fp_old = json.load(f)
            old_params = fp_old.get("params", {})
            mismatches = []
            if list(old_params) != list(fp_new["params"]):
                mismatches.append(
                    f"parameter names/order: checkpoint {list(old_params)} "
                    f"vs current {list(fp_new['params'])}"
                )
            elif fp_old.get("quantiles") == fp_new["quantiles"]:
                for name, old in old_params.items():
                    new = fp_new["params"][name]
                    if old is None or new is None:
                        continue
                    if not np.allclose(old, new, rtol=1e-6, atol=1e-12):
                        mismatches.append(
                            f"{name}: checkpoint quantiles {old} vs current {new}"
                        )
            if mismatches:
                raise ValueError(
                    "resume=True but the current priors differ from the ones the "
                    f"nautilus checkpoint '{filepath}' was built with. Nautilus stores "
                    "unit-cube points and maps them through the CURRENT prior, so "
                    "resuming would silently return a wrong posterior. Rebuild the "
                    "original priors, point 'filepath' at a fresh checkpoint, or set "
                    "resume=False (starts a new run). Mismatched priors:\n  "
                    + "\n  ".join(mismatches)
                )
    return fp_new, sidecar


def run_nautilus(prior, log_likelihood, *,
                 n_live=500, filepath=None, verbose=True,
                 resume=True, prior_check=True, pool=None, seed=None,
                 equal_weight=False, equal_weight_boost=1.0,
                 return_diagnostics=False, run_kwargs=None):
    """Run the sampler and return the posterior samples.

    By default returns the full unequal-weighted dead-point set plus a
    ``weights`` array (normalized), matching standalone nautilus / the
    geometry_ratios lenstronomy absolute script. Set ``equal_weight=True`` to
    resample to i.i.d. draws with no ``weights`` key (nautilus default boost=1
    keeps only ~1/max(w) points — usually too few for multimodal corners).

    ``pool`` is an int: nautilus then creates the pool itself with
    ``initializer=`` (``nautilus/pool.py:59``) and pickles the likelihood once per
    worker instead of once per batch. Note that a seed reproduces a run only at a
    *fixed* ``pool`` value -- nautilus dispatches points in different batches when
    pooled, so the same seed with and without a pool explores differently.
    """
    import json

    import numpy as np
    import nautilus

    fp_new = sidecar = None
    if (filepath is not None and prior_check
            and hasattr(prior, "keys") and hasattr(prior, "dists")):
        fp_new, sidecar = _check_checkpoint_priors(prior, filepath, resume)

    if pool:
        check_pool_callable_from_here()
        force_spawn_start_method()

    sampler = nautilus.Sampler(
        prior,
        log_likelihood,
        n_live=n_live,
        filepath=filepath,
        resume=resume,
        pool=pool,
        seed=seed,
    )
    if sidecar is not None:
        try:
            with open(sidecar, "w") as f:
                json.dump(fp_new, f, indent=1)
        except OSError as exc:
            warnings.warn(f"Could not write prior fingerprint sidecar '{sidecar}': {exc}")
    sampler.run(verbose=verbose, **(run_kwargs or {}))

    points, log_w, _ = sampler.posterior(
        equal_weight=bool(equal_weight),
        equal_weight_boost=float(equal_weight_boost),
    )
    param_names = list(prior.keys)
    samples_dict = {name: np.array(points[:, j])
                    for j, name in enumerate(param_names)}
    if not equal_weight:
        w = np.exp(np.asarray(log_w) - np.max(log_w))
        samples_dict["weights"] = w / w.sum()

    # Nautilus' own convergence diagnostics, taken from the sampler rather than
    # recomputed: evidence, effective sample size and the likelihood-call count
    # are what tell you whether the run converged or merely stopped.
    diagnostics = {
        "log_z": float(sampler.log_z),
        "n_eff": float(sampler.n_eff),
        "n_like": int(sampler.n_like),
        "n_posterior_samples": int(points.shape[0]),
        "n_live": int(n_live),
        "pool": pool,
        "seed": seed,
        "equal_weight": bool(equal_weight),
    }
    print(f"  nautilus: log_z {diagnostics['log_z']:.4f}  "
          f"n_eff {diagnostics['n_eff']:.1f}  n_like {diagnostics['n_like']}  "
          f"samples {diagnostics['n_posterior_samples']}"
          + ("" if equal_weight else "  (weighted)"))

    if return_diagnostics:
        return samples_dict, diagnostics
    return samples_dict
