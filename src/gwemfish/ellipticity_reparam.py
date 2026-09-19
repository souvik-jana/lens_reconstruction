"""
Optional (q, phi) reparametrization for lens mass ellipticity, as an alternative to
sampling e1/e2 directly.

Convention (validated against lenstronomy's Util.param_util and matched by herculens's
copy of the same functions): phi in radians from the +x-axis, q = minor/major axis
ratio in (0, 1]::

    e1 = (1-q)/(1+q) * cos(2*phi)
    e2 = (1-q)/(1+q) * sin(2*phi)

herculens's PyAutoLens/PyAutoGalaxy uses the same physics but a different interface
(degrees, swapped/relabeled components) -- irrelevant here since this module never
touches PAL.
"""

import logging

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from herculens.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity

logger = logging.getLogger(__name__)

VALID_PARAMETRIZATIONS = ("e1e2", "q_phi")

# Single source of truth for the default q/phi bounds, shared by default_qphi_priors
# (numpyro, used by hmc/Fisher/deriv-approx) and qphi_scipy_priors (scipy, used by
# nautilus, which never traces the numpyro model) -- keeps the two backends from
# silently drifting onto different priors for the same cfg setting.
Q_LOW, Q_HIGH = 0.01, 1.0
PHI_LOW, PHI_HIGH = -jnp.pi / 2, jnp.pi / 2


def validate_parametrization(parametrization):
    if parametrization not in VALID_PARAMETRIZATIONS:
        raise ValueError(
            f"Unknown lens_mass_parametrization {parametrization!r}; "
            f"expected one of {VALID_PARAMETRIZATIONS}."
        )
    return parametrization


def default_qphi_priors(prefix):
    """Zero-arg numpyro samplers for '{prefix}_q'/'{prefix}_phi'. phi's range matches
    the existing PIEMD/DPIE convention (profile_prior_rules.py: Uniform(-pi/2, pi/2)).
    q's range is wider than PIEMD/DPIE's default Uniform(0.2, 1.0) -- Uniform(0.01, 1.0)
    per explicit user preference, not an attempt to match PIEMD/DPIE's bound exactly."""
    return {
        f"{prefix}_q": lambda: numpyro.sample(f"{prefix}_q", dist.Uniform(Q_LOW, Q_HIGH)),
        f"{prefix}_phi": lambda: numpyro.sample(
            f"{prefix}_phi", dist.Uniform(PHI_LOW, PHI_HIGH)
        ),
    }


def compute_qphi_ellipticity(prefix, p):
    """Sample '{prefix}_q'/'{prefix}_phi' from the priors dict ``p`` (respecting any
    user override, exactly like every other parameter in this codebase), convert to
    e1/e2, and register them as numpyro.deterministic sites so they land in every
    posterior/Fisher output regardless of parametrization."""
    q = p[f"{prefix}_q"]()
    phi = p[f"{prefix}_phi"]()
    e1, e2 = phi_q2_ellipticity(phi, q)
    e1 = numpyro.deterministic(f"{prefix}_e1", e1)
    e2 = numpyro.deterministic(f"{prefix}_e2", e2)
    return e1, e2


def warn_if_ellipticity_prior_keys_unused(user_priors, parametrization, prefixes):
    """One-time warning (not an error) when a user override targets the ellipticity
    parameter pair that isn't active for ``parametrization`` -- e.g. cfg['priors']
    still sets '{prefix}_e1' while parametrization='q_phi'. The override is dead
    (the default prior for the active pair is used instead); this just says so."""
    if not user_priors:
        return
    dead_suffixes = ("q", "phi") if parametrization == "e1e2" else ("e1", "e2")
    for prefix in prefixes:
        for suffix in dead_suffixes:
            key = f"{prefix}_{suffix}"
            if key in user_priors:
                logger.warning(
                    "cfg['priors'][%r] is set but lens_mass_parametrization=%r -- "
                    "'%s' is not sampled in this mode and this override is unused; "
                    "the default prior for the active parametrization is used instead.",
                    key, parametrization, key,
                )


def mass_qphi_prefixes_from_entries(entries, parametrization):
    """Flat-key prefixes (e.g. 'lens0') of mass components using e1/e2, for the
    parameter_layout/flex/nautilus paths -- empty unless parametrization='q_phi'."""
    if parametrization != "q_phi":
        return frozenset()
    return frozenset(
        e.flat_key[: -len("_e1")] for e in entries
        if e.plane == "mass" and e.param == "e1"
    )


def qphi_scipy_priors(prefix):
    """scipy.stats distributions for nautilus's own (non-numpyro-traced) prior --
    same Q_LOW/Q_HIGH/PHI_LOW/PHI_HIGH bounds as default_qphi_priors (scipy's
    ``uniform(loc, scale)`` takes loc=low, scale=high-low, unlike numpyro's
    ``Uniform(low, high)``)."""
    import scipy.stats as sps

    return {
        f"{prefix}_q": sps.uniform(Q_LOW, Q_HIGH - Q_LOW),
        f"{prefix}_phi": sps.uniform(PHI_LOW, PHI_HIGH - PHI_LOW),
    }


def swap_ellipticity_for_qphi(default_dists, prefixes):
    """Mutate a nautilus ``default_dists`` dict in place: drop '{prefix}_e1'/'_e2'
    and add '{prefix}_q'/'_phi' scipy priors. Nautilus never traces the numpyro
    model during sampling, so this is the only way to change what it proposes."""
    for prefix in prefixes:
        default_dists.pop(f"{prefix}_e1", None)
        default_dists.pop(f"{prefix}_e2", None)
        default_dists.update(qphi_scipy_priors(prefix))


def _is_lens_mass_prefix(prefix):
    return prefix == "lens" or (prefix.startswith("lens") and prefix[len("lens"):].isdigit())


def add_qphi_truth(truth_params):
    """Mutate ``truth_params`` in place: for every lens-mass '{prefix}_e1'/'{prefix}_e2'
    pair already present (legacy flat 'lens_e1' or layout 'lens0_e1' naming -- doesn't
    matter which), add the corresponding '{prefix}_q'/'{prefix}_phi'. Needed so the
    Fisher/deriv-approx expansion point (built from truth_params, simple_pipeline.py's
    run_inference) resolves those keys when lens_mass_parametrization='q_phi'.
    Restricted to lens-mass prefixes (not 'source'/'light') since only lens mass
    ellipticity has this reparametrization -- truth_params also carries Sersic
    source_e1/e2, light_e1/e2 pairs that must not get spurious q/phi truth values."""
    prefixes = {
        key[: -len("_e1")] for key in truth_params
        if key.endswith("_e1") and f"{key[:-len('_e1')]}_e2" in truth_params
        and _is_lens_mass_prefix(key[: -len("_e1")])
    }
    for prefix in prefixes:
        phi, q = ellipticity2phi_q(truth_params[f"{prefix}_e1"], truth_params[f"{prefix}_e2"])
        truth_params[f"{prefix}_q"] = float(q)
        truth_params[f"{prefix}_phi"] = float(phi)
    return truth_params


def qphi_derived_keys(prefixes):
    """'{prefix}_e1'/'{prefix}_e2' for every q_phi-active prefix -- these are
    numpyro.deterministic (derived from '{prefix}_q'/'{prefix}_phi'), not free
    parameters. ``probmodel.get_sample()`` (herculens's NumpyroModel) filters only
    on ``is_observed``, not site type, so it returns deterministic sites too --
    callers building a Fisher/deriv-approx 'keys_to_include' list or a nautilus
    parameter list from ``get_sample()``'s keys must exclude this set, or they end
    up sampling '{prefix}_e1'/'{prefix}_e2' as if they were independent of
    '{prefix}_q'/'{prefix}_phi' (from priors.py's unrelated default e1/e2 prior),
    corrupting the result with spurious extra dimensions."""
    keys = set()
    for prefix in prefixes:
        keys.add(f"{prefix}_e1")
        keys.add(f"{prefix}_e2")
    return keys


def add_qphi_columns_to_samples(samples):
    """Post-hoc: for every lens-mass '{prefix}_q'/'{prefix}_phi' pair in a nautilus
    samples dict (arrays, one row per posterior draw), add '{prefix}_e1'/'{prefix}_e2'
    arrays too. Needed only for nautilus: unlike numpyro MCMC (where lens_e1/lens_e2
    are numpyro.deterministic sites and land in get_samples() automatically),
    nautilus never traces the model, so its q/phi samples have no e1/e2 counterpart
    unless added here -- keeps corner plots / to_source_plane_samples parametrization-
    agnostic regardless of sampler. Restricted to lens-mass prefixes: PIEMD/DPIE mass
    profiles already sample q/phi natively (unrelated to this reparametrization) and
    have no e1/e2 kwarg at all -- without this filter, their genuine q/phi samples
    would get bogus e1/e2 columns fabricated for them."""
    import numpy as np

    prefixes = {
        key[: -len("_q")] for key in samples
        if key.endswith("_q") and f"{key[:-len('_q')]}_phi" in samples
        and _is_lens_mass_prefix(key[: -len("_q")])
    }
    for prefix in prefixes:
        e1, e2 = phi_q2_ellipticity(
            np.asarray(samples[f"{prefix}_phi"]), np.asarray(samples[f"{prefix}_q"])
        )
        samples[f"{prefix}_e1"] = np.asarray(e1)
        samples[f"{prefix}_e2"] = np.asarray(e2)
    return samples


def expand_qphi_params(full, prefixes):
    """Mutate a flat params dict in place: for every prefix with '{prefix}_q'/'_phi'
    (nautilus's sampled values), fill in '{prefix}_e1'/'_e2' too, so downstream
    kwargs-building code (which only knows e1/e2) is unaffected. Mutates rather than
    copies -- every call site passes a dict freshly built one line earlier
    (`{**fixed_params, **params}`) inside nautilus's per-evaluation log_likelihood
    closure, so an extra copy there is pure allocation overhead with nothing to
    protect against aliasing."""
    if not prefixes:
        return full
    for prefix in prefixes:
        qk, phik = f"{prefix}_q", f"{prefix}_phi"
        if qk in full and phik in full:
            e1, e2 = phi_q2_ellipticity(full[phik], full[qk])
            full[f"{prefix}_e1"] = float(e1)
            full[f"{prefix}_e2"] = float(e2)
    return full
