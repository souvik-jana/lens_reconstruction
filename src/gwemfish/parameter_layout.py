"""
Flat parameter names (lens0_*, source0_*, light0_*) aligned with Herculens LensImage models.

Builds ordered keys, default priors from ``profile_prior_rules``, unpacks to kwargs lists
for ``lens_image.model(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .ellipticity_reparam import default_qphi_priors, warn_if_ellipticity_prior_keys_unused
from .profile_prior_rules import required_default_sampler


@dataclass(frozen=True)
class ParamEntry:
    """One inferenced parameter: flat numpyro site name + Herculens kwargs field."""

    flat_key: str
    profile: object
    param: str
    plane: str  # "mass" | "source" | "lens_light"
    component_index: int


def _param_names_for_profile(
    profile: object,
    kwargs_fallback: Optional[Dict[str, Any]],
) -> List[str]:
    cls = type(profile)
    names = getattr(cls, "param_names", None)
    if names is not None:
        return list(names)
    if kwargs_fallback is not None:
        return list(kwargs_fallback.keys())
    raise ValueError(
        f"Profile {cls.__name__} has no param_names; pass kwargs template for that component."
    )


def make_infer_array_shape(lens_image) -> Callable[[object, str], Optional[tuple]]:
    """Infer array shapes for pixelated / shapelet parameters using LensImage grids."""

    mass = lens_image.MassModel
    src = lens_image.SourceModel
    ll = lens_image.LensLightModel

    def infer_array_shape(profile: object, param: str) -> Optional[tuple]:
        if param in ("pixels", "func_pixels", "deriv_pixels", "hess_pixels"):
            for container in (mass, src, ll):
                if not getattr(container, "has_pixels", False):
                    continue
                idx = container.pixelated_index
                if idx is None:
                    continue
                if container.func_list[idx] is not profile:
                    continue
                sh = container.pixelated_shape
                if sh is None:
                    return None
                return tuple(int(x) for x in jnp.shape(jnp.zeros(sh)))

        if param == "amps":
            na = getattr(profile, "num_amplitudes", None)
            if na is not None:
                return (int(na),)
        return None

    return infer_array_shape


def make_infer_array_shape_mass_only(mass_model) -> Callable[[object, str], Optional[tuple]]:
    """Like ``make_infer_array_shape`` but only mass pixelated profiles (GW-only setups)."""

    def infer_array_shape(profile: object, param: str) -> Optional[tuple]:
        if param in ("pixels", "func_pixels", "deriv_pixels", "hess_pixels"):
            if not getattr(mass_model, "has_pixels", False):
                return None
            idx = mass_model.pixelated_index
            if idx is None or mass_model.func_list[idx] is not profile:
                return None
            sh = mass_model.pixelated_shape
            if sh is None:
                return None
            return tuple(int(x) for x in jnp.shape(jnp.zeros(sh)))
        if param == "amps":
            na = getattr(profile, "num_amplitudes", None)
            if na is not None:
                return (int(na),)
        return None

    return infer_array_shape


def build_parameter_layout(
    lens_image,
    *,
    kwargs_lens: Optional[Sequence[Dict[str, Any]]] = None,
    kwargs_source: Optional[Sequence[Dict[str, Any]]] = None,
    kwargs_lens_light: Optional[Sequence[Dict[str, Any]]] = None,
    planes: Tuple[str, ...] = ("mass", "source", "lens_light"),
) -> Tuple[List[ParamEntry], List[str]]:
    """
    Build flat parameter entries from ``lens_image.MassModel / SourceModel / LensLightModel``.

    Templates ``kwargs_*`` supply parameter lists when a profile class omits ``param_names``.
    Pass ``planes=('mass',)`` for GW-only inference (no source / lens-light sampling).

    Returns ``(entries, extra_keys)``; extra_keys reserved for future use.
    """
    entries: List[ParamEntry] = []
    mass = lens_image.MassModel
    source = lens_image.SourceModel
    ll = lens_image.LensLightModel

    kl = list(kwargs_lens) if kwargs_lens is not None else [None] * len(mass.func_list)
    ks = list(kwargs_source) if kwargs_source is not None else [None] * len(source.func_list)
    kll = list(kwargs_lens_light) if kwargs_lens_light is not None else [None] * len(ll.func_list)

    if "mass" in planes:
        for i, prof in enumerate(mass.func_list):
            fb = kl[i] if i < len(kl) else None
            for p in _param_names_for_profile(prof, fb):
                entries.append(ParamEntry(f"lens{i}_{p}", prof, p, "mass", i))

    if "source" in planes:
        for j, prof in enumerate(source.func_list):
            fb = ks[j] if j < len(ks) else None
            for p in _param_names_for_profile(prof, fb):
                entries.append(ParamEntry(f"source{j}_{p}", prof, p, "source", j))

    if "lens_light" in planes:
        for k, prof in enumerate(ll.func_list):
            fb = kll[k] if k < len(kll) else None
            for p in _param_names_for_profile(prof, fb):
                entries.append(ParamEntry(f"light{k}_{p}", prof, p, "lens_light", k))

    extra_keys: List[str] = []
    return entries, extra_keys


def build_mass_parameter_entries(
    mass_model,
    kwargs_lens: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[ParamEntry]:
    """Flat layout for mass components only (GW-only inference without ``LensImage``)."""
    entries: List[ParamEntry] = []
    kl = list(kwargs_lens) if kwargs_lens is not None else [None] * len(mass_model.func_list)
    for i, prof in enumerate(mass_model.func_list):
        fb = kl[i] if i < len(kl) else None
        for p in _param_names_for_profile(prof, fb):
            entries.append(ParamEntry(f"lens{i}_{p}", prof, p, "mass", i))
    return entries


def flat_keys(entries: Sequence[ParamEntry]) -> List[str]:
    return [e.flat_key for e in entries]


def unpack_to_kwargs(
    flat: Dict[str, Any],
    entries: Sequence[ParamEntry],
    *,
    n_mass: int,
    n_source: int,
    n_lens_light: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert flat sampled dict to Herculens kwargs lists (one dict per component)."""
    kwargs_lens = [{} for _ in range(n_mass)]
    kwargs_src = [{} for _ in range(n_source)]
    kwargs_ll = [{} for _ in range(n_lens_light)]

    for e in entries:
        v = flat[e.flat_key]
        if e.plane == "mass":
            kwargs_lens[e.component_index][e.param] = v
        elif e.plane == "source":
            kwargs_src[e.component_index][e.param] = v
        else:
            kwargs_ll[e.component_index][e.param] = v

    return kwargs_lens, kwargs_src, kwargs_ll


def truth_vector_from_kwargs(
    entries: Sequence[ParamEntry],
    *,
    kwargs_lens: Sequence[Dict[str, Any]],
    kwargs_source: Sequence[Dict[str, Any]],
    kwargs_lens_light: Sequence[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a flat truth dict for Fisher / MCMC init from simulation kwargs."""
    out: Dict[str, Any] = {}
    for e in entries:
        if e.plane == "mass":
            out[e.flat_key] = kwargs_lens[e.component_index][e.param]
        elif e.plane == "source":
            out[e.flat_key] = kwargs_source[e.component_index][e.param]
        else:
            out[e.flat_key] = kwargs_lens_light[e.component_index][e.param]
    if extra:
        out.update(extra)
    return out


def _normalize_user_prior(
    name: str,
    value: Any,
    *,
    jnp_mod=jnp,
    numpyro_mod=numpyro,
    dist_mod=dist,
) -> Callable[[], Any]:
    # Distributions are callable in numpyro; check them before callable().
    if isinstance(value, dist_mod.Distribution):
        return lambda n=name, d=value: numpyro_mod.sample(n, d)
    if callable(value):
        return value
    return lambda v=value: numpyro_mod.sample(name, dist_mod.Delta(jnp_mod.asarray(v)))


def build_priors_registry(
    entries: Sequence[ParamEntry],
    *,
    lens_image=None,
    mass_model=None,
    user_priors: Optional[Dict[str, Any]] = None,
    qphi_mass_components: Optional[frozenset] = None,
) -> Dict[str, Callable[[], Any]]:
    """
    Merge profile default samplers with ``user_priors`` (callables, Distributions, or fixed values).

    Pass ``lens_image`` for EM paths, or ``mass_model`` alone for GW-only (no ``LensImage``).

    ``qphi_mass_components``: flat-key prefixes (e.g. {"lens0"}) of mass components
    whose e1/e2 pair should instead be sampled as (q, phi) and converted -- see
    ``ellipticity_reparam.py``. The registry still exposes '{prefix}_e1'/'{prefix}_e2'
    (now derived, via ``compute_qphi_ellipticity``) so ``unpack_to_kwargs`` needs no
    changes; it additionally exposes '{prefix}_q'/'{prefix}_phi' as the actual free
    parameters.
    """
    if lens_image is not None:
        infer_shape = make_infer_array_shape(lens_image)
    elif mass_model is not None:
        infer_shape = make_infer_array_shape_mass_only(mass_model)
    else:
        raise ValueError("build_priors_registry requires lens_image or mass_model.")
    reg: Dict[str, Callable[[], Any]] = {}
    for e in entries:
        reg[e.flat_key] = required_default_sampler(
            profile=e.profile,
            key=e.flat_key,
            param=e.param,
            infer_array_shape=infer_shape,
        )
    # NOTE: this only adds '{prefix}_q'/'{prefix}_phi' default samplers to the
    # registry -- it deliberately does NOT touch reg['{prefix}_e1']/['_e2']. Calling
    # compute_qphi_ellipticity(prefix, reg) independently from both an e1 lookup and
    # an e2 lookup would sample '{prefix}_q'/'{prefix}_phi' twice in one trace (a
    # numpyro duplicate-site error). Callers (FlexProbModel*.model(), the nautilus
    # builders) must call compute_qphi_ellipticity(prefix, p) themselves exactly
    # once per component and skip the registry's e1/e2 entries for that component.
    qphi_mass_components = qphi_mass_components or frozenset()
    for prefix in qphi_mass_components:
        reg.update(default_qphi_priors(prefix))
    user_priors = user_priors or {}
    warn_if_ellipticity_prior_keys_unused(
        user_priors,
        "q_phi" if qphi_mass_components else "e1e2",
        qphi_mass_components or {
            e.flat_key[: -len("_e1")] for e in entries
            if e.plane == "mass" and e.param == "e1"
        },
    )
    for name, val in user_priors.items():
        reg[name] = _normalize_user_prior(name, val)
    return reg


def build_vectorised_lens_kwargs_fn(entries: Sequence[ParamEntry], n_mass: int):
    """
    Return ``build_kwargs_lens_vectorised_fn`` for ``image_samples_to_source_samples``:
    stacks lens-only flat keys into ``kwargs_lens`` list of dicts of (n_samp,) arrays.
    """

    mass_entries = [e for e in entries if e.plane == "mass"]
    keys_by_comp: Dict[int, List[ParamEntry]] = {i: [] for i in range(n_mass)}
    for e in mass_entries:
        keys_by_comp[e.component_index].append(e)

    def build(sample_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        mass_keys = [e.flat_key for e in mass_entries]
        if not mass_keys:
            raise ValueError("No mass parameters in layout for vectorised lens kwargs.")
        n = int(jnp.asarray(sample_dict[mass_keys[0]]).shape[0])
        out: List[Dict[str, Any]] = []
        for i in range(n_mass):
            d: Dict[str, Any] = {}
            for e in keys_by_comp[i]:
                d[e.param] = jnp.asarray(sample_dict[e.flat_key])
            out.append(d)
        # Fill ra_0/dec_0 if missing (Shear)
        for i, kw in enumerate(out):
            if "gamma1" in kw and "ra_0" not in kw:
                kw["ra_0"] = jnp.zeros(n)
            if "gamma1" in kw and "dec_0" not in kw:
                kw["dec_0"] = jnp.zeros(n)
        return out

    return build
