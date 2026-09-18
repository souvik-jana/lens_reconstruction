"""
Configurable ProbModel driven by ``parameter_layout`` (lens0_*, source0_*, light0_*).

Keeps likelihood logic here; profile priors live in ``profile_prior_rules`` via
``parameter_layout.build_priors_registry``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import herculens as hcl

from .config import SOLVER_PARAMS
from .data_sim import compute_gw_from_images
from .ellipticity_reparam import compute_qphi_ellipticity
from .lens_setup import image_count_penalty, remove_central_image, solve_and_select
from .parameter_layout import ParamEntry, flat_keys, unpack_to_kwargs
from .priors import DEFAULT_IMAGE_POSITION_PRIORS_EM, DEFAULT_IMAGE_POSITION_PRIORS_GW
from .prob_model import _sample_image_positions


def _qphi_skip_keys(qphi_mass_components):
    """'{prefix}_e1'/'{prefix}_e2' flat keys to skip in the per-entry sampling loop
    -- computed once per ProbModel instance (see each class's __init__) rather than
    on every call to _sample_flat_entries (i.e. every trace / HMC step)."""
    return {f"{prefix}_e1" for prefix in qphi_mass_components} | {
        f"{prefix}_e2" for prefix in qphi_mass_components
    }


def _sample_flat_entries(entries, p, qphi_mass_components, skip_keys):
    """Sample every ParamEntry into a flat dict, except e1/e2 pairs belonging to a
    mass component in ``qphi_mass_components`` -- those are sampled jointly as
    (q, phi) and converted once via compute_qphi_ellipticity, so lens{i}_e1/lens{i}_e2
    end up in ``flat`` either way and unpack_to_kwargs needs no changes."""
    flat = {}
    for e in entries:
        if e.flat_key in skip_keys:
            continue
        flat[e.flat_key] = p[e.flat_key]()
    for prefix in qphi_mass_components:
        e1, e2 = compute_qphi_ellipticity(prefix, p)
        flat[f"{prefix}_e1"] = e1
        flat[f"{prefix}_e2"] = e2
    return flat


def _default_extra_priors_em_gw() -> Dict[str, Callable[[], Any]]:
    return {
        "noise_sigma_bkg": lambda: numpyro.sample(
            "noise_sigma_bkg", dist.LogUniform(1.0e-6, 1.0e6)
        ),
        "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
        "dL": lambda: numpyro.sample("dL", dist.Uniform(10.0, 50000.0)),
    }


def _default_extra_priors_em_only() -> Dict[str, Callable[[], Any]]:
    return {
        "noise_sigma_bkg": lambda: numpyro.sample(
            "noise_sigma_bkg", dist.LogUniform(1.0e-6, 1.0e6)
        ),
    }


def _default_extra_priors_gw_only() -> Dict[str, Callable[[], Any]]:
    return {
        "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
        "dL": lambda: numpyro.sample("dL", dist.Uniform(0.00001, 50000.0)),
    }


def _default_extra_priors_gw_only_source() -> Dict[str, Callable[[], Any]]:
    return {
        **_default_extra_priors_gw_only(),
        "y0gw": lambda: numpyro.sample("y0gw", dist.Uniform(-1.0, 1.0)),
        "y1gw": lambda: numpyro.sample("y1gw", dist.Uniform(-1.0, 1.0)),
    }


def _default_extra_priors_em_gw_source() -> Dict[str, Callable[[], Any]]:
    return {
        **_default_extra_priors_em_gw(),
        "y0gw": lambda: numpyro.sample("y0gw", dist.Uniform(-1.0, 1.0)),
        "y1gw": lambda: numpyro.sample("y1gw", dist.Uniform(-1.0, 1.0)),
    }


class FlexProbModelEMGW(hcl.NumpyroModel):
    """EM + GW joint model with flat lens*/source*/light* parameters."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        em_observations: Optional[Dict[str, Any]] = None,
        lens_image=None,
        lens_gw=None,
        noise=None,
        image_position_priors: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
        qphi_mass_components: frozenset = frozenset(),
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_em_gw()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.qphi_mass_components = qphi_mass_components
        self._qphi_skip_keys = _qphi_skip_keys(qphi_mass_components)
        self.n_mass = len(lens_image.MassModel.func_list)
        self.n_source = len(lens_image.SourceModel.func_list)
        self.n_lens_light = len(lens_image.LensLightModel.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.lens_gw = lens_gw
        self.noise = noise
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_EM,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat = _sample_flat_entries(self.entries, p, self.qphi_mass_components, self._qphi_skip_keys)
        flat["noise_sigma_bkg"] = p["noise_sigma_bkg"]()
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()
        ## OLD #########################################################
        # kl, ks, kll = unpack_to_kwargs(
        #     flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        # )
        
        # sigma_bkg = flat["noise_sigma_bkg"]
        # model_image = self.lens_image.model(
        #     kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        # )
        ## NEW #########################################################
        kl, ks, kll = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )

        #inject traced k_mst into lens_image mass model before EM likelihood
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        if self.use_mst:
            self.lens_image.MassModel.kappa0 = k_mst_kw

        sigma_bkg = flat["noise_sigma_bkg"]
        model_image = self.lens_image.model(
            kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        )
        ########################################
        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )

        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors
        )

        T_star, dL = flat["T_star"], flat["dL"]
        # k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff, _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw
        )

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]
        epsilon = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )
        numpyro.sample(
            "betx_x_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
            obs=betx_x_diff,
        )
        numpyro.sample(
            "bety_y_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
            obs=bety_y_diff,
        )
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))

    def params2kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kl, ks, kll = unpack_to_kwargs(
            params,
            self.entries,
            n_mass=self.n_mass,
            n_source=self.n_source,
            n_lens_light=self.n_lens_light,
        )
        return {
            "kwargs_lens": kl,
            "kwargs_source": ks,
            "kwargs_lens_light": kll,
            "image_positions": [
                (params.get(f"image_x{i+1}", 0.0), params.get(f"image_y{i+1}", 0.0))
                for i in range(self.n_images)
            ],
        }

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["noise_sigma_bkg", "T_star", "dL"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra + [
            f"image_x{i+1}" for i in range(self.n_images)
        ] + [f"image_y{i+1}" for i in range(self.n_images)]


class FlexProbModelEMOnly(hcl.NumpyroModel):
    """EM-only likelihood with lens*/source*/light* flat parameters."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        em_observations: Optional[Dict[str, Any]] = None,
        lens_image=None,
        noise=None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
        qphi_mass_components: frozenset = frozenset(),
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_em_only()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.qphi_mass_components = qphi_mass_components
        self._qphi_skip_keys = _qphi_skip_keys(qphi_mass_components)
        self.n_mass = len(lens_image.MassModel.func_list)
        self.n_source = len(lens_image.SourceModel.func_list)
        self.n_lens_light = len(lens_image.LensLightModel.func_list)
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.noise = noise
        super().__init__()

    def model(self):
        p = self.priors
        flat = _sample_flat_entries(self.entries, p, self.qphi_mass_components, self._qphi_skip_keys)
        flat["noise_sigma_bkg"] = p["noise_sigma_bkg"]()

        kl, ks, kll = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )
        #inject traced k_mst into lens_image mass model before EM likelihood
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        if self.use_mst:
            self.lens_image.MassModel.kappa0 = k_mst_kw

        sigma_bkg = flat["noise_sigma_bkg"]
        model_image = self.lens_image.model(
            kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        )
        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )

    def all_flat_keys(self) -> List[str]:
        extra = ["noise_sigma_bkg"]
        if self.use_mst:
            extra.append("k_mst")
        return flat_keys(self.entries) + extra


class FlexProbModelGWOnly(hcl.NumpyroModel):
    """GW-only model with lens* mass parameters + image positions."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        lens_gw=None,
        image_position_priors: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
        qphi_mass_components: frozenset = frozenset(),
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_gw_only()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.qphi_mass_components = qphi_mass_components
        self._qphi_skip_keys = _qphi_skip_keys(qphi_mass_components)
        self.n_mass = len(lens_gw.mass_model.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw = lens_gw
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_GW,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.02,
            "epsilon": 0.001,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat = _sample_flat_entries(self.entries, p, self.qphi_mass_components, self._qphi_skip_keys)
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()

        kl, _, _ = unpack_to_kwargs(
            flat,
            self.entries,
            n_mass=self.n_mass,
            n_source=0,
            n_lens_light=0,
        )

        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors
        )
        T_star, dL = flat["T_star"], flat["dL"]
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff, _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw
        )

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]
        epsilon = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )
        numpyro.sample(
            "betx_x_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
            obs=betx_x_diff,
        )
        numpyro.sample(
            "bety_y_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
            obs=bety_y_diff,
        )
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["T_star", "dL"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra + [f"image_x{i+1}" for i in range(self.n_images)] + [
            f"image_y{i+1}" for i in range(self.n_images)
        ]


class FlexProbModelSourcePlaneGWOnly(hcl.NumpyroModel):
    """GW-only model, source-plane parametrisation, flat lens0_*/lens1_*/... mass
    parameters (parameter_layout equivalent of ``ProbModelSourcePlane_GW_only`` in
    ``prob_model.py``).

    Samples y0gw/y1gw directly and solves the lens equation *inside* the model via
    ``solver.solve(...)``, mirroring ``ProbModelSourcePlane_GW_only``'s structure but
    with a flat mass-parameter layout instead of hardcoded single-lens names. Pass a
    ``DifferentiableLensEquationSolver`` (``differentiable_solver.py``) as ``solver``
    for correct gradients (deriv-approx-source / Fisher); the raw helens solver gives
    exact-zero gradients there.

    Like ``ProbModelSourcePlane_GW_only``, this adds no betx_x_diff/bety_y_diff/
    log_jacobian terms -- the solver enforces beta self-consistency exactly by
    construction and the y0gw/y1gw prior is already flat in the source plane.
    """

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        lens_gw=None,
        solver=None,
        solver_params: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
        qphi_mass_components: frozenset = frozenset(),
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_gw_only_source()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.qphi_mass_components = qphi_mass_components
        self._qphi_skip_keys = _qphi_skip_keys(qphi_mass_components)
        self.n_mass = len(lens_gw.mass_model.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw = lens_gw
        self.solver = solver
        self.solver_params = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.02,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat = _sample_flat_entries(self.entries, p, self.qphi_mass_components, self._qphi_skip_keys)
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()
        flat["y0gw"] = p["y0gw"]()
        flat["y1gw"] = p["y1gw"]()

        kl, _, _ = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=0, n_lens_light=0
        )

        lens_center_x = kl[0].get("center_x", 0.0)
        lens_center_y = kl[0].get("center_y", 0.0)
        betas = jnp.array([flat["y0gw"], flat["y1gw"]])

        x_pos_array, y_pos_array, _, sel_flags = solve_and_select(
            self.solver, self.solver_params, betas, kl, self.lens_gw,
            self.n_images, lens_center_x, lens_center_y)
        # Reject parameters whose image count does not match the observation.
        numpyro.factor("image_count", image_count_penalty(sel_flags, self.n_images))

        T_star, dL = flat["T_star"], flat["dL"]
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, _, model_dL_eff,
         _, _, _, _) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )

    def params2kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kl, _, _ = unpack_to_kwargs(
            params, self.entries, n_mass=self.n_mass, n_source=0, n_lens_light=0
        )
        return {
            "kwargs_lens": kl,
            "y0gw": params.get("y0gw", 0.0),
            "y1gw": params.get("y1gw", 0.0),
        }

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["T_star", "dL", "y0gw", "y1gw"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra


class FlexProbModelSourcePlaneEMGW(hcl.NumpyroModel):
    """EM + GW joint model, source-plane parametrisation, flat lens0_*/source0_*/
    light0_* parameters (parameter_layout equivalent of ``ProbModelSourcePlane`` in
    ``prob_model.py``).

    Samples y0gw/y1gw directly and solves the lens equation *inside* the model via
    ``solver.solve(...)`` -- see ``FlexProbModelSourcePlaneGWOnly`` /
    ``ProbModelSourcePlane_GW_only`` docstrings for why no betx_x_diff/bety_y_diff/
    log_jacobian terms are needed here either.
    """

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        em_observations: Optional[Dict[str, Any]] = None,
        lens_image=None,
        lens_gw=None,
        noise=None,
        solver=None,
        solver_params: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
        qphi_mass_components: frozenset = frozenset(),
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_em_gw_source()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.qphi_mass_components = qphi_mass_components
        self._qphi_skip_keys = _qphi_skip_keys(qphi_mass_components)
        self.n_mass = len(lens_image.MassModel.func_list)
        self.n_source = len(lens_image.SourceModel.func_list)
        self.n_lens_light = len(lens_image.LensLightModel.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.lens_gw = lens_gw
        self.noise = noise
        self.solver = solver
        self.solver_params = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.gw_error_scales = {
            "sigma_td": 0.3,
            "sigma_dL_eff": 0.3,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat = _sample_flat_entries(self.entries, p, self.qphi_mass_components, self._qphi_skip_keys)
        flat["noise_sigma_bkg"] = p["noise_sigma_bkg"]()
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()
        flat["y0gw"] = p["y0gw"]()
        flat["y1gw"] = p["y1gw"]()

        kl, ks, kll = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )

        k_mst_kw = p["k_mst"]() if self.use_mst else None
        if self.use_mst:
            self.lens_image.MassModel.kappa0 = k_mst_kw

        sigma_bkg = flat["noise_sigma_bkg"]
        model_image = self.lens_image.model(
            kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        )
        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )

        lens_center_x = kl[0].get("center_x", 0.0)
        lens_center_y = kl[0].get("center_y", 0.0)
        betas = jnp.array([flat["y0gw"], flat["y1gw"]])

        x_pos_array, y_pos_array, _, sel_flags = solve_and_select(
            self.solver, self.solver_params, betas, kl, self.lens_gw,
            self.n_images, lens_center_x, lens_center_y)
        # Reject parameters whose image count does not match the observation.
        numpyro.factor("image_count", image_count_penalty(sel_flags, self.n_images))

        T_star, dL = flat["T_star"], flat["dL"]
        (_, model_time_delays, _, model_dL_eff,
         _, _, _, _) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw)

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )

    def params2kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kl, ks, kll = unpack_to_kwargs(
            params, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )
        return {
            "kwargs_lens": kl,
            "kwargs_source": ks,
            "kwargs_lens_light": kll,
            "y0gw": params.get("y0gw", 0.0),
            "y1gw": params.get("y1gw", 0.0),
        }

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["noise_sigma_bkg", "T_star", "dL", "y0gw", "y1gw"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra
