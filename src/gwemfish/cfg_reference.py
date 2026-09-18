"""
Canonical, complete reference for GWEMFISH's `run_inference` pipeline `cfg` dict.

THIS FILE (`src/gwemfish/cfg_reference.py`) IS THE SINGLE CANONICAL REFERENCE,
importable as `from gwemfish.cfg_reference import COMPLETE_CFG, get_cfg`. It is
NOT `gwemfish.config` (that module holds physical constants, SOLVER_PARAMS, and
grid/PSF/noise defaults -- a different thing entirely). `scripts/cfg.py` and
`examples/scripts/cfg.py` are compatibility symlinks to this file, kept so both
historical import locations (`from cfg import ...`) keep working -- edit this
file only.

`gwemfish.simple_pipeline.make_default_cfg()` returns the *defaults*, but several
keys that matter in practice are either absent from that dict (they are read
directly off `cfg` with `.get(..., default)` at call sites, e.g. `cfg["nautilus"]`,
`cfg["gw"]["source_box_half_width"]`, `cfg["output"]["json_tag"]`) or present but
easy to miss because they only apply to specific `mode`/`method` combinations.
This module is a single place to look up every such key instead of grepping
through a dozen `examples/scripts/*.py` files for "the priors pattern that looks
close enough".

This file supersedes the old practice (see the `gwemfish-infer` Cursor skill,
which used to say "copy priors patterns from the closest examples/scripts/
file") of copy-pasting a whole example script and hoping its cfg block covers
what you need. It also absorbs the older `examples/scripts/cfg.py` template:
its `get_cfg()` (deep-copied default cfg dict) is preserved verbatim in the
"legacy compat" section at the bottom, so existing `from cfg import get_cfg`
consumers keep working.

Usage:

    from cfg import COMPLETE_CFG
    from gwemfish.simple_pipeline import deep_merge_cfg

    my_cfg = deep_merge_cfg(COMPLETE_CFG, {"inference": {"num_warmup": 2000}})

Or, more commonly, just copy the relevant block(s) below (e.g. the "nautilus"
block, PSF_EXAMPLES, or one side of PRIORS_EXAMPLES) straight into your own script's cfg
dict and delete everything else -- COMPLETE_CFG is not meant to be passed to
`run_inference` wholesale (some sibling keys are mutually exclusive, e.g.
`gw.use_mst`/`gw.k_mst` vs the preferred top-level `mst` block; `nautilus` is
read only when method is 'nautilus-source'/'nautilus-image').

Every key below is annotated with:
  - what it does
  - which mode(s) ('EM-only' / 'GW-only' / 'EM+GW') and method(s)
    ('fisher', 'fisher-source', 'deriv-approx', 'hmc', 'hmc-informed',
    'deriv-approx-source', 'hmc-source', 'hmc-informed-source',
    'nautilus-source', 'nautilus-image')
    it applies to -- "all" means all three modes / all ten methods
  - value type
  - default (from `make_default_cfg()` where one exists; otherwise "no
    default -- read via cfg.get(...)" is noted explicitly)

Source of truth for all of this: `src/gwemfish/simple_pipeline.py`
(`make_default_cfg`, `run_inference`, `_build_inference_probmodel`,
`_build_inference_probmodel_source_plane`), `src/gwemfish/nautilus_common.py`,
`src/gwemfish/nautilus_source_inference.py`, `src/gwemfish/nautilus_image_inference.py`,
`src/gwemfish/parameter_layout.py`, `src/gwemfish/config.py`, `src/gwemfish/priors.py`,
`src/gwemfish/pal_bridge.py` (opt-in PAL mirror: simulate_in_pal, plot_system_observation_pal).
"""

from copy import deepcopy

import numpy as np
import numpyro
import numpyro.distributions as dist


# ---------------------------------------------------------------------------
# Method / mode cheat sheet (see run_inference docstring for the authoritative
# version -- this is just so the comments below make sense without flipping
# back and forth to simple_pipeline.py):
#
#   mode:   'EM+GW' | 'GW-only' | 'EM-only'
#   method: 'fisher'                 -- Gaussian N(u0, (-H0)^-1) from Fisher Hessian, all modes
#           'fisher-source'          -- Source-plane counterpart of 'fisher': same Gaussian-sample
#                                       early return, but H0/u0/keys_to_include come from the
#                                       source-plane probmodel (y0gw/y1gw instead of image
#                                       positions). NOT valid for mode='EM-only'.
#           'deriv-approx'           -- NUTS on the Taylor/banana model (image-plane GW), all modes
#           'hmc'                    -- Plain NUTS on the full likelihood (image-plane GW), all modes
#           'hmc-informed'           -- Always-informed NUTS on the full likelihood, all modes
#           'deriv-approx-source'    -- Like deriv-approx but samples y0gw/y1gw directly and
#                                       solves the lens equation inside the model (differentiable
#                                       solver). NOT valid for mode='EM-only'.
#           'hmc-source'             -- Source-plane counterpart of 'hmc'. NOT valid for 'EM-only'.
#           'hmc-informed-source'    -- Source-plane counterpart of 'hmc-informed'. NOT valid for 'EM-only'.
#           'nautilus-source'        -- Nested sampling, source-plane GW (y0gw/y1gw). All modes
#                                       (EM-only nautilus uses the same pixel likelihood as
#                                       nautilus-image and REQUIRES use_parameter_layout=True).
#           'nautilus-image'         -- Nested sampling, image-plane GW (image_x*/image_y*). All modes.
#
# The '-source' family (fisher-source/deriv-approx-source/hmc-source/hmc-informed-source) and
# 'nautilus-source' both sample y0gw/y1gw and both solve the lens equation inside the
# likelihood. They still dispatch through different code paths -- the '-source' family via
# _build_inference_probmodel_source_plane (full NUTS/Taylor machinery), nautilus-source via
# nautilus_source_inference (scipy-prior nested sampling, no gradients) -- but they now build
# the SAME solver from the same cfg["gw"]["solver_params"], so they no longer disagree about
# where the images are.
#
# The image finder is chosen by solver_params["backend"] ('helens' triangle search or
# jaxtronomy's closed-form/grid solver); it is not fixed to helens. Differentiability does not
# come from the finder at all -- the finder's output is stop_gradient-ed, and a Newton polish
# with jax.lax.custom_root re-attaches derivatives via the implicit function theorem. That is
# why the finder is swappable and why nautilus-source, which needs no derivatives, may skip
# the polish (cfg["nautilus"]["polish"]) while the four gradient-based methods may not.
# ---------------------------------------------------------------------------


COMPLETE_CFG = {

    # ---- cfg["jax"]: JAX runtime setup, consumed by gwemfish.setup_jax(**cfg["jax"]) -----------
    # NOTE: simple_pipeline itself does not call setup_jax -- call it yourself at script start,
    # before importing jax-heavy gwemfish submodules.
    "jax": {
        "ncpus": None,        # int or None. None => use all available cores. All modes/methods.
        "enable_x64": True,   # bool. GWEMFISH assumes float64 everywhere; do not set False.
        "platform": "cpu",    # "cpu" | "gpu" | "tpu".
        "verbose": True,      # bool. Print device count / platform on setup.
    },

    # ---- cfg["em"]: EM (imaging) simulation + inference settings --------------------------------
    # Read by setup_em_observation always; read by run_inference for mode in
    # ('EM-only', 'EM+GW') (all methods) to rebuild kwargs_source/kwargs_lens_light templates
    # when use_parameter_layout=True.
    "em": {
        "enabled": True,  # bool. False => setup_em_observation returns {} (skip EM entirely,
                           # e.g. for a pure GW-only run). Mirror with mode='GW-only'.
        "pixel_grid_kwargs": {"npix": 20, "pix_scl": 0.4},   # dict -> setup_pixel_grid(**...)
        # dict -> setup_psf(**...) -> ctx["lens_image"].PSF (fixed for the whole run).
        #
        # Default (Gaussian):
        #   {"psf_type": "GAUSSIAN", "fwhm": 0.2}
        #   fwhm [arcsec], pixel_size [arcsec]; optional truncation (sigma units).
        #   pixel_size MUST equal pixel_grid_kwargs["pix_scl"]: it is the pixel scale the
        #   kernel array is rendered on and nothing resamples it onto the image grid, so a
        #   mismatch leaves PSF.kernel_point_source (what plot_psf draws and what the PAL
        #   mirror injects) sampled on the wrong grid. Omit it and setup_em_observation
        #   fills it in from pix_scl; give a different value and it raises. For a sub-pixel
        #   PSF use PIXEL + kernel_supersampling_factor below, not a smaller pixel_size.
        #
        # Custom / instrument PSF (PIXEL):
        #   {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
        #   my_kernel: 2D numpy array, odd shape (e.g. 5x5), centered on the peak,
        #   sum-normalized (herculens convention). Load from FITS or build by hand.
        #
        # Supersampled kernel (real instrument PSFs are usually delivered this way):
        #   psf_kwargs:      {"psf_type": "PIXEL", "kernel_point_source": fine_kernel,
        #                     "kernel_supersampling_factor": 2}
        #   kwargs_numerics: {"supersampling_factor": 2, "supersampling_convolution": True}
        #   kernel_supersampling_factor only declares that the array is sampled at
        #   pix_scl / factor; herculens degrades it to pix_scl and uses the degraded
        #   kernel unless the numerics supersample by the SAME factor with
        #   supersampling_convolution=True. Mismatched factors are not an error:
        #   herculens discards the supplied kernel and interpolates a replacement
        #   (setup_em_observation warns). Fine kernel must be odd-sized;
        #   (n_coarse - 1) * factor + 1 is a safe size. factor must be a positive INTEGER
        #   (the degrade step averages whole pixels), and pixel_size is ignored for PIXEL --
        #   the kernel carries its own sampling.
        #
        # No convolution:
        #   {"psf_type": "NONE"}
        #
        # See PSF_EXAMPLES below and examples/scripts/example_pixel_psf.py,
        # example_psf_plot_and_pal.py. Verify with plot_psf(ctx) or
        # ctx["lens_image"].PSF.kernel_point_source after setup_em_observation.
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.2},  # pixel_size defaults to pix_scl
        "noise_simu_kwargs": {"npix": 20, "background_rms": 1e-2, "exposure_time": 1e3},
        # Inference-time noise: background_rms=None means it gets SAMPLED (noise_sigma_bkg prior)
        # rather than held fixed -- set a float here (or via cfg["priors"]["noise_sigma_bkg"]) to
        # fix it instead.
        "noise_inf_kwargs": {"npix": 20, "background_rms": None, "exposure_time": 1e3},
        # dict -> herculens numerics kwargs. supersampling_factor=1 is the default and
        # should stay unless there is a reason to raise it; add
        # "supersampling_convolution": True alongside a factor > 1 to convolve on the
        # subgrid (required when the PSF is narrower than a pixel).
        #   recommend_supersampling(cfg)          -> diagnostics + suggested setting
        #   check_supersampling_convergence(cfg)  -> measured convergence across factors
        # Both are advisory and never modify the cfg they are given.
        "kwargs_numerics": {"supersampling_factor": 1},
        "exposure_time": 1e3,  # float, seconds. Passed to simulate_em separately from noise kwargs.
        # (x, y) arcsec: EM source-light center used to derive EM image positions via the lens
        # equation (truth_params['x_image_true_em'] / ['y_image_true_em']). In practice
        # em["kwargs_source"][0]["center_x"/"center_y"] wins if set (see setup_em_observation);
        # this tuple is just the fallback when kwargs_source doesn't set center_x/y explicitly.
        "source_pos": (0.05, 0.1),
        # List[dict], one dict per Sersic-like source-light component (herculens LightModel).
        # Legacy flat truth keys: source_amp/source_R_sersic/source_n/source_e1/source_e2/
        # source_center_x/source_center_y (from kwargs_source[0] only). With
        # use_parameter_layout=True, ALL source components get source{j}_{param} flat names
        # instead (see parameter_layout.build_parameter_layout).
        "kwargs_source": [
            {
                "amp": 4.0, "R_sersic": 0.5, "n_sersic": 2.0,
                "e1": 0.05, "e2": 0.05, "center_x": 0.05, "center_y": 0.1,
            },
        ],
        # List[dict], one dict per lens-light (foreground galaxy light) component. Legacy flat
        # truth keys: light_amp/light_R_sersic/.../light_center_x/light_center_y (light[0] only).
        # With use_parameter_layout=True: light{k}_{param} for every component.
        "kwargs_lens_light": [
            {
                "amp": 8.0, "R_sersic": 1.0, "n_sersic": 3.0,
                "e1": -0.0556, "e2": 0.0962, "center_x": 0.0, "center_y": 0.0,
            },
        ],
        # Zero-arg callables returning herculens model instances, e.g.
        # (lambda: hcl.LightModel([hcl.Sersic()])). Restored after a JSON round-trip via
        # simple_pipeline.restore_em_model_factories (to_serializable stores them as repr strings).
        "source_model_class": None,       # None here => must come from make_default_cfg's default
        "lens_light_model_class": None,   # (DEFAULT_SOURCE_LIGHT_MODEL / DEFAULT_LENS_LIGHT_MODEL).
        "seed": 87651,  # int. RNG seed for simulate_em's noise realization.
    },

    # ---- cfg["gw"]: GW (time-delay / effective-luminosity-distance) simulation + inference -------
    "gw": {
        "enabled": True,  # bool. False => setup_gw_observation is a no-op (returns ctx unchanged).
        "n_images": 4,    # int. Nominal image count -- actual count used by run_inference is
                          # resolved from ctx (len(ctx['x_img_gw']) etc.) via _resolve_gw_n_images;
                          # a mismatch just emits a UserWarning and uses the ctx-derived value.
        "source_pos": (0.05, 1e-6),  # (y0, y1) arcsec: TRUE GW source-plane position. Drives
                                      # image solving / time delays; also becomes truth y0gw/y1gw
                                      # for the '-source' methods (see below).
        "cosmology": {"H0": 67.3, "Om0": 0.316},  # dict -> JAXCosmology(**cosmology). Flat LCDM only.
        # Lens-equation solver settings, nested by image-finder backend. Shared keys apply
        # whichever backend runs; each backend's own knobs live under its name, so settings
        # for the backend you are NOT using are carried along rather than silently dropped
        # and you can flip 'backend' without re-editing anything.
        #
        # These reach the solver used *during inference* for every '-source' method
        # (fisher-source, deriv-approx-source, hmc-source, hmc-informed-source and
        # nautilus-source) as well as the setup-time truth solve. The image-plane methods
        # (fisher, deriv-approx, hmc, hmc-informed, nautilus-image) use no solver at all --
        # they sample image_x{i}/image_y{i} directly -- so these have no effect there.
        #
        # A legacy flat dict is still accepted: each key is routed into its nest with a
        # DeprecationWarning naming the new location.
        "solver_params": {
            # --- shared ---
            "backend": "auto",     # "auto" | "helens" | "jaxtronomy". 'auto' picks jaxtronomy's
                                   # closed-form solver when lens_model_list supports it (one of
                                   # EPL/EPL_NUMBA/SIE/SIS plus optional SHEAR/CONVERGENCE),
                                   # else helens' triangle search.
            "nsolutions": "auto",  # int or "auto" -> n_images + 1. Exactly one spare slot: it
                                   # holds the central image when the profile has one (gamma < 2)
                                   # and stays padding when it does not (gamma >= 2 is singular
                                   # at the centre). Doubles get 3, triples 4, quads 5.
            "n_newton": 8,         # int >= 1. Newton-polish steps -- a step count, NOT an on/off
                                   # switch. The polish supplies the derivatives every
                                   # gradient-based method needs, so 0 raises. To skip polishing
                                   # on nautilus-source (the only method that can), use
                                   # cfg['nautilus']['polish'].
            "duplicate_tol": None, # float arcsec or None. Separation below which two solutions
                                   # are the same image. None -> 1e-6 when polished (positions
                                   # are good to ~1e-9), else half the solver-grid pixel scale.

            # --- helens: any lens model; positions accurate to ~the final triangle ---
            "helens": {
                "niter": 8,               # refinement iterations
                "scale_factor": 2,        # per-iteration triangle shrink
                "nsubdivisions": 5,       # initial subdivisions -- raise this FIRST when an
                                          # image is missed
                "pixel_scale_factor": 0.8,  # search-grid coarseness vs the image grid; lower is
                                            # finer and slower
            },

            # --- jaxtronomy: 'analytical' is closed-form (exact), 'lenstronomy' is grid + Newton ---
            "jaxtronomy": {
                "solver": "analytical",      # "analytical" | "lenstronomy"
                # analytical only
                "magnification_limit": 1e-4, # drops roots fainter than this. Measured for
                                             # EPL+SHEAR, theta_E=2: gamma>=2 has no central
                                             # image at all; gamma=1.7-1.9 produce a degenerate
                                             # 5th root at |mu|=0 (junk, correctly dropped);
                                             # gamma=1.5 has a REAL central image at |mu|~1.4e-3,
                                             # which 1e-4 keeps and jaxtronomy's suggested 1e-1
                                             # would silently discard.
                                             # NOTE this setting defines what counts as an image
                                             # for the simulation as well as the inference -- both
                                             # go through the same solver -- so lowering it can
                                             # raise n_images. That is intentional: the two sides
                                             # stay consistent by construction.
                "Nmeas": 400,                # angular sampling of the 1-D root solve
                "Nmeas_extra": 80,           # extra sampling at the low-shear end
                # lenstronomy only
                "min_distance": 0.01,
                "search_window": 15,
                "precision_limit": 1e-10,
                "num_iter_max": 1200,
                # both
                "arrival_time_sort": True,
            },
        },
        # GW likelihood scale factors (ProbModel* / FlexProbModel*, image-plane methods: fisher,
        # deriv-approx, hmc, hmc-informed, nautilus-image; source-plane methods read the same dict
        # via nautilus_source_inference / _build_inference_probmodel_source_plane too).
        "error_scales": {
            "sigma_td": 0.05,      # float. sigma_time_delay = sigma_td * observed time delays.
            "sigma_dL_eff": 0.2,   # float. sigma_dL_eff = sigma_dL_eff * observed dL_eff.
            # float, seconds. Floor under sigma_td*|obs_td| so tiny/zero time delays don't
            # collapse the likelihood width to ~0. Not in make_default_cfg()'s dict -- read via
            # error_scales.get("sigma_td_floor", 1.0) in prob_model.py/flex_prob_model.py/
            # nautilus_source_inference.py. Add explicitly if the default floor is wrong for
            # your system (e.g. very short time delays).
            "sigma_td_floor": 1.0,
            # float. Soft Normal(0, epsilon) width on the "images must ray-shoot to the same
            # source point" constraint (betx_x_diff/bety_y_diff) in image-plane ProbModel* only.
            # NOT present in any source-plane method (they solve the lens equation directly, no
            # such soft constraint needed). THIS IS THE KEY CAVEAT when comparing an image-plane
            # method's to_source_plane_samples() output against a native source-plane method
            # (nautilus-source / fisher-source / deriv-approx-source / hmc-source /
            # hmc-informed-source): with the
            # default 0.005, image-plane posteriors can look 2-3x wider in the source plane purely
            # from this term, not from the GW likelihood itself. Tighten to ~1e-4 before such a
            # comparison (see gwemfish-infer skill "Question 3.5" for the full workflow / NUTS
            # divergence tradeoffs of tightening this).
            "epsilon": 0.005,
        },
        # float, arcsec. Half-width of the truth-centered uniform box for EACH image_x{i}/image_y{i}
        # prior. mode='GW-only' only (EM+GW gets its image-position prior from
        # image_position_priors_override built the same way, but the box is also driven by this
        # same key in _build_inference_probmodel). Applies to fisher/deriv-approx/hmc/hmc-informed/
        # nautilus-image (image-plane methods only -- irrelevant to the '-source' family/
        # nautilus-source, which use source_box_half_width / source_plane_bounds instead).
        "image_box_half_width": 0.6,
        # float, arcsec. Half-width of the truth-centered uniform box on y0gw/y1gw directly.
        # NOT present in make_default_cfg()'s returned dict -- read via
        # cfg["gw"].get("source_box_half_width", 0.05) inside
        # _build_inference_probmodel_source_plane. Applies ONLY to the '-source' family
        # (fisher-source / deriv-approx-source / hmc-source / hmc-informed-source); irrelevant to
        # nautilus-source,
        # which uses source_plane_bounds below instead (different code path, different convention:
        # this one is a half-width around truth, that one is explicit (low, high) bounds).
        "source_box_half_width": 0.05,
        # dict[str, (low, high)] in DEFAULT_PRIORS_GW_SOURCE_PLANE's key space (lens_theta_E,
        # lens_e1, ..., T_star, dL, y0gw, y1gw). NOT present in make_default_cfg()'s returned dict.
        # Read only by nautilus_source_inference.build_gw_source_plane_problem /
        # build_em_gw_source_plane_problem: merged OVER DEFAULT_PRIORS_GW_SOURCE_PLANE (your entry
        # replaces that key's default (low, high) hard bounds -- used both to build the nautilus
        # scipy Prior's support AND (for non-fixed keys) the shape of default_dists via
        # layout_defaults_from_registry). Applies ONLY to method='nautilus-source'. Use this (not
        # cfg['priors']) to move the y0gw/y1gw box for nautilus-source; cfg['priors'] entries here
        # would need to be full scipy-compatible distributions instead (see parse_cfg_priors).
        "source_plane_bounds": {
            "y0gw": (0.05 - 0.02, 0.05 + 0.02),
            "y1gw": (1e-6 - 0.004, 1e-6 + 0.004),
        },
        # Legacy Mass Sheet Transform location -- prefer the top-level "mst" block below.
        # Still honored as a fallback by _resolve_mst_settings (top-level "mst" wins if enabled).
        "use_mst": False,  # bool.
        "k_mst": 0.0,      # float.
    },

    # ---- cfg["mst"]: preferred Mass Sheet Transform controls (all modes, all methods) ------------
    # If mst.enabled=True, run_inference adds a default Uniform(-0.99999, 0.99999) prior on
    # 'k_mst' unless cfg['priors']['k_mst'] is set. GW forward model uses LensImageGW.compute_mst
    # (JAX-traceable) so k_mst can be sampled or differentiated through.
    "mst": {
        "enabled": False,  # bool.
        "k_mst": 0.0,      # float. Truth / fixed value when not sampling k_mst via priors.
    },

    # ---- cfg["lens"]: lens mass model + geometry (all modes, all methods) -----------------------
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],  # List[str]. Herculens/jaxtronomy profile names,
                                               # e.g. ["EPL", "SHEAR"], ["SIS"], ["NFW", "SHEAR"].
                                               # Determines parameter_layout's lens0_*/lens1_*/...
                                               # blocks 1:1 with use_parameter_layout=True.
                                               # Any length/order. The opt-in PAL mirror converts
                                               # every entry (pal_bridge.MASS_PROFILE_BUILDERS:
                                               # EPL, SIE, SIS, NIE, SHEAR, SHEAR_GAMMA_PSI,
                                               # CONVERGENCE, POINT_MASS, MULTIPOLE, PIEMD, DPIE)
                                               # and raises on the rest (GAUSSIAN, PIXELATED*).
        # List[dict], same length as lens_model_list. Sparse entries are OK: missing keys are
        # filled from config._DEFAULT_KWARGS_BY_LENS_MODEL per profile name (e.g. EPL defaults
        # e1=e2=0, gamma=2.0, center_x=center_y=0.0) so the solver/herculens always sees complete
        # kwargs. This becomes the truth (ctx['kwargs_lens']) used to build legacy lens_* / lens0_*
        # truth_params.
        "kwargs_lens": [
            {"theta_E": 1.2, "e1": 0.0, "e2": 0.1, "gamma": 2.0, "center_x": 0.0, "center_y": 0.0},
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.5,  # float. Lens redshift.
        "zs": 2.0,  # float. Source redshift.
    },

    # ---- cfg["priors"]: user prior overrides (all modes, all methods) -----------------------------
    # Keyed by flat parameter name (legacy 'lens_theta_E' or layout 'lens0_theta_E' etc. --
    # whichever naming convention use_parameter_layout selects). Each value can be:
    #   - a numpyro Distribution instance, e.g. dist.Uniform(0.5, 2.0)  -> sampled with that prior
    #   - a zero-arg callable, e.g. (lambda: numpyro.sample("lens_e1", dist.Normal(0, 0.1)))
    #     -> used as-is (advanced: custom/derived sample sites)
    #   - a fixed scalar/array, e.g. 1.2  -> held FIXED (wrapped as a constant; the key is
    #     REMOVED from Fisher/MCMC's keys_to_include so no gradient is taken w.r.t. it)
    # Omitting a key entirely means: model default prior (image position uniform box for
    # image_x*/image_y*, y0gw/y1gw truth-centered box for -source methods, profile default sampler
    # from profile_prior_rules for use_parameter_layout=True keys, or a ValueError at Fisher
    # expansion-point-building time if the model has no default and truth_params doesn't have it
    # either).
    "priors": {},

    # ---- cfg["inference"]: NUTS / Fisher / informed-NUTS controls ---------------------------------
    "inference": {
        "num_warmup": 6000,    # int. NUTS warmup steps. methods: deriv-approx(-source), hmc(-source),
                               # hmc-informed(-source).
        "num_samples": 12000,  # int. NUTS post-warmup samples (same methods as above).
        "num_chains": 2,       # int. NUTS chains (same methods as above).
        "max_tree_depth": 10,  # int. NUTS tree depth cap (same methods as above).
        "dense_mass": True,    # bool. Dense mass matrix for PLAIN (non-informed) NUTS only, i.e.
                               # method in ('deriv-approx', 'deriv-approx-source', 'hmc',
                               # 'hmc-source') with informed=False/None. Ignored for informed NUTS
                               # (which uses the Fisher-derived mass matrix instead) and for
                               # 'fisher'/'fisher-source'/'nautilus-*'.
        # Hessian-informed NUTS-only knobs (hmc-informed(-source), or hmc(-source)/
        # deriv-approx(-source) with informed=True):
        "hmc_informed_scale": 1.0,          # float. Scales the informed mass matrix / step size.
        "hmc_informed_perturb_scale": 0.1,  # float. Perturbation scale for informed-NUTS init.
        # bool or None. Turns on informed NUTS for 'hmc'/'hmc-source'/'deriv-approx'/
        # 'deriv-approx-source'. IGNORED for 'hmc-informed'/'hmc-informed-source' (those are ALWAYS
        # informed; passing informed=False for them raises ValueError). N/A for 'fisher',
        # 'fisher-source', 'nautilus-source', 'nautilus-image'.
        "informed": None,
        # (n, n) array or None. Overrides the Fisher Hessian H0 used to build the informed-NUTS
        # mass matrix / the Gaussian covariance for 'fisher'/'fisher-source' / the Taylor expansion
        # for 'deriv-approx(-source)'. None => use the H0 that compute_fisher() computes at the
        # expansion point (truth_params merged with any fixed cfg['priors'] literals). Same H0
        # feeds ALL of: fisher(-source), deriv-approx(-source), hmc-informed(-source) -- it is
        # always computed once per run_inference call regardless of method.
        "H0": None,
        "n_fisher_samples": 5000,  # int. method='fisher'/'fisher-source' only: number of Gaussian
                                   # draws N(u0, cov).
        "fisher_order": 2,         # int. compute_fisher's Taylor order (2 = Hessian/quadratic).
        "rng_key": 123,            # int. Seeds setup/likelihood/Fisher/MCMC PRNGKeys (same value
                                   # reused for all of them inside run_inference).
        "prior_sample_rng_key": 123,  # int. Seeds the one-shot probmodel.get_sample() call used to
                                      # discover keys_to_include (the free-parameter name list).
        # Pre-inference checks at the TRUE parameters, before any sampling:
        #   1 images       solver reproduces the simulated positions, no duplicate/spurious central
        #   2 observables  time delays / magnifications / dL_eff match the simulation
        #   3 source box   y0gw/y1gw prior box fits inside the caustic (advisory: a box that pokes
        #                  out guarantees NUTS divergences, since the image-count penalty is a
        #                  cliff with no gradient to push back against)
        #   4 parameters   enough observables for the free parameters? Judged for GW-only only:
        #                  EM+GW / EM-only print the tally marked NA, because the EM pixels
        #                  constrain the model too and never enter the observable count
        #   5 fisher cond  scaled Fisher condition number + eigenvalue range; judged in every
        #                  mode, and the real arbiter of whether the widths mean anything
        #   6 gradient     g0/sqrt|diag H0| ~ 0, i.e. truth really is at the peak
        # Checks 1-3 need a solver, so they are skipped for image-plane methods and EM-only;
        # checks 4-6 need a Fisher expansion, so they are skipped for the nautilus methods.
        # Per-check thresholds. Override any subset; omitted keys keep the default.
        # Defaults live in gwemfish.diagnostics.DEFAULT_THRESHOLDS and were calibrated
        # on real systems rather than chosen: e.g. condition_limit 1e10 sits between a
        # 3-image system with 4 free parameters (cond 5.2e8, widths sigma/truth
        # 0.23-1.14, usable) and the same system with 5 free (cond 9.0e12, widths
        # 14-32x the parameter values). Raise a threshold when a system you trust
        # trips a check for a reason you understand.
        "diagnostics_thresholds": {
            # "position_tol": 1e-4,     # arcsec, check 1: solved vs simulated images
            # "observable_rtol": 1e-3,  # check 2: time delays / dL_eff
            # "condition_limit": 1e10,  # check 5: scaled Fisher condition number
            # "gradient_sigma": 0.5,    # check 6: |g0|/sqrt|diag H0|
        },
        "diagnostics": "warn",  # "warn" (default) | "raise" | "off".
                                # "off" is unsafe for the '-source' family: their expansion is
                                # built AT truth, so a bad solve there corrupts everything
                                # downstream with nothing to signal it.
        # bool. method='hmc-informed'/'hmc-informed-source' only (passed to run_mcmc_informed).
        # NOT present in make_default_cfg()'s returned dict -- read via
        # cfg["inference"].get("regularize", False). If True, eigendecomposes the Fisher mass
        # matrix and clips/regularizes small or negative eigenvalues before use (see
        # gwemfish.inference.run_mcmc_informed docstring) -- helps when H0 is near-singular.
        "regularize": False,
    },

    # ---- cfg["plot"]: plot_posterior / plot_source_posterior appearance (not read by run_inference) ---
    "plot": {
        "plot_mode": "groupwise",  # "groupwise" | "combined" | "subset".
        "color": "#2c3e50",
        "truth_color": "red",
        "show_titles": True,
        "title_kwargs": {"fontsize": 10},
        "title_fmt": ".3f",
        "quantiles": [0.05, 0.5, 0.975],
        "hist_kwargs": {"density": True},  # dict -> corner.corner(..., hist_kwargs=...).
        "params_to_plot": None,  # List[str] or None. Only used for plot_mode in ('combined','subset').
        "figsize": None,
        "save_path": None,  # str or None, relative to cfg['output']['output_dir'] unless absolute.
        # str or None. Suffix appended before the file extension, e.g.
        # "corner_{group_name}.png" + "hmc" -> "corner_{group_name}_hmc.png".
        "save_tag": None,
        # --- plot_system_observation_pal(ctx_pal, cfg=...) (opt-in PAL mirror; not run_inference) ---
        # Defaults: plot both dataset subplots + tracer; save only if output save_pal_* paths set.
        "pal_plot_dataset": True,   # bool. aplt.subplot_imaging_dataset on ctx_pal datasets.
        "pal_plot_tracer": True,    # bool. aplt.subplot_tracer on ctx_pal['tracer'].
        "pal_dataset": "both",      # "pal" | "gwemfish" | "both". Which Imaging to subplot.
                                    # "pal" = PAL-simulated noise; "gwemfish" = exact gwemfish arrays
                                    # wrapped for a later PAL fit (pal-infer golden rule).
    },

    # ---- cfg["source_plane"]: to_source_plane_samples() controls (image-plane -> source-plane
    # ray-shooting of POSTERIOR SAMPLES after the fact -- NOT the same thing as the native
    # '-source'/'nautilus-source' methods, which sample y0gw/y1gw directly and never need this) ----
    "source_plane": {
        "n_images": 4,          # NOTE: unused by to_source_plane_samples -- actual image count
                                 # always follows _resolve_gw_n_images(ctx, cfg), same rule as
                                 # run_inference. Kept for backward compatibility only.
        "n_subsample": None,    # int or None. Subsample posterior draws before ray-shooting.
        "seed": 42,             # int. RNG seed for subsampling.
        "filter_std": None,     # float or None. Drop source-plane draws beyond this many std devs.
        "use_filtered": False,  # bool. Whether filter_std is actually applied.
    },

    # ---- cfg["output"]: save paths (all modes, all methods) ---------------------------------------
    "output": {
        "output_dir": "outputs",  # str. Base dir; every other *_path below is relative to this
                                   # unless it is already absolute.
        "save_samples_path": None,       # str or None, .npz. Method name is auto-appended as a
                                          # tag by run_inference (e.g. "samples.npz" ->
                                          # "samples_hmc_informed.npz") -- you do not need to
                                          # vary this by hand per method.
        "save_truths_path": None,        # str or None, .npz. Same auto-tagging as above.
        "save_source_samples_path": None,  # str or None, .npz. Used by to_source_plane_samples's
                                            # caller convention (not written by run_inference itself).
        "save_system_plot_path": None,   # str or None, .png. plot_system_observation(): clean |
                                          # noisy | S/N map (3 panels) + optional image overlays.
        "save_psf_plot_path": None,      # str or None, .png. plot_psf(): PSF kernel linear+log10.
        # PAL mirror (opt-in, after simulate_in_pal): plot_system_observation_pal() output.
        # None => display only; str => save PNG(s) under output_dir (dirname of path used).
        # Both accept a bare filename (written under output_dir, or the cwd if unset)
        # or a path. The dataset plot ignores the basename and writes
        # dataset_subplot_pal / _gwemfish, since pal_dataset="both" produces two files.
        "save_pal_dataset_plot_path": None,
        "save_pal_tracer_plot_path": None,  # basename honoured (renamed after PAL writes it)
        "json_path": None,  # str or None. Pipeline JSON (injection + setup + samples + truths).
                            # Same auto-method-tagging as save_samples_path/save_truths_path
                            # inside run_inference. Legacy alias: "save_pipeline_json_path".
        # str or None. NOT present in make_default_cfg()'s returned dict. IMPORTANT NUANCE:
        # run_inference's OWN pipeline-JSON save (fisher/deriv-approx/hmc/hmc-informed branches,
        # and the nautilus finish helper) does NOT read this key at all -- it always tags
        # json_path with the METHOD NAME automatically (see save_json_path =
        # _append_tag_to_path(save_json_path, method_norm) in simple_pipeline.py). "json_tag" IS
        # read, however, by to_source_plane_samples (see its docstring) for the SEPARATE pipeline
        # JSON it writes after ray-shooting samples to the source plane. So: setting
        # cfg["output"]["json_tag"] before calling run_inference is harmless but has no effect on
        # run_inference's own JSON; set it (or rely on ctx["cfg"]) before calling
        # to_source_plane_samples if you want a tagged source-plane pipeline JSON. Legacy alias:
        # "save_pipeline_json_tag".
        "json_tag": None,
        # "gw" (default) | "em" | "both" | "none". Which observation(s) plot_system_observation
        # overlays image markers for.
        "system_plot_image_overlay": "gw",
    },

    # ---- Opt-in PAL mirror (NOT part of setup_em_observation / run_inference) -------------------
    # After EM (+ optional GW) simulation:
    #   ctx_pal = simulate_in_pal(ctx)              # gwemfish.pal_bridge; lazy autolens import
    #   plot_system_observation_pal(ctx_pal, cfg=CFG)  # reads cfg["plot"]["pal_*"] + output save paths
    #   save_pal_outputs(ctx_pal, out_dir)          # data_{gwemfish,pal}.fits + psf_*/noise_map_*,
    #                                               # tracer.json; dataset="gwemfish"|"pal" narrows it.
    #                                               # Fit data_gwemfish.fits (pal-infer golden rule).
    #
    # ctx_pal keys: tracer, grid, psf, dataset_pal, dataset_gwemfish, dataset_clean, match_stats, ...
    # match_stats: model_* (noiseless image), noise_map_*, noise_z_std, psf_* (kernel cross-check).
    # The PAL lens galaxy mirrors every profile in lens["lens_model_list"] (any length/order); see
    # pal_bridge.MASS_PROFILE_BUILDERS for the supported names and the gwemfish-pal skill for the
    # per-profile conversion rules.
    # Set em.psf_kwargs psf_type="PIXEL" + kernel_point_source for custom/real PSFs; the same kernel
    # is injected into PAL (Route 1). simulate_in_pal mirrors em.kwargs_numerics supersampling via
    # PAL over_sample_size, so model_max_rel_diff stays at the few x 1e-3 budget for any
    # supersampling_factor. With supersampling_convolution=True herculens convolves on the subgrid
    # while PAL always convolves at the image pixel scale, which leaves ~2-3% of peak that no PAL
    # setting removes (match_stats["supersampling"] records the settings in force).
    #
    # Related plot helpers (gwemfish side, before PAL): plot_system_observation (3 panels incl. S/N),
    # plot_psf, compute_noise_snr_maps(ctx) for standalone sigma/SNR arrays.

    # ---- cfg["use_parameter_layout"]: flat-name convention switch (all modes, all methods) --------
    # bool. False (default) = legacy hardcoded single-main-lens flat names (lens_theta_E, lens_e1,
    # lens_e2, lens_gamma, lens_gamma1, lens_gamma2, lens_center_x, lens_center_y, plus
    # source_*/light_* from kwargs_source[0]/kwargs_lens_light[0] only). Assumes an EPL+SHEAR
    # decomposition; wrong for e.g. SIS-only or multi-lens-plane systems.
    # True = auto-generated flat names for EVERY component in lens_model_list / kwargs_source /
    # kwargs_lens_light: "lens{i}_{param}" (i = index into lens_model_list, e.g. lens0_theta_E,
    # lens1_gamma1 for EPL+SHEAR), "source{j}_{param}", "light{k}_{param}" -- see
    # parameter_layout.build_parameter_layout / build_mass_parameter_entries. Default priors per
    # profile come from gwemfish.profile_prior_rules.required_default_sampler instead of the
    # hardcoded image-plane defaults. As of the current codebase this flag is GENERALIZED: it
    # works uniformly across deriv-approx, hmc, hmc-informed, nautilus-image, AND the source-plane
    # family (fisher-source, deriv-approx-source, hmc-source, hmc-informed-source, nautilus-source)
    # -- not just
    # the original image-plane methods. The ONLY place it is REQUIRED (not just opt-in) is
    # EM-only + method='nautilus-source'/'nautilus-image' (build_em_only_nautilus_problem raises
    # ValueError without it). Everywhere else, False remains a fully supported, non-breaking default.
    "use_parameter_layout": False,

    # ---- cfg["lens_mass_parametrization"]: e1/e2 vs q/phi (all modes, all methods) -----------------
    # "e1e2" (default) or "q_phi". Optional reparametrization of the main lens galaxy's mass
    # ellipticity: instead of sampling e1/e2 directly, samples q (axis ratio, Uniform(0.01, 1.0))
    # and phi (position angle in radians, Uniform(-pi/2, pi/2)), then derives e1/e2 via
    # herculens.Util.param_util.phi_q2_ellipticity (matches lenstronomy's convention exactly --
    # see ellipticity_reparam.py). e1/e2 are still registered as numpyro.deterministic sites, so
    # they appear in every posterior/Fisher output exactly as before regardless of which
    # parametrization was used to sample them -- corner plots, diagnostics, params2kwargs,
    # to_source_plane_samples all keep working unchanged. Works uniformly across hmc, nautilus,
    # nautilus-source, fisher, deriv-approx, and their -source variants; both
    # use_parameter_layout=False (flat "lens_e1"/"lens_q"/"lens_phi") and True ("lens{i}_e1"/
    # "lens{i}_q"/"lens{i}_phi" for the mass component using e1/e2 -- PIEMD/DPIE already sample
    # q/phi natively and are unaffected). Override the q/phi priors themselves the same way as any
    # other parameter, via cfg['priors']['lens_q']/['lens_phi'] (or 'lens{i}_q'/'lens{i}_phi').
    # Setting an e1/e2 override while this is "q_phi" (or a q/phi override while "e1e2") logs one
    # warning at construction -- that override is unused, not applied.
    "lens_mass_parametrization": "e1e2",

    # ---- cfg["nautilus"]: Nautilus nested-sampler controls ----------------------------------------
    # NOT present in make_default_cfg()'s returned dict at all -- entirely optional and only read
    # when method is 'nautilus-source' or 'nautilus-image' (see simple_pipeline._finish_nautilus_run,
    # nautilus_common.run_nautilus, nautilus_source_inference.build_gw_source_plane_problem /
    # build_em_gw_source_plane_problem). Safe to omit for every other method.
    "nautilus": {
        "n_live": 500,     # int. Live points -> nautilus.Sampler(n_live=...).
        "filepath": None,  # str or None. HDF5 checkpoint path -> nautilus.Sampler(filepath=...).
                           # IMPORTANT: changing free parameters or priors after a checkpoint
                           # exists requires resume=False (or delete the .hdf5). Prior changes
                           # are caught by prior_check below; likelihood changes (sigma_td,
                           # epsilon, solver_backend) or n_live changes are NOT -- Nautilus
                           # would silently resume the OLD problem.
        "resume": True,    # bool -> nautilus.Sampler(resume=...).
        "verbose": True,   # bool -> sampler.run(verbose=...).
        "prior_check": True,  # bool, DEFAULT True (nautilus_common.run_nautilus). On every
                              # checkpointed run, writes a prior-fingerprint sidecar
                              # <filepath>.priors.json (per-parameter ppf quantiles; for Uniform
                              # priors the outer two ~= the box edges, so `cat` the file to see
                              # which priors the checkpoint was built under). On resume=True,
                              # raises ValueError if the current priors differ from the sidecar
                              # (nautilus stores unit-cube points and maps them through the
                              # CURRENT prior, so a mismatched resume silently returns a wrong
                              # posterior). Legacy checkpoints without a sidecar warn once and
                              # get one written. Set False to opt out.
        # The following four are forwarded as **run_kwargs to sampler.run(...) (NOT to the
        # Sampler constructor) -- see simple_pipeline._finish_nautilus_run's run_kwarg_keys set:
        "n_eff": None,               # float or None. Target effective sample size to stop at.
        "n_like_max": None,          # int or None. Max likelihood evaluations before stopping.
        "discard_exploration": None,  # bool or None. Discard the exploration-phase live points.
        "timeout": None,             # float or None, seconds. Wall-clock stop condition.
        # The following two are consumed directly by nautilus_source_inference /
        # nautilus_image_inference (NOT forwarded to nautilus.Sampler or sampler.run at all --
        # simple_pipeline._finish_nautilus_run explicitly skips them via its `_skip` set):
        # Newton polish on/off -- nautilus-source ONLY, and the only method where it is a
        # choice. Nested sampling needs no derivatives, so skipping the polish here costs
        # accuracy rather than correctness; every gradient-based method requires it and
        # never reads this key.
        "polish": "auto",   # "auto" | True | False.
                            # "auto" polishes only when it changes the answer: skipped for the
                            # jaxtronomy finders (positions already exact/converged) and applied
                            # for helens (~0.05 arcsec -> ~1e-6). With the default
                            # backend='auto' on EPL+SHEAR this means nautilus-source is both
                            # faster and more accurate than the historical raw-helens path.
                            # False reproduces that historical path exactly, at its cost.
        "solver_backend": "helens",  # DEPRECATED alias of solver_params['backend'], which
                                     # applies to every method rather than nautilus alone.
                                     # Still honoured; emits a DeprecationWarning.
        "solver_validation_tol": 0.05,  # float, arcsec. At problem-build time the solver is
                                        # validated against ctx's truth image positions; a
                                        # residual above this emits a UserWarning. Note the
                                        # image-COUNT check now raises: it compares distinct
                                        # images found, not len(x_pos), which was a
                                        # compile-time constant and could never fire.
    },
}


# ---------------------------------------------------------------------------
# PSF_EXAMPLES
#
# Override cfg["em"]["psf_kwargs"] before setup_em_observation. The PSF is baked
# into ctx["lens_image"] once at setup and applies to simulation and inference
# (all mode/method combinations). For PAL mirror runs, the same kernel is passed
# through simulate_in_pal (see PAL mirror comment block in COMPLETE_CFG).
#
# Minimal PIXEL override in a script:
#
#   import numpy as np
#   from gwemfish.simple_pipeline import make_default_cfg, setup_em_observation
#
#   cfg = make_default_cfg()
#   my_kernel = np.load("instrument_psf.npy")   # or build / read from FITS
#   cfg["em"]["psf_kwargs"] = {
#       "psf_type": "PIXEL",
#       "kernel_point_source": my_kernel,
#   }
#   ctx = setup_em_observation(cfg=cfg)
#   # optional: from gwemfish import plot_psf; plot_psf(ctx, cfg={"output": {"save_psf_plot_path": "psf.png"}})
#
# Kernel requirements (herculens PIXEL PSF):
#   - 2D array, odd size on both axes (e.g. 5x5, 11x11)
#   - centered: peak at array center
#   - typically sum(my_kernel) == 1
#
# Hand-built Gaussian stand-in (see example_psf_plot_and_pal.py):
#
#   pix_scl = cfg["em"]["pixel_grid_kwargs"]["pix_scl"]
#   fwhm = 0.2
#   sigma_px = fwhm / (2 * np.sqrt(2 * np.log(2))) / pix_scl
#   half = 2
#   y, x = np.mgrid[-half:half + 1, -half:half + 1]
#   k = np.exp(-(x**2 + y**2) / (2 * sigma_px**2))
#   my_kernel = k / k.sum()
#   cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
# ---------------------------------------------------------------------------

PSF_EXAMPLE_GAUSSIAN = {
    "psf_type": "GAUSSIAN",
    "fwhm": 0.2,
    # "pixel_size": omitted on purpose -- setup_em_observation sets it to pix_scl. Give a
    # value only to assert it, and it must equal pix_scl (anything else raises).
    # "truncation": 3.0,  # optional, sigma units
}

PSF_EXAMPLE_PIXEL = {
    "psf_type": "PIXEL",
    "kernel_point_source": None,  # replace with your (odd, odd) centered 2D array
    # Kernel sampled at pix_scl / factor. Requires the matching numerics below, else
    # the kernel is degraded to pix_scl and the extra resolution is unused:
    #   "kwargs_numerics": {"supersampling_factor": 2, "supersampling_convolution": True}
    # "kernel_supersampling_factor": 2,
}

PSF_EXAMPLE_NONE = {"psf_type": "NONE"}


# ---------------------------------------------------------------------------
# PRIORS_EXAMPLES
#
# Same physical system (EPL main lens + external SHEAR), same free parameters,
# shown in both flat-naming conventions so the correspondence between
# use_parameter_layout=False and use_parameter_layout=True is unambiguous.
# lens_model_list = ["EPL", "SHEAR"] => layout mode gives "lens0_*" (EPL) and
# "lens1_*" (SHEAR), per parameter_layout.build_parameter_layout /
# build_mass_parameter_entries (flat_key = f"lens{i}_{param}" for the i-th
# entry of lens_model_list; SHEAR's own params are gamma1/gamma2/ra_0/dec_0).
# ---------------------------------------------------------------------------

PRIORS_EXAMPLE_LEGACY_FLAT = {
    # use_parameter_layout=False (default). Hardcoded EPL+SHEAR flat names --
    # these exact keys exist regardless of lens_model_list content (see
    # simple_pipeline._legacy_epl_like_kw / _legacy_shear_kw): if the real lens
    # has no EPL/SHEAR component these fall back to _LEGACY_TRUTH_MASS_KW_FALLBACK
    # (theta_E=1.0, e1=e2=0.0, gamma=2.0, center_x=center_y=0.0) rather than
    # raising, but that fallback is almost never physically meaningful for a
    # non-EPL system -- prefer use_parameter_layout=True for anything beyond
    # single EPL(+SHEAR).
    "lens_theta_E": dist.Uniform(0.5, 2.0),
    "lens_e1": dist.Normal(0.0, 0.3),
    "lens_e2": dist.Normal(0.0, 0.3),
    "lens_gamma": dist.Uniform(1.5, 2.5),
    "lens_center_x": 0.0,   # fixed (float) -- no gradient taken w.r.t. this key
    "lens_center_y": 0.0,   # fixed (float)
    "lens_gamma1": dist.Uniform(-0.3, 0.3),
    "lens_gamma2": dist.Uniform(-0.3, 0.3),
}

PRIORS_EXAMPLE_LAYOUT_FLAT = {
    # use_parameter_layout=True. Same physical priors as above, one block per
    # lens_model_list entry: lens0_* <-> EPL (index 0), lens1_* <-> SHEAR (index 1).
    # Any entry you omit here falls back to the profile's own default sampler
    # from gwemfish.profile_prior_rules (not to the legacy image-plane
    # defaults) -- so the two dicts above/below are NOT drop-in replacements
    # for each other's omissions, only for the keys both explicitly set.
    "lens0_theta_E": dist.Uniform(0.5, 2.0),
    "lens0_e1": dist.Normal(0.0, 0.3),
    "lens0_e2": dist.Normal(0.0, 0.3),
    "lens0_gamma": dist.Uniform(1.5, 2.5),
    "lens0_center_x": 0.0,
    "lens0_center_y": 0.0,
    "lens1_gamma1": dist.Uniform(-0.3, 0.3),
    "lens1_gamma2": dist.Uniform(-0.3, 0.3),
    # SHEAR also carries ra_0/dec_0 (shear reference point); parameter_layout
    # includes them as lens1_ra_0/lens1_dec_0 if SHEAR's param_names lists them.
    # Typically fixed to the lens center, e.g.:
    "lens1_ra_0": 0.0,
    "lens1_dec_0": 0.0,
}


def nautilus_priors_from_fisher_h0(ctx, span=2.0):
    """Build tight Uniform(mu +/- span*sigma) priors for Nautilus from a prior
    Fisher / deriv-approx run, following the gwemfish-infer skill's
    NAUTILUS_SIGMA_SPAN workflow.

    Call this AFTER `run_inference(ctx, mode=..., method='deriv-approx', ...)`
    (or method='fisher') has populated `ctx['likelihood']` and `ctx['fisher']`,
    and BEFORE `run_inference(ctx, mode=..., method='nautilus-source'` or
    `'nautilus-image', ...)`. Mutates and returns ctx['cfg']['priors'] in place.

    Convention used across examples/scripts: span=5.0 for EM-only (see
    em_nautilus.py), span=2.0 for GW-only / EM+GW (see gw_only_nautilus.py,
    gw_only_nautilus_image.py).

    Caveat for method='nautilus-source' specifically: H0 from an image-plane
    deriv-approx run covers lens0_*/lens_*, T_star, dL, image_x*/image_y* --
    it never covers y0gw/y1gw (those only exist in the source-plane
    parametrization). Set cfg["gw"]["source_plane_bounds"] and/or
    cfg["priors"]["y0gw"/"y1gw"] by hand for that method; this function will
    simply skip keys not present in keys_to_include.
    """
    keys = ctx["likelihood"]["keys_to_include"]
    u0 = np.asarray(ctx["likelihood"]["u0"])
    h0 = np.asarray(ctx["fisher"]["H0"])
    fisher_matrix = -h0
    try:
        cov = np.linalg.inv(fisher_matrix)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_matrix)
    sigmas = np.sqrt(np.diag(cov))

    priors = ctx["cfg"].setdefault("priors", {})
    for i, key in enumerate(keys):
        sigma = float(sigmas[i])
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        mu = float(u0[i])
        priors[key] = dist.Uniform(mu - span * sigma, mu + span * sigma)
    return priors


# ===========================================================================
# LEGACY COMPAT -- old examples/scripts/cfg.py template (`get_cfg()`)
#
# Everything below is ported verbatim from the pre-merge examples/scripts/
# cfg.py so existing `from cfg import get_cfg` consumers keep working. It is a
# plain runnable default cfg (deep-copied on each call, lazy herculens imports
# in the light-model factories) -- NOT the annotated reference above. Prefer
# COMPLETE_CFG / make_default_cfg() for new work.
# ===========================================================================


def _default_source_light_model():
    import herculens as hcl

    return hcl.LightModel([hcl.Sersic()])


def _default_lens_light_model():
    import herculens as hcl

    return hcl.LightModel([hcl.Sersic()])


# This legacy template enumerates the older top-level/simple-pipeline options explicitly.
CFG = {
    "jax": {
        "ncpus": None,
        "enable_x64": True,
        "platform": "cpu",
        "verbose": True,
    },
    "em": {
        "enabled": True,
        # If omitted in your own cfg, pipeline uses package defaults.
        "pixel_grid_kwargs": {"npix": 20, "pix_scl": 0.4},
        # psf_type: GAUSSIAN | PIXEL (kernel_point_source) | NONE
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.2},  # pixel_size defaults to pix_scl
        "noise_simu_kwargs": {"npix": 20, "background_rms": 0.005, "exposure_time": 1000},
        "noise_inf_kwargs": {"npix": 20, "background_rms": None, "exposure_time": 1000},
        "kwargs_numerics": {"supersampling_factor": 1},
        "exposure_time": 1000,
        "source_pos": (0.05, 0.1),
        "kwargs_source": [
            {
                "amp": 100.0,
                "R_sersic": 0.3,
                "n_sersic": 1.5,
                "e1": 0.1,
                "e2": 0.0,
                "center_x": 0.05,
                "center_y": 0.1,
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 1.0,
                "R_sersic": 1.0,
                "n_sersic": 2.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
        # Zero-arg factories → ``herculens.LightModel``; swap components to match simulation.
        "source_model_class": _default_source_light_model,
        "lens_light_model_class": _default_lens_light_model,
        "seed": 87651,
    },
    "gw": {
        "enabled": True,
        "n_images": 4,
        "source_pos": (0.05, 1e-6),
        "cosmology": {"H0": 67.3, "Om0": 0.316},
        "solver_params": {
            "num_iter_max": 200,
            "precision_limit": 1e-10,
            "search_window": 4.0,
            "num_random_init": 12,
        },
        # sigma_td: fractional scale; sigma_td_floor: absolute minimum (seconds).
        # Effective sigma_td = max(sigma_td_floor, sigma_td * time_delay).
        # sigma_dL_eff * gw_obs['dL_eff'], epsilon * ones_like(betx_x_diff).
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_td_floor": 1.0,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
        "image_box_half_width": 0.6,
    },
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.0,
                "e1": 0.1,
                "e2": 0.0,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.05, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.5,
        "zs": 1.5,
    },
    # Parameter priors override registry:
    # - callable zero-arg sampler
    # - numpyro distribution instance
    # - fixed scalar/array
    "priors": {},
    # No "nautilus" block in this legacy template (see COMPLETE_CFG["nautilus"] above).
    # Note for checkpointed nautilus runs: "prior_check": True (default) -- refuses
    # resume=True if current priors differ from the checkpoint's <filepath>.priors.json
    # sidecar; opt out via cfg["nautilus"]["prior_check"] = False.
    "inference": {
        "num_warmup": 6000,
        "num_samples": 12000,
        "num_chains": 2,
        "max_tree_depth": 10,
        "dense_mass": True,
        # Informed NUTS toggle for method='hmc' or method='deriv-approx' (banana model).
        # For method='hmc-informed', informed NUTS is always used.
        "informed": None,
        "hmc_informed_scale": 1.0,
        "hmc_informed_perturb_scale": 0.1,
        # Optional custom Hessian for informed NUTS; None uses Fisher H0 from compute_fisher.
        "H0": None,
        "n_fisher_samples": 5000,
        "fisher_order": 2,
        "rng_key": 123,
        "prior_sample_rng_key": 123,
    },
    "plot": {
        "plot_mode": "groupwise",  # groupwise | combined | subset
        "color": "#2c3e50",
        "truth_color": "red",
        "show_titles": True,
        "title_kwargs": {"fontsize": 10},
        "title_fmt": ".3f",
        "quantiles": [0.05, 0.5, 0.975],
        "hist_kwargs": {"density": True},
        "params_to_plot": None,
        "figsize": None,
        # Use {group_name} for groupwise separate files.
        "save_path": None,
        # Optional suffix tag appended before extension.
        "save_tag": None,
        # plot_system_observation_pal (opt-in PAL mirror)
        "pal_plot_dataset": True,
        "pal_plot_tracer": True,
        "pal_dataset": "both",
    },
    "source_plane": {
        "n_images": 4,
        "n_subsample": None,
        "seed": 42,
        "filter_std": None,
        "use_filtered": False,
    },
    "output": {
        "output_dir": "outputs",
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "save_psf_plot_path": None,
        "save_pal_dataset_plot_path": None,
        "save_pal_tracer_plot_path": None,
        "json_path": None,
        # Optional suffix tag appended to json_path.
        "json_tag": None,
        "system_plot_image_overlay": "gw",
    },
}


def get_cfg():
    """Return a deep copy of the legacy full config template (see CFG above)."""
    return deepcopy(CFG)
