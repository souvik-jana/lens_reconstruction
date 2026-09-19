"""
GWEMFISH tutorial: config -> simulate -> PAL mirror -> plots -> multi-mode
inference -> corner plots, using only gwemfish's native public functions.

Inference matrix (one shared ``ctx``):
  EM+GW    -> fisher-source, deriv-approx-source
  GW-only  -> fisher-source, deriv-approx-source
  EM-only  -> fisher, deriv-approx
    (``*-source`` methods are not defined for mode='EM-only': there is no GW
    source position to sample without a GW likelihood.)

Other supported methods not run here: hmc, hmc-informed, hmc-source,
hmc-informed-source, nautilus-image, nautilus-source.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX devices: {jax.devices()}")

import matplotlib
import numpyro.distributions as dist

matplotlib.use("Agg")

from gwemfish import (
    make_default_cfg,
    plot_posterior,
    plot_psf,
    plot_source_plane_caustic_with_localization_from_setup,
    plot_source_posterior,
    plot_system_observation,
    plot_system_observation_pal,
    prune_gw_images,
    run_inference,
    save_pal_outputs,
    setup_em_observation,
    setup_gw_observation,
    simulate_in_pal,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_comparison_corner

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "tutorial", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Configuration -------------------------------------------------
CFG = make_default_cfg()
CFG["use_parameter_layout"] = True  # flat lens0_*/lens1_*/light0_*/source0_* names
CFG["gw"]["n_images"] = 2  # double; n_images is a hint, prune after setup to match ctx
CFG["gw"]["source_box_half_width"] = 0.8  # default 0.05; Uniform y0gw/y1gw box around truth
CFG["source_plane"]["n_images"] = 2
CFG["inference"]["num_chains"] = 10  # default 2; matches XLA host device count
CFG["inference"]["num_samples"] = 14000  # default 12000; NUTS post-warmup draws per chain
CFG["inference"]["num_warmup"] = 6000

# Reference: every relevant cfg knob, shown at its default value. Uncomment and
# edit any line to override before simulating (this tutorial runs on defaults).
# CFG["jax"]["ncpus"] = None
# CFG["jax"]["enable_x64"] = True
# CFG["jax"]["platform"] = "cpu"
# CFG["jax"]["verbose"] = True
#
# CFG["em"]["enabled"] = True
# CFG["em"]["pixel_grid_kwargs"]["npix"] = 20
# CFG["em"]["pixel_grid_kwargs"]["pix_scl"] = 0.4
# CFG["em"]["psf_kwargs"]["psf_type"] = "GAUSSIAN"
# CFG["em"]["psf_kwargs"]["fwhm"] = 0.2
# CFG["em"]["noise_simu_kwargs"]["background_rms"] = 0.01
# CFG["em"]["noise_simu_kwargs"]["exposure_time"] = 1000.0
# CFG["em"]["noise_inf_kwargs"]["background_rms"] = None  # None -> estimated from model
# CFG["em"]["kwargs_numerics"]["supersampling_factor"] = 1
# CFG["em"]["exposure_time"] = 1000.0
# CFG["em"]["source_pos"] = (0.05, 0.1)
# CFG["em"]["kwargs_source"] = [{"amp": 4.0, "R_sersic": 0.5, "n_sersic": 2.0, "e1": 0.05, "e2": 0.05, "center_x": 0.05, "center_y": 0.1}]
# CFG["em"]["kwargs_lens_light"] = [{"amp": 8.0, "R_sersic": 1.0, "n_sersic": 3.0, "e1": -0.0556, "e2": 0.0962, "center_x": 0.0, "center_y": 0.0}]
# CFG["em"]["seed"] = 87651
# # source_model_class / lens_light_model_class default to single-Sersic factories;
# # override with e.g. `lambda: hcl.LightModel([hcl.Sersic()])` (import herculens as hcl).
#
# CFG["gw"]["enabled"] = True
# CFG["gw"]["n_images"] = 4
CFG["gw"]["source_pos"] = (0.2, 0.01)  # outside the inner caustic -> physical double
# CFG["gw"]["cosmology"]["H0"] = 67.3
# CFG["gw"]["cosmology"]["Om0"] = 0.316
# CFG["gw"]["solver_params"]["backend"] = "auto"
# CFG["gw"]["solver_params"]["nsolutions"] = "auto"
CFG["gw"]["error_scales"]["sigma_td"] = 0.001
CFG["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
# CFG["gw"]["error_scales"]["epsilon"] = 0.005
# CFG["gw"]["image_box_half_width"] = 0.6
# CFG["gw"]["source_box_half_width"] = 0.05
#
# CFG["mst"]["enabled"] = False
# CFG["mst"]["k_mst"] = 0.0
#
# CFG["lens"]["lens_model_list"] = ["EPL", "SHEAR"]
# CFG["lens"]["kwargs_lens"] = [
#     {"theta_E": 2.0, "e1": -0.0556, "e2": 0.0962, "gamma": 2.0, "center_x": 0.0, "center_y": 0.0},
#     {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
# ]
# CFG["lens"]["zl"] = 0.5
# CFG["lens"]["zs"] = 2.0
#
# CFG["inference"]["num_warmup"] = 6000
# CFG["inference"]["num_samples"] = 12000
# CFG["inference"]["num_chains"] = 2
# CFG["inference"]["max_tree_depth"] = 10
# CFG["inference"]["dense_mass"] = True
# CFG["inference"]["informed"] = None
# CFG["inference"]["n_fisher_samples"] = 5000
# CFG["inference"]["fisher_order"] = 2
# CFG["inference"]["rng_key"] = 123
# CFG["inference"]["prior_sample_rng_key"] = 123
# CFG["inference"]["diagnostics"] = "warn"
#
# CFG["plot"]["plot_mode"] = "groupwise"
# CFG["plot"]["color"] = "#2c3e50"
# CFG["plot"]["truth_color"] = "red"
# CFG["plot"]["quantiles"] = [0.05, 0.5, 0.975]
#
# CFG["source_plane"]["n_images"] = 2
# CFG["source_plane"]["seed"] = 42
#
# CFG["output"]["output_dir"] = "outputs"
#
# CFG["use_parameter_layout"] = False
# CFG["lens_mass_parametrization"] = "e1e2"

# --- 2. Simulate EM+GW ctx --------------------------------------------
ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
ctx = prune_gw_images(ctx, n_keep=2)

truth_params = ctx["truth_params"]
print(f"Simulated {ctx['n_images']} GW images. truth_params keys: {sorted(truth_params)}")

# --- 3. Plot the simulation (image, noisy obs, SNR map; PSF) ----------
plot_system_observation(
    ctx,
    cfg={"output": {"output_dir": OUTPUT_DIR, "save_system_plot_path": "system_observation.png"}},
)
plot_psf(ctx, cfg={"output": {"output_dir": OUTPUT_DIR, "save_psf_plot_path": "psf.png"}})

# --- 4. Mirror the same system in PyAutoLens (PAL) ---------------------
ctx_pal = simulate_in_pal(ctx)

plot_system_observation_pal(
    ctx_pal,
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "save_pal_dataset_plot_path": "pal_dataset_subplot.png",
            "save_pal_tracer_plot_path": "pal_tracer_subplot.png",
        },
        "plot": {"pal_plot_dataset": True, "pal_plot_tracer": True, "pal_dataset": "both"},
    },
)
save_pal_outputs(ctx_pal, OUTPUT_DIR)

stats = ctx_pal["match_stats"]
print("gwemfish <-> PAL consistency:")
for key, val in stats.items():
    print(f"  {key} = {val:.3e}" if isinstance(val, float) else f"  {key} = {val}")

# --- 5. Inference across modes -----------------------------------------
# cfg['priors'][name] accepts three kinds of override:
#   - a fixed scalar/array  -> held constant, dropped from the free-parameter count
#   - a numpyro.distributions.Distribution -> replaces the default prior, still sampled
#   - a zero-arg callable   -> custom numpyro.sample(...) logic
#
# With use_parameter_layout=True, names are lens0_*/lens1_*/light0_*/source0_*
# (lens0 = EPL, lens1 = SHEAR). lens1_ra_0/lens1_dec_0 (the shear reference
# point) become free sample sites under this naming and are redundant with the
# mass centre, so they are fixed at truth in every mode.
#
# GW-only has only 3 GW observables (2 images -> 1 time delay + 2 dL_eff) but
# many free mass params by default -> Fisher condition number blows up
# (diagnostic FAIL). Fix the full EPL+shear mass model and dL at truth so the
# free set is T_star, y0gw, y1gw (3), matching the observable count. y0gw/y1gw
# stay free inside the truth-centered source_box_half_width box. Uncomment the
# dist.Normal lines below to replace that box with a Distribution override.
#
# EM+GW: the fisher-source gradient diagnostic flagged several lens-light
# directions as off-centre (light0_n_sersic, light0_e1, light0_center_x, ...).
# Lens light is a nuisance component here (not the science target), so fix
# the lens mass centre and the full lens-light Sersic profile at truth too.
src = ctx["cfg"]["gw"]["source_pos"]
src_hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
y0_prior = dist.Uniform(float(src[0]) - src_hw, float(src[0]) + src_hw)
y1_prior = dist.Uniform(float(src[1]) - src_hw, float(src[1]) + src_hw)

PRIORS_BY_MODE = {
    "EM+GW": {
        "dL": float(truth_params["dL"]),
        "T_star": float(truth_params["T_star"]),
        # "y0gw": y0_prior,
        # "y1gw": y1_prior,
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        # "lens0_center_x": float(truth_params["lens0_center_x"]),
        # "lens0_center_y": float(truth_params["lens0_center_y"]),
        # "light0_center_x": float(truth_params["light0_center_x"]),
        # "light0_center_y": float(truth_params["light0_center_y"]),
        # "light0_e1": float(truth_params["light0_e1"]),
        # "light0_e2": float(truth_params["light0_e2"]),
        # "light0_R_sersic": float(truth_params["light0_R_sersic"]),
        # "light0_n_sersic": float(truth_params["light0_n_sersic"]),
        # "light0_amp": float(truth_params["light0_amp"]),
    },
    "GW-only": {
        "lens1_gamma1": float(truth_params["lens1_gamma1"]),
        "lens1_gamma2": float(truth_params["lens1_gamma2"]),
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        "lens0_e1": float(truth_params["lens0_e1"]),
        "lens0_e2": float(truth_params["lens0_e2"]),
        "lens0_theta_E": float(truth_params["lens0_theta_E"]),
        "lens0_center_x": float(truth_params["lens0_center_x"]),
        "lens0_center_y": float(truth_params["lens0_center_y"]),
        "lens0_gamma": float(truth_params["lens0_gamma"]),
        "dL": float(truth_params["dL"]),
        # "T_star": float(truth_params["T_star"]),
        "y0gw": dist.Uniform(-0.3, 0.5),
        "y1gw": dist.Uniform(-0.3, 0.3),
        # "y0gw": dist.Normal(loc=float(src[0]), scale=0.02),
        # "y1gw": dist.Normal(loc=float(src[1]), scale=0.02),
    },
    "EM-only": {
        "lens1_ra_0": float(truth_params["lens1_ra_0"]),
        "lens1_dec_0": float(truth_params["lens1_dec_0"]),
        # "lens0_center_x": float(truth_params["lens0_center_x"]),
        # "lens0_center_y": float(truth_params["lens0_center_y"]),
        # "light0_center_x": float(truth_params["light0_center_x"]),
        # "light0_center_y": float(truth_params["light0_center_y"]),
        # "light0_e1": float(truth_params["light0_e1"]),
        # "light0_e2": float(truth_params["light0_e2"]),
        # "light0_R_sersic": float(truth_params["light0_R_sersic"]),
        # "light0_n_sersic": float(truth_params["light0_n_sersic"]),
        # "light0_amp": float(truth_params["light0_amp"]),
    },
}

RESULTS = {}

for mode, methods in [
    # ("EM+GW", ("fisher-source", "deriv-approx-source")),
    ("GW-only", ("fisher-source", "deriv-approx-source")),
    # ("EM-only", ("fisher", "deriv-approx")),
]:
    for method in methods:
        tag = f"{mode}_{method}".replace("-", "_").replace("+", "").replace(" ", "")
        samples, truths = run_inference(
            ctx,
            mode=mode,
            method=method,
            cfg={
                "priors": PRIORS_BY_MODE[mode],
                "output": {"output_dir": OUTPUT_DIR, "json_tag": tag},
                "inference": {
                    "num_chains": 12,
                    "num_samples": 15000,
                    "num_warmup": 8000,
                    **({"informed": True} if method in ("deriv-approx", "deriv-approx-source") else {}),
                },
            },
        )
        RESULTS[(mode, method)] = (samples, truths)
        print(f"[{mode} / {method}] {len(samples)} sampled params, {len(next(iter(samples.values())))} draws")

# --- 6. Corner plots -----------------------------------------------------

# # EM+GW: source-plane samples already carry y0gw/y1gw directly, no
# # to_source_plane_samples conversion needed. Grouped by parameter category.
# mode = "EM+GW"
# samples_fisher, truths_fisher = RESULTS[(mode, "fisher-source")]
# samples_deriv, truths_deriv = RESULTS[(mode, "deriv-approx-source")]
# mode_dir = os.path.join(OUTPUT_DIR, "emgw")
# os.makedirs(mode_dir, exist_ok=True)

# plot_source_posterior(
#     samples_deriv,
#     truths=truths_deriv,
#     cfg={"output": {"output_dir": mode_dir}, "plot": {"plot_mode": "groupwise", "save_path": "source_posterior_deriv_approx_{group_name}.png"}},
# )
# plot_source_plane_caustic_with_localization_from_setup(
#     source_samples=samples_deriv,
#     ctx=ctx,
#     truths_source=truths_deriv,
#     level=0.90,
#     save_path=os.path.join(mode_dir, "source_localization_90.png"),
# )

# keys_common = frozenset(samples_fisher) & frozenset(samples_deriv)
# groups = create_default_param_groups({k: samples_deriv[k] for k in keys_common})
# truths_common = {k: v for k, v in truths_deriv.items() if k in truths_fisher}
# truths_dict = {g: {p: truths_common[p] for p in ps if p in truths_common} for g, ps in groups.items()}

# plot_comparison_corner(
#     samples_fisher,
#     samples_deriv,
#     groups,
#     labels=("fisher-source", "deriv-approx-source"),
#     truths_dict=truths_dict,
#     save_path=os.path.join(mode_dir, "comparison_{group_name}.png"),
#     hist_kwargs={"density": True},
# )

# GW-only: one combined corner with every sampled parameter together (not
# split into groups) via plot_mode="combined".
mode = "GW-only"
samples_fisher, truths_fisher = RESULTS[(mode, "fisher-source")]
samples_deriv, truths_deriv = RESULTS[(mode, "deriv-approx-source")]
mode_dir = os.path.join(OUTPUT_DIR, "gw_only")
os.makedirs(mode_dir, exist_ok=True)

plot_source_posterior(
    samples_deriv,
    truths=truths_deriv,
    cfg={"output": {"output_dir": mode_dir}, "plot": {"plot_mode": "combined", "save_path": "source_posterior_deriv_approx_combined.png"}},
)
plot_source_plane_caustic_with_localization_from_setup(
    source_samples=samples_deriv,
    ctx=ctx,
    truths_source=truths_deriv,
    level=0.90,
    save_path=os.path.join(mode_dir, "source_localization_90.png"),
)

keys_common = frozenset(samples_fisher) & frozenset(samples_deriv)
truths_common = {k: v for k, v in truths_deriv.items() if k in truths_fisher}
all_params_group = {"all_params": sorted(keys_common)}
truths_dict = {"all_params": {p: truths_common[p] for p in all_params_group["all_params"] if p in truths_common}}

plot_comparison_corner(
    samples_fisher,
    samples_deriv,
    all_params_group,
    labels=("fisher-source", "deriv-approx-source"),
    truths_dict=truths_dict,
    save_path=os.path.join(mode_dir, "comparison_{group_name}.png"),
    hist_kwargs={"density": True},
)

# # EM-only: plain image-plane/mass params, no y0gw/y1gw.
# samples_fisher_em, truths_fisher_em = RESULTS[("EM-only", "fisher")]
# samples_deriv_em, truths_deriv_em = RESULTS[("EM-only", "deriv-approx")]
# em_dir = os.path.join(OUTPUT_DIR, "em_only")
# os.makedirs(em_dir, exist_ok=True)

# plot_posterior(
#     samples_deriv_em,
#     truths=truths_deriv_em,
#     cfg={"output": {"output_dir": em_dir}, "plot": {"plot_mode": "groupwise", "save_path": "posterior_deriv_approx_{group_name}.png"}},
# )

# groups_em = create_default_param_groups(samples_deriv_em)
# truths_common_em = {k: v for k, v in truths_deriv_em.items() if k in truths_fisher_em}
# truths_dict_em = {g: {p: truths_common_em[p] for p in ps if p in truths_common_em} for g, ps in groups_em.items()}

# plot_comparison_corner(
#     samples_fisher_em,
#     samples_deriv_em,
#     groups_em,
#     labels=("fisher", "deriv-approx"),
#     truths_dict=truths_dict_em,
#     save_path=os.path.join(em_dir, "comparison_{group_name}.png"),
#     hist_kwargs={"density": True},
# )

# print(f"\nDone. Outputs under {OUTPUT_DIR}/")
