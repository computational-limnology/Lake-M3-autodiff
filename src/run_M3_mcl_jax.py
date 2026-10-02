"""
RUN a previously-trained MCL (ML-corrected) kz model standalone.

`train_M3_mcl_jax.py` trains an LSTM correction to the process-based
`eddy_diffusivity_hendersonSellers` closure (see its own module docstring
for the full design) and saves the trained weights plus every hyperparameter
needed to reproduce its forward pass to a pickle (`--nn-params-out`, default
`mcl_nn_params.pkl`). This script is the "run" half of that pair: it loads
that pickle and replays the exact same hybrid (process + learned
correction) simulation `simulate_hybrid()` computes, standalone -- no
training loop, no `jax.grad`, and (unlike `train_M3_mcl_jax.py`) no
temperature observations required at all, since nothing here is fit to
anything. The physical setup (lake/model/run config, forcing, initial
state) is read exactly the way `run_M3_jax.py` reads it, so this script can
be pointed at the same data directory `train_M3_mcl_jax.py` was trained on,
over the same window or a different one (a different `--steps`, or a
`run_config.csv` whose `start_time`/`end_time` has since been edited to
cover new forcing data) -- the trained correction is just a function of the
per-step forcing/state features, not tied to the exact steps it was
trained over.

Output is intentionally unambiguous about its provenance. The saved `.npz`
uses the same field names `run_M3_jax.py`'s `res_lake1_jax_full.npz` does
(`temp`/`o2`/`docr`/`docl`/`pocr`/`pocl`/`times`/`depth`/`volume`), so it is
a drop-in input to `plot_output.py` and anything else that already reads
that schema -- but it also carries a `model_source` field
(`"mcl_hybrid_inference"`) plus every hyperparameter and the `--nn-params`
path actually used (`mcl_target`, `mcl_hidden_size`,
`mcl_depth_basis_degree`, `mcl_nn_params_path`, ...), so a later look at the
file (or anyone it's shared with) cannot mistake it for a plain
process-based run. The console output says the same thing up front, in
words, before the simulation starts. The default output filename
(`res_lake1_mcl_hybrid.npz`) is likewise deliberately distinct from both
`run_M3_jax.py`'s `res_lake1_jax_full.npz` (pure process-based) and
`train_M3_mcl_jax.py`'s own `mcl_result.npz` (that script's own
baseline-vs-hybrid training-time comparison file) -- three different files,
three different names, by design.

Usage:
    python src/run_M3_mcl_jax.py Ravn [--nn-params mcl_nn_params.pkl]
        [--steps N] [--chunk-steps N] [--out res_lake1_mcl_hybrid.npz]

`--chunk-steps` only controls the memory/`jax.checkpoint` granularity of the
underlying scan (see `simulate_hybrid()`'s module-level docstring in
`train_M3_mcl_jax.py`) -- the simulated trajectory is identical regardless
of its value, since nothing is differentiated through here.
"""
import argparse
import os
import pickle
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from processBased_lakeModel_functions import (
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile,
    provide_phosphorus, provide_carbon, get_lake_config, get_model_params, get_run_config,
    get_ice_and_snow,
)
from jax_lakeModel_functions import default_params, default_geometry_wq
from run_M3_jax import build_forcing_series, add_wq_forcing_series
from train_M3_mcl_jax import (
    simulate_hybrid, DEFAULT_CHUNK_STEPS,
    DEFAULT_MAX_LOG_K0, DEFAULT_MAX_LOG_ALPHA, DEFAULT_MAX_LOG_N, DEFAULT_MIN_KZ,
)


def apply_forcing_features(forcing, feature_stats, forcing_feature_keys):
    """Z-score each of `forcing_feature_keys` using the SAVED (train-window)
    `feature_stats` from the trained-model pickle -- never recomputed here,
    since re-deriving mean/std from whatever window this script happens to
    be run over would silently give the network different-looking inputs
    than it was trained on. Same `(val - mean) / std` formula
    `train_M3_mcl_jax.py`'s own `build_forcing_features()` uses, just
    applying stored stats instead of computing new ones."""
    cols = []
    for key in forcing_feature_keys:
        mean, std = feature_stats[key]
        cols.append((forcing[key] - mean) / std)
    return jnp.stack(cols, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".")
    parser.add_argument("--nn-params", type=str, default="mcl_nn_params.pkl",
                         help="path (relative to data_dir) to the trained-weights pickle saved by "
                              "train_M3_mcl_jax.py's --nn-params-out (default mcl_nn_params.pkl)")
    parser.add_argument("--steps", type=int, default=None,
                         help="total simulation window in hourly steps (default: the full record "
                              "per run_config.csv's start_time/end_time) -- need not match the "
                              "window the model was trained over")
    parser.add_argument("--chunk-steps", type=int, default=None,
                         help="scan chunk size (default: whatever train_M3_mcl_jax.py used, read "
                              "from the pickle if present, else its own default "
                              f"{DEFAULT_CHUNK_STEPS}); purely a memory/compile-granularity knob, "
                              "has no effect on the result")
    parser.add_argument("--out", type=str, default="res_lake1_mcl_hybrid.npz",
                         help="output .npz path (relative to data_dir); default "
                              "res_lake1_mcl_hybrid.npz, deliberately distinct from "
                              "run_M3_jax.py's res_lake1_jax_full.npz and train_M3_mcl_jax.py's "
                              "own mcl_result.npz -- see module docstring")
    args = parser.parse_args()

    os.chdir(args.data_dir)

    nn_params_path = os.path.abspath(args.nn_params)
    if not os.path.exists(nn_params_path):
        raise SystemExit(
            f"{nn_params_path} not found -- train a model first with train_M3_mcl_jax.py "
            "(--nn-params-out picks where it saves the weights this script loads)."
        )
    with open(nn_params_path, "rb") as f:
        trained = pickle.load(f)

    nn_params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.float64), trained["nn_params"])
    target = trained["target"]
    hidden_size = trained["hidden_size"]
    depth_basis_degree = trained["depth_basis_degree"]
    feature_stats = trained["feature_stats"]
    forcing_feature_keys = trained["forcing_feature_keys"]
    max_log_correction = trained["max_log_correction"]
    max_kz = trained["max_kz"]
    # `.get(..., DEFAULT_MIN_KZ)` rather than a bare `.get(...)`: pickles
    # saved before the kz floor was added lack this key -- defaulting to
    # DEFAULT_MIN_KZ (rather than None) reproduces the same floor those
    # pickles' *training* run would have applied had it existed yet, so
    # inference stays consistent with training even for older pickles.
    min_kz = trained.get("min_kz", DEFAULT_MIN_KZ)
    ri_alpha_init = trained.get("ri_alpha_init")
    ri_n_init = trained.get("ri_n_init")
    ri_memory_hours = trained.get("ri_memory_hours")
    # `.get(..., DEFAULT_...)` rather than a bare `.get(...)`: these three are
    # `target == "ri"`-only (unused, so harmless as None, for `target == "kz"`
    # pickles), but an `target == "ri"` pickle saved before these hyperparameters
    # existed would otherwise pass `None` into simulate_hybrid's `jnp.clip(...,
    # -max_log_k0, max_log_k0)` and crash -- same backward-compatibility
    # treatment `chunk_steps` already gets above.
    max_log_k0 = trained.get("max_log_k0", DEFAULT_MAX_LOG_K0)
    max_log_alpha = trained.get("max_log_alpha", DEFAULT_MAX_LOG_ALPHA)
    max_log_n = trained.get("max_log_n", DEFAULT_MAX_LOG_N)

    chunk_steps = args.chunk_steps if args.chunk_steps is not None else trained.get("chunk_steps", DEFAULT_CHUNK_STEPS)

    print(f"Running trained MCL hybrid model -- NOT the plain process-based model -- "
          f"using weights from {nn_params_path}")
    if target == "kz":
        print(f"  target=kz  hidden_size={hidden_size}  depth_basis_degree={depth_basis_degree}  "
              f"max_log_correction={max_log_correction:g}  max_kz={max_kz:g}")
    else:
        print(f"  target=ri  hidden_size={hidden_size}  "
              f"ri_alpha_init={ri_alpha_init:g}  ri_n_init={ri_n_init:g}  "
              f"ri_memory_hours={ri_memory_hours:g}  max_kz={max_kz:g}")

    lake_num = 1
    lake_config = get_lake_config("./lake_config.csv", lake_num)
    model_params = get_model_params("./model_params.csv", lake_num)
    run_config = get_run_config("./run_config.csv", lake_num)
    ice_and_snow = get_ice_and_snow("./ice_and_snow.csv", lake_num)

    windfactor = float(lake_config["WindSpeed"])
    nx = int(run_config["nx"]); dt = float(run_config["dt"]); dx = float(run_config["dx"])
    area, depth, volume, hypso_weight = get_hypsography(
        hypsofile=run_config["hypso_ini_file"], dx=dx, nx=nx, outflow_depth=float(lake_config["outflow_depth"]),
    )

    desired_start = pd.Timestamp(run_config["start_time"])
    desired_end = pd.Timestamp(run_config["end_time"])
    n_steps_full = len(pd.date_range(desired_start, desired_end, freq="h"))
    n_steps = args.steps if args.steps is not None else n_steps_full
    step_times = np.arange(1, (n_steps + 1) * dt, dt)[:n_steps]

    simulated_end = desired_start + pd.Timedelta(seconds=float(step_times[-1]))
    lake_name = str(lake_config.name) if lake_config.name is not None else None
    print(
        f"Simulating {lake_name + ' ' if lake_name else ''}"
        f"from {desired_start:%Y-%m-%d %H:%M} to {simulated_end:%Y-%m-%d %H:%M} "
        f"({n_steps} hourly steps, nx={nx} vertical grid cells)"
    )

    meteo_all = provide_meteorology(
        meteofile=run_config["meteo_ini_file"], windfactor=windfactor, lat=lake_config["Latitude"],
        lon=lake_config["Longitude"], elev=lake_config["Elevation"], startDate=desired_start,
    )
    u_ini = initial_profile(initfile=run_config["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=desired_start)
    wq_ini = wq_initial_profile(
        initfile=run_config["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=desired_start,
    )
    tp_boundary = provide_phosphorus(tpfile=run_config["tp_ini_file"], startingDate=desired_start, startTime=1)
    carbon_data = provide_carbon(ocloadfile=run_config["oc_load_file"], startingDate=desired_start, startTime=1)
    carbon_data = carbon_data.dropna(subset=["oc"])

    forcing = build_forcing_series(meteo_all, step_times, windfactor)
    forcing = add_wq_forcing_series(forcing, tp_boundary, carbon_data, step_times)

    mean_depth = float(np.sum(volume) / np.max(area))
    hydro_res_time_hr = float(model_params["hydro_res_time"]) * 8760
    geometry = default_geometry_wq(
        area=area, depth=depth, volume=volume, dx=dx, dt=dt, latitude=float(lake_config["Latitude"]),
        altitude=float(lake_config["Elevation"]), hypso_weight=hypso_weight, mean_depth=mean_depth,
        hydro_res_time_hr=hydro_res_time_hr,
    )
    phys_params = default_params(
        model_params, ice_and_snow,
        diffusion_method=str(run_config.get("diffusion_method", "hendersonSellers")),
    )

    def _to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        return bool(x)

    ice_state = dict(
        ice=_to_bool(ice_and_snow["ice"]), Hi=ice_and_snow["Hi"], Hs=ice_and_snow["Hs"],
        Hsi=ice_and_snow["Hsi"], iceT=ice_and_snow["iceT"], rho_snow=ice_and_snow["rho_snow"],
    )

    u0 = jnp.asarray(deepcopy(u_ini), dtype=jnp.float64)
    o2_0 = jnp.asarray(deepcopy(wq_ini[0]), dtype=jnp.float64)
    docr_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.75, dtype=jnp.float64)
    docl_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.25, dtype=jnp.float64)
    pocr_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)
    pocl_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)
    init_state = (u0, o2_0, docr_0, docl_0, pocr_0, pocl_0)

    forcing_features = apply_forcing_features(forcing, feature_stats, forcing_feature_keys)

    print("\nRunning hybrid (process + learned correction) simulation...")
    t0 = time.time()
    hybrid_fn = jax.jit(lambda p: simulate_hybrid(
        p, phys_params, geometry, forcing, ice_state, init_state,
        chunk_steps, hidden_size, depth_basis_degree, forcing_features,
        target=target,
        max_log_correction=max_log_correction, max_kz=max_kz, min_kz=min_kz,
        max_log_k0=max_log_k0, max_log_alpha=max_log_alpha, max_log_n=max_log_n,
        ri_alpha_init=ri_alpha_init, ri_n_init=ri_n_init, ri_memory_hours=ri_memory_hours,
    ))
    hybrid = hybrid_fn(nn_params)
    jax.block_until_ready(hybrid)
    print(f"  done in {time.time() - t0:.1f}s ({(time.time() - t0) / n_steps * 1000:.3f} ms/step)")

    # --- save output: run_M3_jax.py-compatible schema, plus kz/ri
    # diagnostics and explicit MCL provenance fields (see module docstring).
    out_path = os.path.join(os.getcwd(), args.out)
    save_kwargs = dict(
        temp=np.asarray(hybrid["u"]), o2=np.asarray(hybrid["o2"]),
        docr=np.asarray(hybrid["docr"]), docl=np.asarray(hybrid["docl"]),
        pocr=np.asarray(hybrid["pocr"]), pocl=np.asarray(hybrid["pocl"]),
        uvel=np.asarray(hybrid["uvel"]), vvel=np.asarray(hybrid["vvel"]),
        E_seiche=np.asarray(hybrid["E_seiche"]),
        times=step_times, depth=np.asarray(depth), volume=np.asarray(volume),
        kz=np.asarray(hybrid["kz"]), kz_process=np.asarray(hybrid["kz_process"]),
        # --- provenance: makes this file's origin unambiguous even without
        # reading this script's console output alongside it ---
        model_source=np.array("mcl_hybrid_inference"),
        mcl_nn_params_path=np.array(nn_params_path),
        mcl_target=np.array(target),
        mcl_hidden_size=hidden_size,
        mcl_chunk_steps=chunk_steps,
    )
    if target == "kz":
        save_kwargs.update(
            mcl_depth_basis_degree=depth_basis_degree,
            mcl_max_log_correction=max_log_correction,
            mcl_max_kz=max_kz, mcl_min_kz=min_kz,
        )
    else:
        save_kwargs.update(
            ri=np.asarray(hybrid["ri"]), k0=np.asarray(hybrid["k0"]),
            alpha=np.asarray(hybrid["alpha"]), n=np.asarray(hybrid["n"]),
            dens_diff=np.asarray(hybrid["dens_diff"]), Q_net=np.asarray(hybrid["Q_net"]),
            mcl_ri_alpha_init=ri_alpha_init, mcl_ri_n_init=ri_n_init,
            mcl_ri_memory_hours=ri_memory_hours, mcl_max_kz=max_kz, mcl_min_kz=min_kz,
        )
    np.savez(out_path, **save_kwargs)
    print(f"\nSaved MCL hybrid-model output (model_source='mcl_hybrid_inference') to {out_path}")


if __name__ == "__main__":
    main()
