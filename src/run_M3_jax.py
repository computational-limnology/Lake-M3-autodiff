"""
Driver for the JAX lake-model port, mirroring `run_M3.py` but calling
`run_full_model` (or, with `--temp-only`, the lighter `run_temperature_model`)
from `jax_lakeModel_functions.py` instead of the numpy `run_wq_model`.

Data loading (bathymetry, meteorology, lake/model/run config, initial
profiles, phosphorus/carbon boundary data) reuses the *unchanged*
numpy/pandas loader functions from `processBased_lakeModel_functions.py`
-- those only run once before the time loop and are not part of the
differentiable hot path, so there is no benefit to porting them.

This script lives in `src/`, next to `processBased_lakeModel_functions.py`
and `jax_lakeModel_functions.py`, and imports them directly (Python puts a
script's own directory on `sys.path` automatically) -- run it the same way
you would run `run_M3.py`:

    python src/run_M3_jax.py /path/to/Ravn [--steps N] [--temp-only]

`--steps N` truncates the run to the first N forcing steps (handy for
quick validation runs); omit it to run the full config-specified period.
`--temp-only` runs just the Phase-1 thermal engine (temperature held
`kd_light` constant, no O2/DOC/POC state) instead of the full model.
"""
import argparse
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from processBased_lakeModel_functions import (
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile,
    provide_phosphorus, provide_carbon, get_lake_config, get_model_params, get_run_config,
    get_ice_and_snow, get_num_data_columns,
)
from jax_lakeModel_functions import (
    run_temperature_model, run_full_model, default_params, default_geometry_wq,
)


def build_forcing_series(daily_meteo, times, wind_factor):
    """Reproduce exactly the `interp1d(..., kind='linear')` calls used in
    `run_wq_model()` (lines ~4259-4279 of processBased_lakeModel_functions.py)
    so the forcing seen by the JAX model matches the numpy model bit-for-bit."""

    def interp(col, factor=1.0):
        vals = daily_meteo[col].values * factor
        fillvals = tuple(vals[[0, -1]])
        f = interp1d(daily_meteo.dt.values, vals, kind="linear", fill_value=fillvals, bounds_error=False)
        return jnp.asarray(f(times))

    return dict(
        Jsw=interp("Shortwave_Radiation_Downwelling_wattPerMeterSquared"),
        Jlw=interp("Longwave_Radiation_Downwelling_wattPerMeterSquared"),
        Tair=interp("Air_Temperature_celsius"),
        ea=interp("ea"),
        Uw=interp("Ten_Meter_Elevation_Wind_Speed_meterPerSecond", factor=wind_factor),
        CC=interp("Cloud_Cover"),
        Pa=interp("Surface_Level_Barometric_Pressure_pascal"),
        RH=interp("Relative_Humidity_percent"),
        PP=interp("Precipitation_millimeterPerDay"),
    )


def add_wq_forcing_series(forcing, phosphorus_data, carbon_data, times):
    """TP and carbon-load forcing, matching `run_wq_model`'s own
    `interp1d` calls exactly (line ~4280-4286) -- note `carbon` uses
    `fill_value="extrapolate"`, unlike every other series above."""
    TP_fillvals = tuple(phosphorus_data.tp.values[[0, -1]])
    TP = interp1d(phosphorus_data.dt.values, phosphorus_data.tp.values, kind="linear",
                   fill_value=TP_fillvals, bounds_error=False)
    carbon = interp1d(carbon_data["dt"].values, carbon_data["hourly_carbon"].values, kind="linear",
                       fill_value="extrapolate", bounds_error=False)
    forcing = dict(forcing)
    forcing["TP"] = jnp.asarray(TP(times))
    forcing["carbon"] = jnp.asarray(carbon(times))
    return forcing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=".")
    parser.add_argument("--steps", type=int, default=None, help="truncate to first N forcing steps")
    parser.add_argument("--temp-only", action="store_true", help="run only the Phase-1 thermal engine")
    args = parser.parse_args()

    os.chdir(args.data_dir)

    num_lakes = get_num_data_columns("./lake_config.csv", "Zmax")
    lake_num = 1

    lake_config = get_lake_config("./lake_config.csv", lake_num)
    model_params = get_model_params("./model_params.csv", lake_num)
    run_config = get_run_config("./run_config.csv", lake_num)
    ice_and_snow = get_ice_and_snow("./ice_and_snow.csv", lake_num)

    windfactor = float(lake_config["WindSpeed"])
    nx = int(run_config["nx"])
    dt = float(run_config["dt"])
    dx = float(run_config["dx"])

    area, depth, volume, hypso_weight = get_hypsography(
        hypsofile=run_config["hypso_ini_file"], dx=dx, nx=nx, outflow_depth=float(lake_config["outflow_depth"])
    )

    desired_start = pd.Timestamp(run_config["start_time"])
    desired_end = pd.Timestamp(run_config["end_time"])
    startingDate = desired_start
    n_days = (desired_end - desired_start).days + (desired_end - desired_start).seconds / 86400
    hydrodynamic_timestep = 24 * dt
    total_runtime = n_days * hydrodynamic_timestep / dt
    endTime = 1 + total_runtime
    times_dt = pd.date_range(startingDate, desired_end, freq="h")
    n_steps = len(times_dt)

    meteo_all = provide_meteorology(
        meteofile=run_config["meteo_ini_file"], windfactor=windfactor, lat=lake_config["Latitude"],
        lon=lake_config["Longitude"], elev=lake_config["Elevation"], startDate=startingDate,
    )
    u_ini = initial_profile(initfile=run_config["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=startingDate)

    step_times = np.arange(1, n_steps * dt, dt)
    if args.steps is not None:
        step_times = step_times[: args.steps]

    forcing = build_forcing_series(meteo_all, step_times, windfactor)

    geometry = dict(
        area=jnp.asarray(area), depth=jnp.asarray(depth), volume=jnp.asarray(volume),
        dx=dx, dt=dt, latitude=float(lake_config["Latitude"]),
    )
    params = default_params(model_params, ice_and_snow)

    def _to_bool(x):
        # get_ice_and_snow() leaves non-numeric cells (e.g. "FALSE") as raw
        # strings, so a plain bool(x) would treat "FALSE" as truthy -- parse
        # the intended meaning explicitly instead.
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        return bool(x)

    ice_state = dict(
        ice=_to_bool(ice_and_snow["ice"]), Hi=ice_and_snow["Hi"], Hs=ice_and_snow["Hs"],
        Hsi=ice_and_snow["Hsi"], iceT=ice_and_snow["iceT"], rho_snow=ice_and_snow["rho_snow"],
    )

    u0 = jnp.asarray(deepcopy(u_ini), dtype=jnp.float64)

    if args.temp_only:
        run_fn = jax.jit(lambda u0: run_temperature_model(u0, forcing, geometry, params, ice_state=ice_state))

        print(f"Running JAX thermal engine (temp-only) for {len(step_times)} steps (nx={nx})...")
        t0 = time.time()
        final_state, u_all = run_fn(u0)
        u_all.block_until_ready()
        t1 = time.time()
        print(f"Done in {t1 - t0:.2f} s ({(t1 - t0) / len(step_times) * 1000:.3f} ms/step)")

        out_path = os.path.join(os.getcwd(), "res_lake1_jax_temp.npz")
        np.savez(out_path, temp=np.asarray(u_all), times=step_times, depth=np.asarray(depth))
        print(f"Saved temperature output to {out_path}")
        return

    # --- full model: temperature + water quality (O2, DOCr/DOCl, POCr/POCl) ---
    wq_ini = wq_initial_profile(
        initfile=run_config["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=startingDate,
    )
    tp_boundary = provide_phosphorus(
        tpfile=run_config["tp_ini_file"], startingDate=startingDate, startTime=1,
    )
    carbon_data = provide_carbon(
        ocloadfile=run_config["oc_load_file"], startingDate=startingDate, startTime=1,
    )
    carbon_data = carbon_data.dropna(subset=["oc"])

    forcing = add_wq_forcing_series(forcing, tp_boundary, carbon_data, step_times)

    mean_depth = float(np.sum(volume) / np.max(area))
    hydro_res_time_hr = float(model_params["hydro_res_time"]) * 8760
    geometry = default_geometry_wq(
        area=area, depth=depth, volume=volume, dx=dx, dt=dt, latitude=float(lake_config["Latitude"]),
        altitude=float(lake_config["Elevation"]), hypso_weight=hypso_weight, mean_depth=mean_depth,
        hydro_res_time_hr=hydro_res_time_hr,
    )

    o2_0 = jnp.asarray(deepcopy(wq_ini[0]), dtype=jnp.float64)
    docr_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.75, dtype=jnp.float64)
    docl_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.25, dtype=jnp.float64)
    pocr_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)
    pocl_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)

    run_fn = jax.jit(lambda u0: run_full_model(
        u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, forcing, geometry, params, ice_state=ice_state,
    ))

    print(f"Running full JAX lake model (temp + WQ) for {len(step_times)} steps (nx={nx})...")
    t0 = time.time()
    final_state, per_step = run_fn(u0)
    per_step["u"].block_until_ready()
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} s ({(t1 - t0) / len(step_times) * 1000:.3f} ms/step)")

    out_path = os.path.join(os.getcwd(), "res_lake1_jax_full.npz")
    np.savez(
        out_path,
        temp=np.asarray(per_step["u"]), o2=np.asarray(per_step["o2"]), docr=np.asarray(per_step["docr"]),
        docl=np.asarray(per_step["docl"]), pocr=np.asarray(per_step["pocr"]), pocl=np.asarray(per_step["pocl"]),
        times=step_times, depth=np.asarray(depth), volume=np.asarray(volume),
    )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
