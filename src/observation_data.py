"""
Shared helpers for locating and loading observed data (temperature,
dissolved oxygen, dissolved organic carbon) that `calibrate_M3_jax.py`,
`run_M3_mcl_jax.py`, `plot_output.py`, `plot_mcl_kz.py`,
`plot_ri_diagnostics.py` and `integrate_buoy_temperature.py` all rely on --
this exists so there is exactly one implementation of each, rather than
several near-identical copies (there used to be two independent copies of
the buoy-JSON loader alone, one in `calibrate_M3_jax.py` and one in
`plot_output.py`, already drifting apart in their docstrings).

Two kinds of file are involved for a dataset like Ravn, and this module's
job is to make callers agnostic to which files actually exist for a given
dataset by always going through `run_config.csv`:

  * The manual-profile files `run_config.csv` already names -- `u_ini_file`
    (temperature, e.g. `L0001-HD.csv`) and `wq_ini_file` (dissolved oxygen
    "do" / dissolved organic carbon "doc", e.g. `L0001-WQ.csv`) -- both a
    "long" format of (datetime, depth, value) rows, sparse and irregular
    in time (roughly monthly profiles for Ravn).
  * A separate, much higher-frequency buoy thermistor-chain export,
    present only for some datasets, in whatever raw format that dataset's
    provider uses -- Ravn's is JSON files plus a sensor->depth mapping
    file (`load_buoy_temperature`); Mendota's is a single NTL-LTER CSV
    export (`load_ntl_high_frequency_temperature`). `integrate_buoy_temperature.py`/
    `integrate_ntl_temperature.py` merge these into `u_ini_file` directly
    -- see those scripts' module docstrings -- so every reader below
    needs nothing high-frequency-format-specific at all: `u_ini_file`
    already has everything once the relevant merge script has been run.
    The two `load_*` functions above are kept here (rather than deleted)
    only because those merge scripts need them.
"""
import json
import os

import numpy as np
import pandas as pd

DEFAULT_JSON_FILES = ("ravn_2023.json", "ravn_2024.json")
DEFAULT_SENSOR_FILE = "sensor_level_buoy.sen"
DEFAULT_NTL_FILE = "ntl130_2_v14.csv"


def load_run_config(data_dir, lake_num=1, run_config_path=None):
    """`processBased_lakeModel_functions.get_run_config()`, resolved against
    `data_dir` by default. Returns `None` (rather than raising) if the file
    doesn't exist -- every caller in this module treats a missing
    `run_config.csv` as "fall back to the hardcoded default filename", not
    an error, so datasets that predate this file (or that never adopt it)
    keep working exactly as before."""
    # Imported lazily to keep this module importable (e.g. from
    # integrate_buoy_temperature.py) without needing every dependency
    # processBased_lakeModel_functions.py itself pulls in.
    from processBased_lakeModel_functions import get_run_config

    path = run_config_path or os.path.join(data_dir, "run_config.csv")
    if not os.path.exists(path):
        return None
    return get_run_config(path, lake_num)


def resolve_data_path(data_dir, run_config, key, default_filename):
    """Resolve `run_config[key]` (e.g. `"./L0001-HD.csv"`, exactly as
    written in `run_config.csv`) against `data_dir`, regardless of the
    process's current working directory or the exact relative-path
    spelling used there. Falls back to `default_filename` if `run_config`
    is `None` or doesn't have that key (no `run_config.csv`, or an older
    one missing a field), so callers work the same as before this existed
    for a dataset that isn't set up this way."""
    if run_config is not None and key in run_config.index:
        name = os.path.basename(str(run_config[key]))
    else:
        name = default_filename
    return os.path.join(data_dir, name)


def load_temperature_dataframe(data_dir, run_config=None):
    """Read `run_config`'s `u_ini_file` (falling back to `L0001-HD.csv`) as
    a long-format `(datetime, Depth_meter, Water_Temperature_celsius)`
    DataFrame -- the single source of temperature observations for every
    script in this project. If the high-frequency buoy record should be
    included, run `integrate_buoy_temperature.py` once first to fold it
    into this same file; this function does not look for the buoy JSON
    files itself. Returns `None` if the file doesn't exist."""
    path = resolve_data_path(data_dir, run_config, "u_ini_file", "L0001-HD.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"datetime", "Depth_meter", "Water_Temperature_celsius"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "Depth_meter", "Water_Temperature_celsius"]]


def load_water_quality_dataframe(data_dir, run_config=None):
    """Read `run_config`'s `wq_ini_file` (falling back to `L0001-WQ.csv`)
    as-is (`datetime, depth, observation, variable, ...`) -- the single
    source of dissolved-oxygen/DOC observations for every script in this
    project. Returns `None` if the file doesn't exist."""
    path = resolve_data_path(data_dir, run_config, "wq_ini_file", "L0001-WQ.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_buoy_temperature(
    data_dir,
    json_files=DEFAULT_JSON_FILES,
    sensor_file=DEFAULT_SENSOR_FILE,
):
    """Load high-frequency buoy thermistor-chain temperature data from the
    raw JSON exports (`ravn_2023.json`/`ravn_2024.json` by default), if
    present, average it down to hourly values, and reshape it to the same
    long-format schema `load_temperature_dataframe()`/`L0001-HD.csv` use
    (`datetime`, `Depth_meter`, `Water_Temperature_celsius`).

    Each JSON file has the shape
    `{"data": {"tempCableReadings": {"timestamps": [...], "t1": [...],
    ..., "t18": [...]}}}`, with `timestamps` as Unix epoch seconds (native
    sampling is irregular, roughly every 15 minutes) and `t1..t18` the
    temperature (degC) at each of the 18 thermistor depths (looked up from
    `sensor_file`, whitespace-delimited with columns `sensor, position,
    specification`, where `position` is that sensor's depth in meters).
    Native readings are averaged into hourly bins (`resample("1h").mean()`)
    before being reshaped to long format, since the native ~15-minute
    cadence is finer than the model's hourly timestep. Any of `json_files`
    that is missing is skipped; returns `None` if none are found, or if
    `sensor_file` is missing (e.g. for datasets other than Ravn), so this
    is a no-op unless the buoy files are actually present.

    The returned rows also include a synthetic depth=0 m entry per
    timestamp, copying the shallowest sensor's reading (1 m for Ravn's
    chain, which has no sensor above that) -- see the inline comment above
    the surface-extension code for why this is needed.

    Used by `integrate_buoy_temperature.py` to fold this record into
    `u_ini_file` -- see that script's module docstring. Nothing else in
    this project should need to call this directly any more; read
    temperature observations via `load_temperature_dataframe()` instead."""
    sensor_path = os.path.join(data_dir, sensor_file)
    if not os.path.exists(sensor_path):
        return None
    sensors = pd.read_csv(sensor_path, sep=r"\s+")
    depth_by_sensor = dict(zip(sensors["sensor"].astype(int), sensors["position"].astype(float)))

    hourly_frames = []
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            payload = json.load(f)
        readings = payload["data"]["tempCableReadings"]
        sensor_cols = [k for k in readings if k != "timestamps"]
        wide = pd.DataFrame(
            {col: readings[col] for col in sensor_cols},
            index=pd.to_datetime(np.asarray(readings["timestamps"], dtype="int64"), unit="s"),
        ).sort_index()
        hourly_frames.append(wide.resample("1h").mean())

    if not hourly_frames:
        return None

    hourly = pd.concat(hourly_frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    hourly.index.name = "datetime"

    long = hourly.reset_index().melt(
        id_vars="datetime", var_name="sensor", value_name="Water_Temperature_celsius"
    )
    long["Depth_meter"] = long["sensor"].str[1:].astype(int).map(depth_by_sensor)
    long = long.dropna(subset=["Depth_meter", "Water_Temperature_celsius"])
    long = long[["datetime", "Depth_meter", "Water_Temperature_celsius"]]

    # Extend to the surface (0 m). The shallowest sensor sits at
    # min(depth_by_sensor.values()) (1 m for Ravn's 18-sensor chain -- see
    # sensor_level_buoy.sen), but the model grid needs values down to a few
    # tenths of a meter, and `initial_profile()`/`wq_initial_profile()` in
    # processBased_lakeModel_functions.py (kept untouched) pick whichever
    # single u_ini_file timestamp is nearest the run's start date and
    # interpolate over only that timestamp's depths -- so once buoy rows
    # are merged into u_ini_file, any start date inside the buoy coverage
    # period would otherwise land on an hourly buoy-only row lacking a
    # near-surface value and crash that interpolation. Add a synthetic
    # depth=0 m row per timestamp that copies the shallowest sensor's
    # reading, assuming a well-mixed near-surface layer -- the same
    # assumption implicit in the manual profile file's own 0 m samples.
    shallowest_depth = min(depth_by_sensor.values())
    surface = long[long["Depth_meter"] == shallowest_depth].copy()
    surface["Depth_meter"] = 0.0
    return pd.concat([long, surface], ignore_index=True)


def load_ntl_high_frequency_temperature(data_dir, filename=DEFAULT_NTL_FILE):
    """Load Mendota's high-frequency buoy thermistor-chain temperature
    record from the NTL-LTER data package 130 export (`ntl130_2_v14.csv`
    by default -- "North Temperate Lakes LTER: High Frequency Water
    Temperature Data - Lake Mendota Buoy"), and reshape it to the same
    long-format schema `load_temperature_dataframe()`/`observedTemp.txt`
    use (`datetime`, `Depth_meter`, `Water_Temperature_celsius`).

    Columns used: `sampledate` (`YYYY-MM-DD`) + `hour` (`HHMM` as an int,
    e.g. `2100` = 21:00) combine into `datetime` -- note this dataset's
    convention of `hour == 2400` meaning the *following* day's 00:00 falls
    out for free from plain minute arithmetic (`2400` -> 1440 minutes ->
    a full day added), no special-casing needed. `depth` (meters, already
    real depths -- no sensor->depth mapping file needed, unlike Ravn's
    buoy chain) -> `Depth_meter`; `wtemp` (degC) ->
    `Water_Temperature_celsius`. Rows with a missing `wtemp` are dropped.
    `flag_wtemp` (NTL's per-reading data-quality flag) is not filtered on
    here -- this is deliberately the simple version; drop flagged rows
    yourself first if you want stricter QC.

    This dataset also logs a `hour == 0` reading for the day *after* most
    `hour == 2400` rows (8760 of 8783 in the 2006-2025 export), both
    intended as the same midnight moment and only ~0.03 degC apart on
    median -- i.e. the same collision `load_buoy_temperature()`'s surface
    extension avoids for a different reason, here arising from the source
    data itself rather than anything this function adds. Resolved by
    keeping the `hour == 0` reading whenever both land on the same
    `datetime`/`Depth_meter` (an arbitrary but deterministic tiebreak,
    rather than leaving it to accidental sort order); the rarer standalone
    `hour == 2400` reading with no next-day match is kept as-is.

    Returns `None` if `filename` isn't present in `data_dir`."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, usecols=["sampledate", "hour", "depth", "wtemp"])
    df = df.dropna(subset=["wtemp"])

    minutes = (df["hour"] // 100) * 60 + (df["hour"] % 100)
    datetime = pd.to_datetime(df["sampledate"]) + pd.to_timedelta(minutes, unit="m")

    long = pd.DataFrame({
        "datetime": datetime,
        "Depth_meter": df["depth"].astype(float),
        "Water_Temperature_celsius": df["wtemp"].astype(float),
        "_hour": df["hour"].values,
    })
    long = long.sort_values(["datetime", "_hour"], kind="stable")
    long = long.drop_duplicates(subset=["datetime", "Depth_meter"], keep="first")
    return long.drop(columns="_hour").reset_index(drop=True)
