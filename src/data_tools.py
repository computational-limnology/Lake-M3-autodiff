"""
Data-preparation and observation-loading utilities, collected in one module
(formerly `observation_data.py`, `fix_datetime_format.py`,
`fix_missing_do_surface.py`, `integrate_buoy_temperature.py` and
`integrate_ntl_temperature.py`).

Import it where needed:
    from data_tools import load_run_config, load_temperature_dataframe, load_water_quality_dataframe

Run the one-off data-prep steps through its subcommands:
    python src/data_tools.py fix-datetime    Mendota/ME_obs_depths3_wpoc.csv [--column datetime] [--out PATH] [--no-backup]
    python src/data_tools.py fix-surface     Mendota/ME_obs_depths3_wpoc.csv [--variable do|poc] [--out PATH] [--no-backup]
    python src/data_tools.py integrate-buoy  Ravn    [--json-files a.json,b.json] [--sensor-file F] [--out PATH] [--no-backup]
    python src/data_tools.py integrate-ntl   Mendota [--ntl-file F] [--out PATH] [--no-backup]

Sections
--------
0. Project layout helpers: mcl_path (MCL weights/results in <project>/mcl/
   with a _<lake> suffix), lake_dir_for_result, lake_tag.
1. Observation loading (all readers resolve file names through
   `run_config.csv`): load_run_config, resolve_data_path,
   load_temperature_dataframe, load_water_quality_dataframe, plus the raw
   high-frequency loaders load_buoy_temperature (Ravn JSON chain) and
   load_ntl_high_frequency_temperature (Mendota NTL-LTER 130 export).
2. Datetime normalisation (fix-datetime): `processBased_lakeModel_functions`
   parses observation files with a bare `pd.to_datetime`, which fails on
   files that mix `YYYY-MM-DD HH:MM:SS` and date-only rows (Mendota's
   ME_obs_depths* exports). Rewrites the column in one format; rows that
   cannot be parsed are dropped.
3. Synthetic surface rows (fix-surface): adds a depth = 0 m row per sample
   datetime of a variable that lacks one, copying its shallowest reading.
   Needed for 'do' (wq_initial_profile interpolates from 0 m) and for
   single-depth variables such as Mendota's 'poc' (the calibration loader
   needs >= 2 depths per profile).
4. High-frequency temperature merges (integrate-buoy / integrate-ntl): fold
   a buoy record into the `u_ini_file` named by run_config.csv, so every
   script reads one temperature file. The merge is additive; on an exact
   (datetime, depth) duplicate the original file's value is kept.

All in-place writers save a `.bak` copy first unless --no-backup is given;
--out writes elsewhere instead (run_config.csv is then not updated).
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd



# ===========================================================================
# 0. Project layout: lake folders and the shared mcl/ folder
# ===========================================================================
# MCL artefacts (trained weights `*.pkl` and training results `*.npz`) live
# in <project>/mcl/, not inside the lake folders, with the lake folder name
# appended as a suffix (mcl_nn_params_Ravn.pkl, mcl_result_Mendota.npz,
# mcl_add_Ravn.pkl, ...), so weights trained on one lake can be found and
# read from any script or folder.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCL_DIR = os.path.join(PROJECT_ROOT, "mcl")


def lake_tag(data_dir):
    """Lake name used as the mcl/ file suffix: the lake folder's name."""
    return os.path.basename(os.path.normpath(os.path.abspath(data_dir)))


def mcl_path(name, data_dir, base_dir=None, reading=False):
    """Resolve an MCL weights/result file name.

    * A bare file name (no directory part), e.g. the defaults
      "mcl_nn_params.pkl"/"mcl_result.npz" or "mcl_add.pkl", goes to
      <project>/mcl/<stem>_<lake><ext>; the suffix is not added twice if
      the name already ends in _<lake>.
    * A name with a directory part (or an absolute path) is used as given,
      relative to `base_dir` (default: the current working directory), so
      any explicit location still works.
    * reading=True (loading weights): a bare name that already exists in
      mcl/ as given (e.g. "mcl_rbf9_Ravn.pkl" while running another lake) is
      used directly, so other lakes' weights can be named without a path."""
    if os.path.isabs(name):
        return name
    if os.path.dirname(name):
        return os.path.abspath(os.path.join(base_dir or os.getcwd(), name))
    if reading and os.path.exists(os.path.join(MCL_DIR, name)) and not os.path.exists(
            os.path.join(MCL_DIR, "{}_{}{}".format(*os.path.splitext(name)[:1], lake_tag(data_dir), os.path.splitext(name)[1]))):
        return os.path.join(MCL_DIR, name)
    stem, ext = os.path.splitext(name)
    tag = lake_tag(data_dir)
    if not stem.endswith("_" + tag):
        stem = f"{stem}_{tag}"
    return os.path.join(MCL_DIR, stem + ext)


def lake_dir_for_result(result_path, explicit=None):
    """Lake data folder (holding run_config.csv) belonging to a result file.

    `explicit` wins if given. Otherwise: the result's own folder if it holds
    a run_config.csv (results written inside a lake folder); else, for files
    in mcl/, the lake named by the file's `_<lake>` suffix (any folder in the
    project root with a run_config.csv). Falls back to the result's folder."""
    if explicit:
        return os.path.abspath(explicit)
    path = os.path.abspath(result_path)
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.exists(os.path.join(folder, "run_config.csv")):
        return folder
    if not os.path.isdir(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        lakes = sorted(
            (d for d in os.listdir(PROJECT_ROOT)
             if os.path.exists(os.path.join(PROJECT_ROOT, d, "run_config.csv"))),
            key=len, reverse=True,
        )
        for lake in lakes:
            if stem.endswith("_" + lake):
                return os.path.join(PROJECT_ROOT, lake)
    return folder


# ===========================================================================
# 1. Observation loading
# ===========================================================================

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
    # `data_tools.py integrate-buoy`) without needing every dependency
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
    included, run `data_tools.py integrate-buoy` once first to fold it
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

    Used by `data_tools.py integrate-buoy` to fold this record into
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

# ===========================================================================
# 2. Datetime normalisation (fix-datetime)
# ===========================================================================

def normalize_datetime_column(df, column="datetime"):
    """Return `(cleaned_df, n_dropped)` with `column` reformatted to a
    single consistent `YYYY-MM-DD HH:MM:SS` string, parsing whatever mix
    of formats pandas' `format="mixed"` can make sense of (e.g. full
    timestamps alongside date-only strings, which are treated as
    midnight). Rows that still fail to parse are dropped."""
    parsed = pd.to_datetime(df[column], format="mixed", errors="coerce")
    n_dropped = int(parsed.isna().sum())
    cleaned = df.loc[parsed.notna()].copy()
    cleaned[column] = parsed.loc[parsed.notna()].dt.strftime("%Y-%m-%d %H:%M:%S")
    return cleaned, n_dropped


def _cli_fix_datetime(args):

    if not os.path.exists(args.path):
        raise SystemExit(f"{args.path} does not exist.")

    df = pd.read_csv(args.path)
    if args.column not in df.columns:
        raise SystemExit(f"{args.path} has no '{args.column}' column -- available: {list(df.columns)}")

    cleaned, n_dropped = normalize_datetime_column(df, args.column)

    if args.out:
        out_path = args.out
    else:
        out_path = args.path
        if not args.no_backup:
            backup_path = args.path + ".bak"
            shutil.copy(args.path, backup_path)
            print(f"Backed up original to {backup_path}")

    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned)} rows ({n_dropped} dropped -- unparseable '{args.column}' values) to {out_path}")

# ===========================================================================
# 3. Synthetic surface rows (fix-surface)
# ===========================================================================

def add_missing_surface_rows(df, variable="do"):
    """Return `(df_with_added_rows, n_added)`. For every distinct
    `datetime` that has a `variable == variable` row but none at
    `depth == 0`, add one: a copy of that datetime's shallowest existing
    row (for that variable) with `depth` set to 0."""
    var_rows = df[df["variable"] == variable]
    has_zero = set(var_rows.loc[var_rows["depth"] == 0, "datetime"])
    needs_zero = var_rows[~var_rows["datetime"].isin(has_zero)]
    if needs_zero.empty:
        return df, 0

    shallowest_idx = needs_zero.groupby("datetime")["depth"].idxmin()
    synthetic = df.loc[shallowest_idx].copy()
    synthetic["depth"] = 0.0
    return pd.concat([df, synthetic], ignore_index=True), len(synthetic)


# Backward-compatible alias for the original 'do'-only name.
def add_missing_do_surface_rows(df):
    return add_missing_surface_rows(df, variable="do")


def _cli_fix_surface(args):

    if not os.path.exists(args.path):
        raise SystemExit(f"{args.path} does not exist.")

    df = pd.read_csv(args.path)
    required = {"datetime", "depth", "variable", "observation"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{args.path} is missing required columns: {sorted(missing)}")

    fixed, n_added = add_missing_surface_rows(df, variable=args.variable)

    if args.out:
        out_path = args.out
    else:
        out_path = args.path
        if not args.no_backup:
            backup_path = args.path + ".bak"
            shutil.copy(args.path, backup_path)
            print(f"Backed up original to {backup_path}")

    fixed.to_csv(out_path, index=False)
    print(f"Added {n_added} synthetic depth=0 '{args.variable}' rows (one per sample datetime that lacked one) -- "
          f"wrote {len(fixed)} rows to {out_path}")

# ===========================================================================
# 4. High-frequency temperature merges (integrate-buoy / integrate-ntl)
# ===========================================================================

def merge_temperature_observations(original, buoy_long):
    """Concatenate the existing long-format temperature table `original`
    (as read from `u_ini_file`) with `buoy_long` (see
    `data_tools.load_buoy_temperature`), drop exact
    `(datetime, Depth_meter)` duplicates keeping `original`'s own row, and
    return the merged, time-sorted DataFrame with the same 3-column schema
    both inputs share."""
    required = {"datetime", "Depth_meter", "Water_Temperature_celsius"}
    missing = required.difference(original.columns)
    if missing:
        raise ValueError(f"Existing temperature file is missing required columns: {sorted(missing)}")
    original = original[["datetime", "Depth_meter", "Water_Temperature_celsius"]].copy()
    original["datetime"] = pd.to_datetime(original["datetime"])

    merged = pd.concat([original, buoy_long], ignore_index=True)
    merged = merged.sort_values(["datetime", "Depth_meter"], kind="stable")
    merged = merged.drop_duplicates(subset=["datetime", "Depth_meter"], keep="first")
    return merged.reset_index(drop=True)


def _cli_integrate_buoy(args):

    run_config = load_run_config(args.data_dir, args.lake)
    hd_path = resolve_data_path(args.data_dir, run_config, "u_ini_file", "L0001-HD.csv")
    if not os.path.exists(hd_path):
        raise SystemExit(f"{hd_path} (u_ini_file) does not exist -- nothing to merge into.")

    json_files = tuple(f.strip() for f in args.json_files.split(",") if f.strip())
    buoy_long = load_buoy_temperature(args.data_dir, json_files=json_files, sensor_file=args.sensor_file)
    if buoy_long is None:
        print(f"No buoy data found ({', '.join(json_files)} / {args.sensor_file} not present in "
              f"{args.data_dir}) -- nothing to merge. {hd_path} left untouched.")
        return

    original = pd.read_csv(hd_path)
    merged = merge_temperature_observations(original, buoy_long)
    added_n = len(merged) - len(original)

    if args.out:
        out_path = args.out
    else:
        out_path = hd_path
        if not args.no_backup:
            backup_path = hd_path + ".bak"
            shutil.copy(hd_path, backup_path)
            print(f"Backed up original to {backup_path}")

    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows ({len(original)} original + {added_n} new from the buoy record) "
          f"to {out_path}")
    if args.out:
        print(f"Note: run_config.csv's u_ini_file still points at {hd_path} -- update it to "
              f"'{os.path.relpath(out_path, args.data_dir)}' if you want the rest of the pipeline "
              "to pick up the merged file.")


def _cli_integrate_ntl(args):

    run_config = load_run_config(args.data_dir, args.lake)
    u_path = resolve_data_path(args.data_dir, run_config, "u_ini_file", "observedTemp.txt")
    if not os.path.exists(u_path):
        raise SystemExit(f"{u_path} (u_ini_file) does not exist -- nothing to merge into.")

    ntl_long = load_ntl_high_frequency_temperature(args.data_dir, filename=args.ntl_file)
    if ntl_long is None:
        print(f"No NTL high-frequency file found ({args.ntl_file} not present in {args.data_dir}) -- "
              f"nothing to merge. {u_path} left untouched.")
        return

    original = pd.read_csv(u_path)
    merged = merge_temperature_observations(original, ntl_long)
    added_n = len(merged) - len(original)

    if args.out:
        out_path = args.out
    else:
        out_path = u_path
        if not args.no_backup:
            backup_path = u_path + ".bak"
            shutil.copy(u_path, backup_path)
            print(f"Backed up original to {backup_path}")

    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows ({len(original)} original + {added_n} new from the NTL high-frequency "
          f"record) to {out_path}")
    if args.out:
        print(f"Note: run_config.csv's u_ini_file still points at {u_path} -- update it to "
              f"'{os.path.relpath(out_path, args.data_dir)}' if you want the rest of the pipeline "
              "to pick up the merged file.")

# ===========================================================================
# Command line
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fix-datetime", help="normalise a CSV datetime column to YYYY-MM-DD HH:MM:SS (in place)")
    p.add_argument("path", help="CSV file to normalize")
    p.add_argument("--column", type=str, default="datetime", help="datetime column name (default 'datetime')")
    p.add_argument("--out", type=str, default=None, help="write here instead of overwriting path in place")
    p.add_argument("--no-backup", action="store_true", help="skip the .bak backup (ignored with --out)")
    p.set_defaults(func=_cli_fix_datetime)

    p = sub.add_parser("fix-surface", help="add synthetic depth=0 m rows for a WQ variable (in place)")
    p.add_argument("path", help="water-quality observations CSV (datetime, depth, variable, observation)")
    p.add_argument("--variable", type=str, default="do", help="variable to fix (default 'do'; e.g. 'poc')")
    p.add_argument("--out", type=str, default=None, help="write here instead of overwriting path in place")
    p.add_argument("--no-backup", action="store_true", help="skip the .bak backup (ignored with --out)")
    p.set_defaults(func=_cli_fix_surface)

    p = sub.add_parser("integrate-buoy", help="merge Ravn buoy JSON temperature into u_ini_file")
    p.add_argument("data_dir", nargs="?", default=".", help="lake folder with run_config.csv and buoy files")
    p.add_argument("--lake", type=int, default=1, help="lake/column number in run_config.csv (default 1)")
    p.add_argument("--json-files", type=str, default=",".join(DEFAULT_JSON_FILES),
                   help=f"comma-separated buoy JSON files in data_dir (default {','.join(DEFAULT_JSON_FILES)})")
    p.add_argument("--sensor-file", type=str, default=DEFAULT_SENSOR_FILE,
                   help=f"sensor->depth mapping file in data_dir (default {DEFAULT_SENSOR_FILE})")
    p.add_argument("--out", type=str, default=None, help="write here instead of overwriting u_ini_file")
    p.add_argument("--no-backup", action="store_true", help="skip the .bak backup (ignored with --out)")
    p.set_defaults(func=_cli_integrate_buoy)

    p = sub.add_parser("integrate-ntl", help="merge Mendota NTL-LTER high-frequency temperature into u_ini_file")
    p.add_argument("data_dir", nargs="?", default=".", help="lake folder with run_config.csv and the NTL export")
    p.add_argument("--lake", type=int, default=1, help="lake/column number in run_config.csv (default 1)")
    p.add_argument("--ntl-file", type=str, default=DEFAULT_NTL_FILE,
                   help=f"NTL export filename in data_dir (default {DEFAULT_NTL_FILE})")
    p.add_argument("--out", type=str, default=None, help="write here instead of overwriting u_ini_file")
    p.add_argument("--no-backup", action="store_true", help="skip the .bak backup (ignored with --out)")
    p.set_defaults(func=_cli_integrate_ntl)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
