"""
Merge the high-frequency buoy thermistor-chain temperature record
(`ravn_2023.json`/`ravn_2024.json` + `sensor_level_buoy.sen`) into the
manual-profile temperature file `run_config.csv`'s `u_ini_file` names
(`L0001-HD.csv` for Ravn), so that every script in this project that reads
temperature observations -- `calibrate_M3_jax.py`, `run_M3_mcl_jax.py`,
`plot_output.py`, `plot_mcl_kz.py`, `plot_ri_diagnostics.py`, and
`run_M3_calibrate_then_mcl.py` (which just runs the first two) -- gets ALL
of it (the sparse manual profiles AND the dense hourly buoy record) by
simply reading the path `run_config.csv` already names, with no separate
buoy-JSON-specific loading logic needed anywhere else (see
`observation_data.py`, which is what those scripts now use).

This is a one-time (or run-again-when-new-buoy-exports-arrive) data-prep
step, not part of the modeling pipeline itself: run it once per dataset
before calibrating/training/plotting. If a dataset has no buoy JSON files
at all (i.e. isn't Ravn), this is a no-op -- there is nothing to fold in,
and `u_ini_file` is used exactly as before.

Usage:
    python src/integrate_buoy_temperature.py Ravn
    python src/integrate_buoy_temperature.py Ravn --json-files ravn_2023.json,ravn_2024.json
    python src/integrate_buoy_temperature.py Ravn --out Ravn/L0001-HD-merged.csv
    python src/integrate_buoy_temperature.py Ravn --no-backup

By default, this overwrites the file named by `run_config.csv`'s
`u_ini_file` *in place*, after first saving a timestamped `.bak` copy --
the same convention `apply_calibration.py` uses for `model_params.csv` --
so `run_config.csv` itself never needs to change, and the original is
always one `cp` away from being restored. Pass `--out PATH` to instead
write the merged table to a separate file (you would then need to point
`run_config.csv`'s `u_ini_file` at it yourself; the script reminds you of
this when `--out` is used).

Existing rows in `u_ini_file` are never modified or dropped: the merge is
strictly additive (buoy rows are appended, then everything is re-sorted by
time/depth), except that an exact `(datetime, Depth_meter)` duplicate
between the two sources keeps the *original* file's own value, on the
assumption that a manual/reference reading is more trustworthy than the
automated thermistor-chain mapping wherever they happen to coincide
exactly (in practice this essentially never happens -- the manual profile
dates are roughly monthly while the buoy record is hourly).
"""
import argparse
import os
import shutil

import pandas as pd

from observation_data import (
    DEFAULT_JSON_FILES, DEFAULT_SENSOR_FILE, load_buoy_temperature, load_run_config, resolve_data_path,
)


def merge_temperature_observations(original, buoy_long):
    """Concatenate the existing long-format temperature table `original`
    (as read from `u_ini_file`) with `buoy_long` (see
    `observation_data.load_buoy_temperature`), drop exact
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".",
                         help="folder containing run_config.csv, the u_ini_file it names, and the buoy files")
    parser.add_argument("--lake", type=int, default=1, help="lake/column number in run_config.csv (default 1)")
    parser.add_argument("--json-files", type=str, default=",".join(DEFAULT_JSON_FILES),
                         help="comma-separated buoy JSON export filenames, relative to data_dir "
                              f"(default {','.join(DEFAULT_JSON_FILES)})")
    parser.add_argument("--sensor-file", type=str, default=DEFAULT_SENSOR_FILE,
                         help=f"sensor->depth mapping file, relative to data_dir (default {DEFAULT_SENSOR_FILE})")
    parser.add_argument("--out", type=str, default=None,
                         help="write the merged table here instead of overwriting u_ini_file in place "
                              "(you'll need to point run_config.csv's u_ini_file at it yourself)")
    parser.add_argument("--no-backup", action="store_true",
                         help="skip writing a timestamped .bak backup before overwriting u_ini_file in "
                              "place (ignored if --out is given)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
