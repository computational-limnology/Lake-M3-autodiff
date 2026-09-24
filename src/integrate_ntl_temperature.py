"""
Merge Mendota's high-frequency buoy thermistor-chain temperature record
(`ntl130_2_v14.csv`, NTL-LTER data package 130) into the manual-profile
temperature file `run_config.csv`'s `u_ini_file` names (`observedTemp.txt`
for Mendota), so that every script in this project that reads temperature
observations -- `calibrate_M3_jax.py`, `run_M3_mcl_jax.py`,
`plot_output.py`, `plot_mcl_kz.py`, `plot_ri_diagnostics.py`, and
`run_M3_calibrate_then_mcl.py` (which just runs the first two) -- gets ALL
of it (the sparse manual profiles AND the dense buoy record) by simply
reading the path `run_config.csv` already names, with no separate
NTL-specific loading logic needed anywhere else (see `observation_data.py`,
which is what those scripts now use). This is the Mendota counterpart of
`integrate_buoy_temperature.py` (Ravn's buoy JSON export); the two exist
separately because the two datasets' raw high-frequency formats have
nothing in common, but they share the same merge logic
(`merge_temperature_observations`, imported from that script) and the same
CLI conventions below.

This is a one-time (or run-again-when-the-NTL-export-is-updated) data-prep
step, not part of the modeling pipeline itself: run it once per dataset
before calibrating/training/plotting. If a dataset has no NTL file at all
(i.e. isn't Mendota), this is a no-op -- there is nothing to fold in, and
`u_ini_file` is used exactly as before.

Usage:
    python src/integrate_ntl_temperature.py Mendota
    python src/integrate_ntl_temperature.py Mendota --ntl-file ntl130_2_v14.csv
    python src/integrate_ntl_temperature.py Mendota --out Mendota/observedTemp_merged.txt
    python src/integrate_ntl_temperature.py Mendota --no-backup

By default, this overwrites the file named by `run_config.csv`'s
`u_ini_file` *in place*, after first saving a timestamped `.bak` copy --
the same convention `apply_calibration.py` uses for `model_params.csv` and
`integrate_buoy_temperature.py` uses for Ravn -- so `run_config.csv` itself
never needs to change, and the original is always one `cp` away from being
restored. Pass `--out PATH` to instead write the merged table to a
separate file (you would then need to point `run_config.csv`'s
`u_ini_file` at it yourself; the script reminds you of this when `--out`
is used).

Existing rows in `u_ini_file` are never modified or dropped: the merge is
strictly additive (NTL rows are appended, then everything is re-sorted by
time/depth), except that an exact `(datetime, Depth_meter)` duplicate
between the two sources keeps the *original* file's own value, on the
assumption that a manual/reference reading is more trustworthy than the
automated thermistor-chain record wherever they happen to coincide exactly
(in practice this essentially never happens -- the manual profile dates
are irregular while the buoy record is sub-daily).
"""
import argparse
import os
import shutil

import pandas as pd

from integrate_buoy_temperature import merge_temperature_observations
from observation_data import DEFAULT_NTL_FILE, load_ntl_high_frequency_temperature, load_run_config, resolve_data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".",
                         help="folder containing run_config.csv, the u_ini_file it names, and the NTL export")
    parser.add_argument("--lake", type=int, default=1, help="lake/column number in run_config.csv (default 1)")
    parser.add_argument("--ntl-file", type=str, default=DEFAULT_NTL_FILE,
                         help=f"NTL high-frequency export filename, relative to data_dir (default {DEFAULT_NTL_FILE})")
    parser.add_argument("--out", type=str, default=None,
                         help="write the merged table here instead of overwriting u_ini_file in place "
                              "(you'll need to point run_config.csv's u_ini_file at it yourself)")
    parser.add_argument("--no-backup", action="store_true",
                         help="skip writing a timestamped .bak backup before overwriting u_ini_file in "
                              "place (ignored if --out is given)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
