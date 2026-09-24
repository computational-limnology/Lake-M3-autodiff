"""
Normalize a CSV file's datetime column to one consistent
`YYYY-MM-DD HH:MM:SS` string format, in place.

Why this is needed: `processBased_lakeModel_functions.py` (kept untouched)
reads observation files with a bare `pd.to_datetime(obs['datetime'])` --
no `format=`/`errors=` argument -- in several places (`wq_initial_profile`,
`initial_profile`, `provide_meteorology`, ...). Pandas infers a single
format from the first few values and then requires every row to match it
exactly, so a file that mixes formats (e.g. most rows as
`2006-06-27 21:00:00` but some as just `2006-06-28`, with the implied
midnight time dropped) crashes with a `ValueError` the first time it hits
a row that doesn't match the inferred format -- this is exactly what
happened with Mendota's `ME_obs_depths3_wpoc.csv` (and the same issue is
present in `ME_obs_depths.csv`, `ME_obs_depths2.csv`, and
`ME_obs_depths4_wpoc.hfdo.csv` -- apparently a shared quirk of however
this whole file family gets exported). This script fixes the file at the
source so every downstream reader -- the untouched original functions
above, and this project's own `observation_data.py` -- parses it the same
way, rather than patching each reader separately.

Usage:
    python src/fix_datetime_format.py Mendota/ME_obs_depths3_wpoc.csv
    python src/fix_datetime_format.py Mendota/ME_obs_depths3_wpoc.csv --column datetime
    python src/fix_datetime_format.py Mendota/ME_obs_depths3_wpoc.csv --out Mendota/ME_obs_depths3_wpoc_fixed.csv
    python src/fix_datetime_format.py Mendota/ME_obs_depths3_wpoc.csv --no-backup

By default, this overwrites the file *in place*, after first saving a
timestamped `.bak` copy -- the same convention `apply_calibration.py`/
`integrate_buoy_temperature.py`/`integrate_ntl_temperature.py` use
elsewhere in this project. Pass `--out PATH` to instead write the cleaned
table to a separate file.

Rows whose datetime string can't be parsed at all (not merely
inconsistently formatted -- genuinely missing or malformed) are dropped;
the script reports how many.
"""
import argparse
import os
import shutil

import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="CSV file to normalize")
    parser.add_argument("--column", type=str, default="datetime", help="datetime column name (default 'datetime')")
    parser.add_argument("--out", type=str, default=None,
                         help="write the cleaned table here instead of overwriting path in place")
    parser.add_argument("--no-backup", action="store_true",
                         help="skip writing a timestamped .bak backup before overwriting in place "
                              "(ignored if --out is given)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
