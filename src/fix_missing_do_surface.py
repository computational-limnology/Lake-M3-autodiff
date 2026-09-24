"""
Add a synthetic depth=0 m row for every sample datetime of a given
water-quality `variable` (default 'do', dissolved oxygen) that doesn't
already have one, copying that datetime's shallowest available reading of
that variable.

Why: `processBased_lakeModel_functions.wq_initial_profile()` (kept
untouched) already does exactly this for 'doc' -- its own comment reads
"assumed mixed epi, if no 0m available then it pulls from shallowest
option" -- but the equivalent block is missing for 'do' a few lines
above. So whenever the 'do' sample nearest a run's start_time happens to
be a single, non-zero-depth reading -- very common in Mendota's
ME_obs_depths* exports, where most sample datetimes only have ONE depth
(typically 0.5 m, from a high-frequency single-sensor buoy reading)
rather than a full manual profile -- `wq_initial_profile()`'s
interpolation fails with "A value (0.0) in x_new is below the
interpolation range's minimum value". This script applies the exact same
assumption the original code already encodes for 'doc' to 'do' (and, via
`--variable`, to any other single-depth-per-timestamp variable) too, but
as a one-time data fix (since the reference model code must stay
untouched) rather than a code change.

This same single-depth-per-timestamp shape also breaks
`calibrate_M3_jax.py`'s observation loader for a *different* reason:
`load_observations()`'s `profile_to_grid()` needs at least 2 distinct
depths per sample datetime to build an `interp1d` profile at all (with
only 1, it can't tell which grid cells that single reading should apply
to, so it skips the date entirely rather than guessing) -- this is what
made Mendota's 'poc' rows (single depth, 1.0 m, at every hourly
timestamp) score 0 profile dates in calibration despite the file having
tens of thousands of 'poc' rows. Running this script with
`--variable poc` fixes that the same way it fixes 'do': one synthetic
depth=0 m row per timestamp, so `interp1d` has 2 points ([0, 1.0] m here)
and at least the near-surface grid cells inside that range get scored.

Usage:
    python src/fix_missing_do_surface.py Mendota/ME_obs_depths3_wpoc.csv
    python src/fix_missing_do_surface.py Mendota/ME_obs_depths3_wpoc.csv --variable poc
    python src/fix_missing_do_surface.py Mendota/ME_obs_depths3_wpoc.csv --out Mendota/ME_obs_depths3_wpoc_fixed.csv
    python src/fix_missing_do_surface.py Mendota/ME_obs_depths3_wpoc.csv --no-backup

By default, this overwrites the file *in place*, after first saving a
timestamped `.bak` copy -- the same convention used elsewhere in this
project. Pass `--no-backup` if you already have a trusted backup from an
earlier step (e.g. right after `fix_datetime_format.py`, so as not to
clobber that one's `.bak` with an already-modified version). Existing rows
are never modified; only missing depth=0 rows (of the selected `--variable`)
are added, one per sample datetime that needs one. Multiple variables can
be fixed by running this script again with a different `--variable` --
each run's own `.bak` only captures that run's starting state, so run it
once per variable in sequence (each backing up the previous run's output)
rather than trying to fix several variables in one pass.
"""
import argparse
import os
import shutil

import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="water-quality observations CSV to fix (datetime, depth, variable, observation columns)")
    parser.add_argument("--variable", type=str, default="do",
                         help="which 'variable' column value to fix (default 'do'; e.g. 'poc' for "
                              "Mendota's single-depth-per-timestamp POC rows)")
    parser.add_argument("--out", type=str, default=None,
                         help="write the fixed table here instead of overwriting path in place")
    parser.add_argument("--no-backup", action="store_true",
                         help="skip writing a timestamped .bak backup before overwriting in place "
                              "(ignored if --out is given)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
