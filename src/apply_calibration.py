"""
Write calibrate_M3_jax.py's calibration_result.csv back into model_params.csv.

Usage:
    python src/apply_calibration.py Ravn
    python src/apply_calibration.py Ravn --lake 1
    python src/apply_calibration.py Ravn --no-backup

For every parameter listed in `calibration_result.csv` (parameter,initial,
calibrated columns -- exactly what `calibrate_M3_jax.py` writes), overwrites
that parameter's value in `model_params.csv`'s "Lake<N>" column with the
"calibrated" value. Every other row/column in `model_params.csv` -- every
parameter that wasn't calibrated, and the Description/Units columns -- is
left completely untouched.

This edits `model_params.csv` as plain text (via the `csv` module) rather
than round-tripping it through `pandas.read_csv`/`to_csv`: reading the whole
file into a DataFrame and writing it back out reformats every cell pandas
touches (scientific-notation case, trailing zeros, blank cells becoming the
literal text "None", ...), which would turn a 3-parameter change into a
much larger, spurious diff across the whole file. Editing only the specific
cells that changed keeps the rest of the file byte-for-byte identical.

A backup (`model_params.csv.bak`) is written first by default, since this
overwrite is otherwise only undoable via git or by hand.
"""
import argparse
import csv
import os
import shutil

import pandas as pd


def _is_numeric(cell):
    try:
        float(cell)
        return True
    except ValueError:
        return False


def update_model_params_from_calibration(model_params_file, calibration_result_file, lake_num=1, backup=True):
    """Overwrite the "Lake<lake_num>" column of `model_params_file` with the
    "calibrated" values from `calibration_result_file`. Returns a list of
    (parameter, old_value_str, new_value_str) tuples for whatever actually
    changed (parameters already equal to their calibrated value are
    skipped). Raises ValueError if `calibration_result_file` names a
    parameter that isn't a row in `model_params_file`.
    """
    cal_df = pd.read_csv(calibration_result_file, index_col="parameter")

    with open(model_params_file, "rb") as f:
        had_trailing_newline = f.read().endswith((b"\n", b"\r"))

    with open(model_params_file, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    row_by_name = {row[0]: row for row in body}

    missing = [p for p in cal_df.index if p not in row_by_name]
    if missing:
        raise ValueError(
            f"{calibration_result_file} lists parameters not found in "
            f"{model_params_file}: {missing}"
        )

    # Same convention get_model_params()/get_run_config() use elsewhere in
    # this project: values start in the first column whose entry (for a
    # known-numeric row, "km") actually parses as a number -- i.e. after
    # the descriptive "Description"/"Units" text columns -- and each
    # subsequent column is one more lake ("Lake1", "Lake2", ...).
    first_col_idx = next(i for i, v in enumerate(row_by_name["km"]) if _is_numeric(v))
    col_idx = first_col_idx + (lake_num - 1)

    if backup:
        shutil.copy(model_params_file, model_params_file + ".bak")

    changes = []
    for name, result_row in cal_df.iterrows():
        row = row_by_name[name]
        old = row[col_idx]
        new = str(result_row["calibrated"])
        if old.strip() != new:
            changes.append((name, old, new))
        row[col_idx] = new

    with open(model_params_file, "w", newline="") as f:
        csv.writer(f).writerows([header] + body)

    if not had_trailing_newline:
        # match the original file exactly if it had no final newline --
        # csv.writer always terminates the last row too.
        with open(model_params_file, "rb") as f:
            content = f.read()
        with open(model_params_file, "wb") as f:
            f.write(content.rstrip(b"\r\n"))

    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".",
                         help="folder containing model_params.csv and calibration_result.csv (e.g. Ravn)")
    parser.add_argument("--lake", type=int, default=1, help="lake/column number to update (default 1 -> 'Lake1')")
    parser.add_argument("--model-params-file", default=None, help="override path to model_params.csv")
    parser.add_argument("--calibration-file", default=None, help="override path to calibration_result.csv")
    parser.add_argument("--no-backup", action="store_true", help="skip writing model_params.csv.bak first")
    args = parser.parse_args()

    model_params_file = args.model_params_file or os.path.join(args.data_dir, "model_params.csv")
    calibration_file = args.calibration_file or os.path.join(args.data_dir, "calibration_result.csv")

    changes = update_model_params_from_calibration(
        model_params_file, calibration_file, lake_num=args.lake, backup=not args.no_backup,
    )

    if not changes:
        print(f"No differences -- {model_params_file} already matches the calibrated "
              f"values in {calibration_file}.")
        return

    print(f"Updated {model_params_file} from {calibration_file}:")
    for name, old, new in changes:
        print(f"  {name:28s} {old.strip():>16s} -> {new}")
    if not args.no_backup:
        print(f"(original saved to {model_params_file}.bak)")


if __name__ == "__main__":
    main()
