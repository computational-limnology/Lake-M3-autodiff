"""
Sequential pipeline: gradient-based physical-parameter calibration
(`calibrate_M3_jax.py`) followed by the LSTM kz-correction training
(`run_M3_mcl_jax.py`), applying the calibrated parameters to
`model_params.csv` in between (`apply_calibration.py`) so the mcl stage
trains its correction on top of the newly calibrated physics rather than
whatever was on disk before.

Usage:
    python src/run_M3_calibrate_then_mcl.py Ravn --params Cd,at_factor
        [--cal-steps N] [--cal-chunk-steps N] [--cal-iters K] [--cal-lr LR]
        [--cal-early-stop-patience N] [--cal-early-stop-tol TOL]
        [--cal-variables temp,o2,doc]
        [--hidden-size H] [--depth-basis-degree D] [--kz-reg LAMBDA]
        [--mcl-steps N] [--mcl-chunk-steps N] [--mcl-iters K] [--mcl-lr LR]
        [--mcl-early-stop-patience N] [--mcl-early-stop-tol TOL]
        [--train-start DATE --train-end DATE --test-start DATE --test-end DATE]
        [--seed S] [--out result.npz] [--nn-params-out nn_params.pkl]
        [--skip-calibration] [--skip-mcl]

Why three stages, and why subprocesses:

Stage 1 (calibration) runs `calibrate_M3_jax.py` as a subprocess rather than
importing it in-process, because its `main()` does `os.chdir(args.data_dir)`
-- calling that in-process would leave this process's cwd corrupted for the
later stages. `--params p1,p2,...` (this wrapper's own top-level flag) is
passed straight through to `calibrate_M3_jax.py`'s own `--params`, which --
per the change documented in that script's module docstring -- skips its
sensitivity screen entirely (neither the combined nor the per-variable
ranking is computed) and calibrates exactly the named parameters directly.
This is the "focus calibration on specific keywords, skip sensitivity
analysis" mode this wrapper is built around, so `--params` is the flag most
users of this script will want to set. Omitting it is still allowed --
calibration then falls back to `calibrate_M3_jax.py`'s own auto-selection
via its sensitivity screen (`--cal-select-mode`/`--cal-topk`/
`--cal-topk-per-variable`) -- but that mode is slower and better exercised
by running `calibrate_M3_jax.py` on its own.

Stage 2 (apply) writes the calibrated values from stage 1's
`calibration_result.csv` into `model_params.csv` via
`apply_calibration.update_model_params_from_calibration()`, imported and
called directly (not subprocessed) since it is a plain function with no cwd
side effects. `backup=True` (its own default) keeps a timestamped copy of
`model_params.csv` before overwriting it.

Stage 3 (mcl) runs `run_M3_mcl_jax.py` as a subprocess (same cwd reason as
stage 1) against the now-updated `model_params.csv`, so the LSTM kz
correction is trained on top of the calibrated physical parameters rather
than the pre-calibration ones. All physical parameters are held fixed
during this stage, exactly as `run_M3_mcl_jax.py` does on its own -- only
its network weights are trained.

`--skip-calibration` leaves `model_params.csv` untouched (stage 2 is
skipped along with it, since there is nothing new to apply) and goes
straight to the mcl stage against whatever parameters are already on disk.
`--skip-mcl` stops after stage 2, e.g. to inspect the calibration result
before committing to a full mcl training run. Giving both is a no-op.

Flags are split into a `--cal-` prefixed group (passed to the calibration
stage), an unprefixed/`--mcl-` prefixed group (passed to the mcl stage),
and `--params`, which is shared conceptually but only consumed by stage 1.
The prefixes exist because `--iters`, `--lr`, `--steps` and `--chunk-steps`
are meaningful, differently-defaulted flags in *both* underlying scripts;
`--hidden-size`, `--depth-basis-degree`, `--kz-reg`, the train/test date
flags, `--seed`, `--out` and `--nn-params-out` are unique to the mcl stage
and so are left unprefixed. Defaults for every pass-through flag are
imported directly from the two scripts' own `DEFAULT_*` constants, so this
wrapper's `--help` output and behavior stay in sync with theirs
automatically.
"""
import argparse
import os
import subprocess
import sys
import time

from apply_calibration import update_model_params_from_calibration
from calibrate_M3_jax import (
    DEFAULT_CHUNK_STEPS as CAL_DEFAULT_CHUNK_STEPS,
    DEFAULT_ITERS as CAL_DEFAULT_ITERS,
    DEFAULT_LR as CAL_DEFAULT_LR,
    DEFAULT_EARLY_STOP_PATIENCE as CAL_DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EARLY_STOP_TOL as CAL_DEFAULT_EARLY_STOP_TOL,
    DEFAULT_TOPK as CAL_DEFAULT_TOPK,
    DEFAULT_TOPK_PER_VARIABLE as CAL_DEFAULT_TOPK_PER_VARIABLE,
)
from run_M3_mcl_jax import (
    DEFAULT_HIDDEN_SIZE, DEFAULT_DEPTH_BASIS_DEGREE, DEFAULT_KZ_REG,
    DEFAULT_TRAIN_START, DEFAULT_TRAIN_END, DEFAULT_TEST_START, DEFAULT_TEST_END,
    DEFAULT_ITERS as MCL_DEFAULT_ITERS,
    DEFAULT_LR as MCL_DEFAULT_LR,
    DEFAULT_CHUNK_STEPS as MCL_DEFAULT_CHUNK_STEPS,
    DEFAULT_EARLY_STOP_PATIENCE as MCL_DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EARLY_STOP_TOL as MCL_DEFAULT_EARLY_STOP_TOL,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_calibration_stage(data_dir_abs, args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "calibrate_M3_jax.py"), data_dir_abs]
    if args.cal_steps is not None:
        cmd += ["--steps", str(args.cal_steps)]
    cmd += ["--chunk-steps", str(args.cal_chunk_steps)]
    cmd += ["--iters", str(args.cal_iters)]
    cmd += ["--lr", str(args.cal_lr)]
    cmd += ["--early-stop-patience", str(args.cal_early_stop_patience)]
    cmd += ["--early-stop-tol", str(args.cal_early_stop_tol)]
    if args.params:
        cmd += ["--params", args.params]
    else:
        cmd += ["--topk", str(args.cal_topk)]
        cmd += ["--topk-per-variable", str(args.cal_topk_per_variable)]
        cmd += ["--select-mode", args.cal_select_mode]
    if args.cal_variables:
        cmd += ["--variables", args.cal_variables]

    print(f"\n=== Stage 1/3: calibration ===\n$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"--- calibration finished in {time.time() - t0:.1f}s ---")


def run_apply_stage(data_dir_abs):
    model_params_csv = os.path.join(data_dir_abs, "model_params.csv")
    result_csv = os.path.join(data_dir_abs, "calibration_result.csv")
    if not os.path.exists(result_csv):
        raise SystemExit(f"Expected {result_csv} after the calibration stage but it wasn't found.")

    print(f"\n=== Stage 2/3: applying {os.path.basename(result_csv)} to {os.path.basename(model_params_csv)} ===")
    changes = update_model_params_from_calibration(model_params_csv, result_csv, lake_num=1, backup=True)
    if changes:
        for name, old, new in changes:
            print(f"  {name}: {old} -> {new}")
    else:
        print("  (no parameter changes -- calibration_result.csv matched the values already on disk)")


def run_mcl_stage(data_dir_abs, args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_M3_mcl_jax.py"), data_dir_abs]
    if args.mcl_steps is not None:
        cmd += ["--steps", str(args.mcl_steps)]
    cmd += ["--chunk-steps", str(args.mcl_chunk_steps)]
    cmd += ["--hidden-size", str(args.hidden_size)]
    cmd += ["--depth-basis-degree", str(args.depth_basis_degree)]
    cmd += ["--kz-reg", str(args.kz_reg)]
    cmd += ["--iters", str(args.mcl_iters)]
    cmd += ["--lr", str(args.mcl_lr)]
    cmd += ["--early-stop-patience", str(args.mcl_early_stop_patience)]
    cmd += ["--early-stop-tol", str(args.mcl_early_stop_tol)]
    cmd += ["--train-start", args.train_start, "--train-end", args.train_end]
    cmd += ["--test-start", args.test_start, "--test-end", args.test_end]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--out", args.out, "--nn-params-out", args.nn_params_out]

    print(f"\n=== Stage 3/3: LSTM kz-correction training ===\n$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"--- mcl training finished in {time.time() - t0:.1f}s ---")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".")
    parser.add_argument("--params", type=str, default=None,
                         help="comma-separated parameter names to calibrate directly, skipping "
                              "calibrate_M3_jax.py's sensitivity screen entirely (recommended -- "
                              "this is the fast, targeted mode this wrapper is built around). "
                              "If omitted, the calibration stage falls back to its own "
                              "sensitivity-screen auto-selection (--cal-select-mode/--cal-topk/"
                              "--cal-topk-per-variable).")
    parser.add_argument("--skip-calibration", action="store_true",
                         help="skip stages 1-2 entirely and go straight to mcl training against "
                              "whatever parameters are already in model_params.csv")
    parser.add_argument("--skip-mcl", action="store_true",
                         help="stop after the calibration + apply stages, skip mcl training")

    cal = parser.add_argument_group("calibration stage (calibrate_M3_jax.py)")
    cal.add_argument("--cal-steps", type=int, default=None,
                      help="calibration-stage simulation window in hourly steps (default: full "
                           "record per run_config.csv)")
    cal.add_argument("--cal-chunk-steps", type=int, default=CAL_DEFAULT_CHUNK_STEPS,
                      help=f"default {CAL_DEFAULT_CHUNK_STEPS}")
    cal.add_argument("--cal-iters", type=int, default=CAL_DEFAULT_ITERS,
                      help=f"max Adam iterations for calibration (default {CAL_DEFAULT_ITERS})")
    cal.add_argument("--cal-lr", type=float, default=CAL_DEFAULT_LR,
                      help=f"calibration log-space Adam learning rate (default {CAL_DEFAULT_LR})")
    cal.add_argument("--cal-early-stop-patience", type=int, default=CAL_DEFAULT_EARLY_STOP_PATIENCE)
    cal.add_argument("--cal-early-stop-tol", type=float, default=CAL_DEFAULT_EARLY_STOP_TOL)
    cal.add_argument("--cal-topk", type=int, default=CAL_DEFAULT_TOPK,
                      help="only used when --params is omitted (--cal-select-mode combined)")
    cal.add_argument("--cal-topk-per-variable", type=int, default=CAL_DEFAULT_TOPK_PER_VARIABLE,
                      help="only used when --params is omitted (--cal-select-mode per-variable, the default)")
    cal.add_argument("--cal-select-mode", choices=["per-variable", "combined"], default="per-variable",
                      help="only used when --params is omitted")
    cal.add_argument("--cal-variables", type=str, default=None,
                      help="comma-separated subset of temp,o2,doc to calibrate against "
                           "(default: all with observations in the window)")

    mcl = parser.add_argument_group("mcl stage (run_M3_mcl_jax.py)")
    mcl.add_argument("--mcl-steps", type=int, default=None,
                      help="mcl-stage simulation window in hourly steps (default: full record)")
    mcl.add_argument("--mcl-chunk-steps", type=int, default=MCL_DEFAULT_CHUNK_STEPS,
                      help=f"default {MCL_DEFAULT_CHUNK_STEPS}")
    mcl.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE,
                      help=f"LSTM hidden size (default {DEFAULT_HIDDEN_SIZE})")
    mcl.add_argument("--depth-basis-degree", type=int, default=DEFAULT_DEPTH_BASIS_DEGREE,
                      help=f"degree of the polynomial-in-depth kz correction (default {DEFAULT_DEPTH_BASIS_DEGREE})")
    mcl.add_argument("--kz-reg", type=float, default=DEFAULT_KZ_REG,
                      help=f"L2 penalty on the NN's weights (default {DEFAULT_KZ_REG})")
    mcl.add_argument("--mcl-iters", type=int, default=MCL_DEFAULT_ITERS,
                      help=f"max Adam iterations for mcl training (default {MCL_DEFAULT_ITERS})")
    mcl.add_argument("--mcl-lr", type=float, default=MCL_DEFAULT_LR,
                      help=f"mcl Adam learning rate (default {MCL_DEFAULT_LR})")
    mcl.add_argument("--mcl-early-stop-patience", type=int, default=MCL_DEFAULT_EARLY_STOP_PATIENCE)
    mcl.add_argument("--mcl-early-stop-tol", type=float, default=MCL_DEFAULT_EARLY_STOP_TOL)
    mcl.add_argument("--train-start", type=str, default=DEFAULT_TRAIN_START)
    mcl.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    mcl.add_argument("--test-start", type=str, default=DEFAULT_TEST_START)
    mcl.add_argument("--test-end", type=str, default=DEFAULT_TEST_END)
    mcl.add_argument("--seed", type=int, default=0)
    mcl.add_argument("--out", type=str, default="mcl_result.npz")
    mcl.add_argument("--nn-params-out", type=str, default="mcl_nn_params.pkl")

    args = parser.parse_args()

    if args.skip_calibration and args.skip_mcl:
        raise SystemExit("--skip-calibration and --skip-mcl together leave nothing for this "
                          "script to do.")

    data_dir_abs = os.path.abspath(args.data_dir)
    t_start = time.time()

    if not args.skip_calibration:
        if not args.params:
            print("No --params given: the calibration stage will run its own sensitivity screen "
                  f"(--cal-select-mode {args.cal_select_mode}) before selecting parameters -- see "
                  "calibrate_M3_jax.py's module docstring. Pass --params p1,p2,... to skip "
                  "straight to calibrating named parameters instead.")
        run_calibration_stage(data_dir_abs, args)
        run_apply_stage(data_dir_abs)
    else:
        print("--skip-calibration: leaving model_params.csv untouched, starting the mcl stage "
              "from whatever physical parameters are already on disk.")

    if not args.skip_mcl:
        run_mcl_stage(data_dir_abs, args)
    else:
        print("--skip-mcl: stopping after the calibration/apply stage.")

    print(f"\nPipeline finished in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
