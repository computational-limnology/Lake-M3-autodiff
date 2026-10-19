"""
Plot the process-based vs. LSTM-corrected eddy diffusivity (kz) from either
  * a `train_M3_mcl_jax.py` result (mcl/mcl_result_<lake>.npz): baseline
    run vs hybrid run, with the training/test windows shaded, or
  * a `run_M3_mcl_jax.py` result (<lake>/res_lake1_mcl_hybrid.npz): there is
    no separate baseline run in that file, so the "process-based" kz shown is
    `kz_process`, the closure's own kz diagnosed along the hybrid trajectory
    (i.e. what the closure asked for before the LSTM corrected it). A
    baseline temperature can be added with `--baseline <res_lake1_jax_full.npz>`
    (run_M3_jax.py output from the same start_time). By default the lake
    folder's res_lake1_jax_full.npz is used when it covers the same run (same
    number of steps and, when stored, the same start_time); without a
    baseline only the hybrid temperature is plotted and PB metrics are NaN. No train/test shading (not stored).
alongside a handful of
single-time kz(depth) profiles and a temperature check at two depths
against observations -- all in one figure.

Layout (one figure, three rows):
  1. Two heatmaps, time x depth, sharing a log color scale: kz_baseline
     (process-based `eddy_diffusivity_hendersonSellers`) and kz_hybrid
     (process-based kz corrected by the trained LSTM -- see
     `train_M3_mcl_jax.py`'s module docstring for the correction design).
  2. A row of small panels, one per snapshot date (`--profile-dates` or
     `--n-profile-dates` evenly-spaced defaults), each plotting kz(depth)
     at that single time step for both baseline and hybrid on a shared
     log-x/depth-y axis -- lets you see *where* and *by how much* the
     network is adjusting mixing, rather than only the aggregate heatmap.
  3. Temperature time series at two depths (`--depths`, default 1,30 m):
     baseline and hybrid model temperature plus the observed record (read
     via `plot_output._load_temperature_observations`, which follows
     `run_config.csv`'s `u_ini_file` -- run `data_tools.py integrate-buoy`
     once beforehand if the high-frequency buoy record should be included)
     at the nearest observed depth, with the training/test windows shaded
     for reference.

Second figure (<out>_depths.png): one panel per observed depth with the PB
(process-based) and MCL (hybrid) temperature at the nearest model layer and
the observations, titled with RMSE/NSE/KGE of PB and MCL against the
observations (nearest model hour). The same metrics -- also split into the
train/test windows when the file has them -- are printed and written to
<out>_depth_metrics.csv. Third figure (<out>_coefficients.png): the LSTM's
correction coefficients.

Usage:
    python mcl/plot_mcl_kz.py mcl/mcl_result_Ravn.npz
        [--depths 1,30] [--n-profile-dates 5] [--profile-dates DATE,DATE,...]
        [--out Ravn/mcl_kz_plot.png]
"""
# These MCL scripts live in <project>/mcl/; the model code they build on is
# in <project>/src/, so put that on the import path first.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from data_tools import lake_dir_for_result
import pandas as pd
from matplotlib.colors import LogNorm

from plot_output import (
    _load_temperature_observations, _load_start_time, _model_datetimes,
    _filter_to_modeled_period, _rmse, _nse, _kge,
)


def load_mcl_result(path):
    with np.load(path) as d:
        result = {k: np.asarray(d[k]) for k in d.files}
    return normalize_result(result)


def normalize_result(result):
    """Map a run_M3_mcl_jax.py inference file onto the training-result keys
    this script (and plot_ri_diagnostics.py) uses. Training results pass
    through unchanged. Sets result["_source"] to "training" or "inference"."""
    if "kz_baseline" in result or "kz_hybrid" in result:
        result["_source"] = "training"
        return result
    if "kz" not in result or "temp" not in result:
        raise KeyError("Neither a train_M3_mcl_jax.py result (kz_baseline/kz_hybrid) nor a "
                       "run_M3_mcl_jax.py result (kz/kz_process/temp) -- keys: " + ", ".join(sorted(result)))
    result["_source"] = "inference"
    result["kz_hybrid"] = result["kz"]
    result["kz_baseline"] = result.get("kz_process", result["kz"])
    result["temp_hybrid"] = result["temp"]
    for key in ("coef_a", "coef_b", "kz_add", "ri", "k0", "alpha", "n", "dens_diff", "Q_net"):
        if key in result and f"{key}_hybrid" not in result:
            result[f"{key}_hybrid"] = result[key]
    # no train/test windows in an inference file: -1 disables shade_windows()
    for key in ("train_lo", "train_hi", "test_lo", "test_hi"):
        result.setdefault(key, np.asarray(-1))
    # settings the coefficient plot needs; inference files store them in the
    # weights pickle (path in mcl_nn_params_path) rather than in the npz
    if "mcl_rbf_bands" in result and "mcl_band_labels" not in result:
        try:
            from train_M3_mcl_jax import RBF_BAND_SETS
            result["mcl_band_labels"] = np.array(RBF_BAND_SETS[str(result["mcl_rbf_bands"])]["labels"])
        except Exception:
            pass
    pkl = str(result.get("mcl_nn_params_path", ""))
    if pkl and ("mcl_k_ref" not in result or "mcl_max_add_factor" not in result):
        import pickle
        cand = [pkl, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(pkl))]
        for c in cand:
            if os.path.exists(c):
                with open(c, "rb") as f:
                    trained = pickle.load(f)
                result.setdefault("mcl_k_ref", np.asarray(trained.get("k_ref", 1e-5)))
                result.setdefault("mcl_max_add_factor", np.asarray(trained.get("max_add_factor", 20.0)))
                break
    return result


def nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def pick_profile_datetimes(model_datetimes, n, requested=None):
    """Either the explicitly `--profile-dates` (each snapped to the
    nearest simulated timestamp), or `n` timestamps evenly spaced across
    the record (excluding the very first/last step, which are rarely
    interesting snapshots)."""
    if requested:
        picks = []
        for date_str in requested:
            target = pd.Timestamp(date_str)
            idx = int(np.argmin(np.abs(model_datetimes - target)))
            picks.append(idx)
        return picks
    idx = np.linspace(0, len(model_datetimes) - 1, n + 2, dtype=int)[1:-1]
    return list(idx)


def plot_kz_heatmap(ax, kz, model_datetimes, depth, norm, cmap, title):
    n_time, n_depth = kz.shape
    mesh = ax.pcolormesh(
        np.arange(n_time + 1), -np.arange(n_depth + 1), kz.T,
        cmap=cmap, norm=norm, shading="auto",
    )
    ax.set_title(title)
    ax.set_ylabel("Depth (m)")
    ytick_pos = -np.arange(n_depth)[::max(1, n_depth // 8)]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{depth[i]:g}" for i in range(0, n_depth, max(1, n_depth // 8))])
    if model_datetimes is not None:
        tick_idx = np.linspace(0, n_time - 1, min(8, n_time), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([model_datetimes[i].strftime("%Y-%m-%d") for i in tick_idx], rotation=45, ha="right")
    return mesh


def shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi):
    n = len(model_datetimes)
    if 0 <= train_lo < train_hi <= n:
        ax.axvspan(model_datetimes[train_lo], model_datetimes[min(train_hi, n - 1)],
                   color="tab:orange", alpha=0.08, label="train window", zorder=0)
    if 0 <= test_lo < test_hi <= n:
        ax.axvspan(model_datetimes[test_lo], model_datetimes[min(test_hi, n - 1)],
                   color="tab:blue", alpha=0.08, label="test window", zorder=0)


def plot_coefficients(result, model_datetimes, train_lo, train_hi, test_lo, test_hi):
    """Second figure: time series of the LSTM's per-step correction
    coefficients (`--target kz` runs saved after coef_a/coef_b were added).

    Top: a_k, the log-scaling coefficient of each basis function (kz_process
    is multiplied by exp(sum_k a_k * phi_k(z)); 0 = no change).
    Bottom (additive form only): the mixing each basis function adds on its
    own, K_ref * min(softplus(scale * b_k), max_add_factor) [m^2/s], log axis.
    For `--depth-coord mixed_layer_rbf` the basis functions are the named
    depth bands; otherwise they are polynomial terms x^k. Daily means are
    plotted for readability; train/test windows are shaded."""
    if "coef_a_hybrid" not in result:
        return None
    a = np.asarray(result["coef_a_hybrid"])
    b = np.asarray(result["coef_b_hybrid"]) if "coef_b_hybrid" in result else None
    nb = a.shape[1]
    if "mcl_band_labels" in result:
        labels = [str(x) for x in np.asarray(result["mcl_band_labels"])]
    else:
        labels = [f"x^{k}" for k in range(nb)]
    depth_coord = str(result["mcl_depth_coord"]) if "mcl_depth_coord" in result else "?"

    n_rows = 2 if b is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.2 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = plt.colormaps.get_cmap("viridis")(np.linspace(0, 0.95, nb))

    def daily(x):
        if model_datetimes is None:
            return np.arange(x.shape[0]), x
        df = pd.DataFrame(x, index=model_datetimes).resample("D").mean()
        return df.index, df.to_numpy()

    t, a_d = daily(a)
    ax = axes[0]
    if model_datetimes is not None:
        shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi)
    for k in range(nb):
        ax.plot(t, a_d[:, k], color=colors[k], lw=1.2, label=labels[k])
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("a_k (log-scaling of kz_process)")
    note = ""
    if "mcl_rbf_bands" in result:
        note = f", bands: {str(result['mcl_rbf_bands'])}"
    if any("ML+" in lab for lab in labels):
        note += "; ML+x% = centred x% of the water column below the mixed layer"
    ax.set_title(f"Multiplicative correction coefficients a_k  (depth coordinate: {depth_coord}{note})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=min(nb + 2, 5), loc="upper left")

    if b is not None:
        k_ref = float(result.get("mcl_k_ref", 1e-5))
        cap = float(result.get("mcl_max_add_factor", 20.0))
        scale = float(result.get("mcl_add_output_scale", 5.0))
        added = k_ref * np.minimum(np.logaddexp(0.0, scale * b), cap)  # softplus
        t, add_d = daily(added)
        ax = axes[1]
        if model_datetimes is not None:
            shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi)
        for k in range(nb):
            ax.plot(t, add_d[:, k], color=colors[k], lw=1.2, label=labels[k])
        ax.set_yscale("log")
        ax.set_ylabel("K_ref*softplus(scale*b_k)  (m^2/s)")
        ax.set_title("Additive correction per basis function b_k, as added mixing")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=min(nb + 2, 5), loc="upper left")
    axes[-1].set_xlabel("Date" if model_datetimes is not None else "Time step")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _fit_metrics(sim, obs):
    ok = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[ok], obs[ok]
    if len(obs) < 3:
        return dict(n=int(len(obs)), rmse=np.nan, nse=np.nan, kge=np.nan)
    return dict(n=int(len(obs)), rmse=_rmse(sim, obs), nse=_nse(sim, obs), kge=_kge(sim, obs))


def depth_comparison(observations, model_datetimes, depth, temp_pb, temp_mcl,
                     train=None, test=None, min_obs=3):
    """Match every observation to the nearest model layer and hour, per
    observed depth. Returns (series, table): `series` is a list of dicts per
    observed depth (obs datetimes/values, model layer index, matched PB/MCL
    values), `table` a DataFrame with n/RMSE/NSE/KGE of PB and MCL per depth
    for the whole period and, when given, the train/test windows
    ((lo, hi) step-index tuples). Observed depths deeper than the model grid
    (by more than one layer) or with fewer than `min_obs` values are skipped.
    PB may be shorter than MCL (e.g. --baseline); it is only scored where it
    exists."""
    dx = float(np.median(np.diff(depth))) if len(depth) > 1 else 1.0
    times = pd.DatetimeIndex(model_datetimes)
    n_pb = temp_pb.shape[0] if temp_pb is not None else 0
    series, rows = [], []
    for d_obs, grp in observations.groupby("Depth_meter"):
        d_obs = float(d_obs)
        if d_obs > depth[-1] + dx or d_obs < depth[0] - dx:
            continue
        grp = grp.dropna(subset=["Water_Temperature_celsius"]).sort_values("datetime")
        if len(grp) < min_obs:
            continue
        k = int(np.argmin(np.abs(depth - d_obs)))
        ot = pd.DatetimeIndex(grp["datetime"])
        ti = np.clip(np.searchsorted(times, ot), 0, len(times) - 1)
        tl = np.maximum(ti - 1, 0)
        ti = np.where(np.abs(times[tl] - ot) < np.abs(times[ti] - ot), tl, ti)
        obs = grp["Water_Temperature_celsius"].to_numpy(float)
        mcl = temp_mcl[ti, k]
        pb = np.where(ti < n_pb, temp_pb[np.minimum(ti, max(n_pb - 1, 0)), k], np.nan) if n_pb else None
        series.append(dict(depth_obs=d_obs, layer=k, datetime=ot, obs=obs, pb=pb, mcl=mcl))
        row = dict(depth_obs=d_obs, depth_model=float(depth[k]))
        windows = [("all", None)] + [(name, w) for name, w in (("train", train), ("test", test))
                                      if w is not None and 0 <= w[0] < w[1]]
        for wname, w in windows:
            sel = np.ones(len(ti), bool) if w is None else (ti >= w[0]) & (ti < w[1])
            for mname, sim in (("PB", pb), ("MCL", mcl)):
                m = _fit_metrics(sim[sel], obs[sel]) if sim is not None else dict(n=0, rmse=np.nan, nse=np.nan, kge=np.nan)
                for key, val in m.items():
                    row[f"{mname}_{key}_{wname}"] = val
        rows.append(row)
    return series, pd.DataFrame(rows)


def plot_depth_comparison(series, table, model_datetimes, temp_pb, temp_mcl, depth,
                          train_lo, train_hi, test_lo, test_hi, ncols=3):
    """Figure: one panel per observed depth with the full PB and MCL model
    series at the matched layer, the observations, and RMSE/NSE/KGE of both
    in the panel title (whole period)."""
    n = len(series)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 2.7 * nrows), sharex=True, squeeze=False)
    x = model_datetimes
    n_pb = temp_pb.shape[0] if temp_pb is not None else 0
    for i, (sr, (_, row)) in enumerate(zip(series, table.iterrows())):
        ax = axes[i // ncols, i % ncols]
        shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi)
        k = sr["layer"]
        if n_pb:
            ax.plot(x[:n_pb], temp_pb[:n_pb, k], color="tab:gray", lw=0.9, label="PB")
        ax.plot(x, temp_mcl[:, k], color="tab:red", lw=0.8, label="MCL")
        ax.scatter(sr["datetime"], sr["obs"], s=4, color="black", alpha=0.35, zorder=5, label="observed")

        def fmt(m):
            if not np.isfinite(row[f"{m}_rmse_all"]):
                return f"{m}: n/a"
            return (f"{m}: RMSE {row[f'{m}_rmse_all']:.2f}, NSE {row[f'{m}_nse_all']:.2f}, "
                    f"KGE {row[f'{m}_kge_all']:.2f}")
        ax.set_title(f"{sr['depth_obs']:g} m obs / {depth[k]:g} m model  (n={int(row['MCL_n_all'])})\n"
                     f"{fmt('PB')}\n{fmt('MCL')}", fontsize=8)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("T (deg C)")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", markerscale=3)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle("Temperature per observed depth: process-based (PB) vs MCL hybrid vs observations "
                 "(metrics over the whole modelled period)", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def print_depth_table(table):
    has_split = any(c.endswith("_test") for c in table.columns)
    print("\nTemperature fit per observed depth (whole period"
          + ("; test window in brackets" if has_split else "") + "):")
    print(f"  {'depth':>6} {'n':>6} | {'PB RMSE':>8} {'NSE':>7} {'KGE':>7} | {'MCL RMSE':>8} {'NSE':>7} {'KGE':>7}")
    for _, r in table.iterrows():
        def cell(m, key, w="all"):
            v = r.get(f"{m}_{key}_{w}", np.nan)
            return f"{v:.3f}" if np.isfinite(v) else "nan"
        line = (f"  {r['depth_obs']:6g} {int(r['MCL_n_all']):6d} | {cell('PB','rmse'):>8} {cell('PB','nse'):>7} "
                f"{cell('PB','kge'):>7} | {cell('MCL','rmse'):>8} {cell('MCL','nse'):>7} {cell('MCL','kge'):>7}")
        if has_split:
            line += (f"   [test: PB {cell('PB','rmse','test')}/{cell('PB','nse','test')}/{cell('PB','kge','test')}"
                     f", MCL {cell('MCL','rmse','test')}/{cell('MCL','nse','test')}/{cell('MCL','kge','test')}]")
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_path", help="Path to mcl_result.npz (from train_M3_mcl_jax.py).")
    parser.add_argument("--depths", type=str, default="1,30",
                         help="Comma-separated depths (meters) for the temperature time-series "
                              "panels (default: 1,30). Each is snapped to the nearest model depth "
                              "for the simulated series, and to the nearest observed depth for "
                              "the scatter overlay.")
    parser.add_argument("--n-profile-dates", type=int, default=5,
                         help="Number of evenly-spaced kz(depth) snapshot panels (default 5); "
                              "ignored if --profile-dates is given.")
    parser.add_argument("--profile-dates", type=str, default=None,
                         help="Comma-separated dates (e.g. '2023-02-01,2023-07-01') to use for "
                              "the kz(depth) snapshot panels instead of evenly-spaced defaults.")
    parser.add_argument("--observations", type=str, default=None,
                         help="Optional explicit path to a temperature observations CSV; by default "
                              "it is read from run_config.csv's u_ini_file, looked up in the lake folder (--data-dir) of "
                              "mcl_result.npz.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output image path (default: <result name>_kz_plot.png beside the result).")
    parser.add_argument("--baseline", type=str, default=None,
                         help="run_M3_mcl_jax.py results only: a run_M3_jax.py output whose "
                              "temperature is the process-based (PB) baseline (default: "
                              "<lake>/res_lake1_jax_full.npz if it covers the same run)")
    parser.add_argument("--no-baseline", action="store_true",
                         help="run_M3_mcl_jax.py results: don't look for <lake>/res_lake1_jax_full.npz "
                              "as the PB baseline")
    parser.add_argument("--start-time", type=str, default=None,
                         help="start date of the run (e.g. 2023-01-01); needed for result files "
                              "without a stored start_time when run_config.csv's start_time has "
                              "changed since the run")
    parser.add_argument("--min-obs", type=int, default=10,
                         help="observed depths with fewer values are left out of the per-depth "
                              "figure/metrics (default 10)")
    parser.add_argument("--depth-cols", type=int, default=3,
                         help="columns of the per-depth temperature figure (default 3)")
    parser.add_argument("--data-dir", type=str, default=None,
                         help="lake data folder (run_config.csv, observations); default: the "
                              "result's folder, or for mcl/<name>_<lake>.npz the folder <lake>")
    args = parser.parse_args()

    result = load_mcl_result(args.result_path)
    result_dir = os.path.dirname(os.path.abspath(args.result_path))
    # lake folder holding run_config.csv / observations: the result's own
    # folder, or for files in mcl/ the lake named by the _<lake> suffix
    data_dir = lake_dir_for_result(args.result_path, args.data_dir)

    times = result["times"]
    depth = result["depth"]
    n_depth = depth.shape[0]
    kz_baseline = result["kz_baseline"]
    kz_hybrid = result["kz_hybrid"]
    temp_baseline = result.get("temp_baseline")
    temp_hybrid = result["temp_hybrid"]
    inference = result["_source"] == "inference"
    if inference:
        print("run_M3_mcl_jax.py output: 'process-based' kz = kz_process diagnosed along the hybrid run "
              "(no separate baseline run in this file); no train/test windows stored.")
        baseline_path = args.baseline
        if baseline_path is None and not args.no_baseline:
            # default: the plain run_M3_jax.py output in the lake folder, if it
            # covers the same run (same length and, when stored, same start)
            cand = os.path.join(data_dir, "res_lake1_jax_full.npz")
            if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(args.result_path):
                baseline_path = cand
        if baseline_path:
            with np.load(baseline_path) as b:
                b_temp = np.asarray(b["temp"])
                b_start = str(b["start_time"]) if "start_time" in b.files else None
            r_start = str(result["start_time"]) if "start_time" in result else None
            same_start = b_start is None or r_start is None or pd.Timestamp(b_start) == pd.Timestamp(r_start)
            if b_temp.shape[1] != temp_hybrid.shape[1]:
                print(f"Baseline {baseline_path} skipped: {b_temp.shape[1]} layers vs {temp_hybrid.shape[1]}.")
            elif not same_start:
                print(f"Baseline {baseline_path} skipped: starts {b_start}, this run starts {r_start}.")
            elif args.baseline is None and b_temp.shape[0] != temp_hybrid.shape[0]:
                print(f"Baseline {baseline_path} not used automatically: {b_temp.shape[0]} steps vs "
                      f"{temp_hybrid.shape[0]} in this run (pass --baseline to use it anyway).")
            else:
                temp_baseline = b_temp
                print(f"PB (baseline) temperature from {baseline_path} ({b_temp.shape[0]} steps"
                      + ("" if b_start else "; no stored start_time, assumed to start with this run") + ")")
        if temp_baseline is None:
            print("No PB baseline run available -> PB metrics are NaN. Run src/run_M3_jax.py for the same "
                  "period, or pass --baseline <file>.")
    train_lo, train_hi = int(result["train_lo"]), int(result["train_hi"])
    test_lo, test_hi = int(result["test_lo"]), int(result["test_hi"])

    # Start time: --start-time > the run's own start_time (saved in result
    # files written after this option existed) > the lake's run_config.csv.
    # The last one is only right if run_config.csv's start_time hasn't been
    # edited since the run -- older files from a shorter run (e.g. 2023-2024)
    # would otherwise be placed at the wrong dates.
    if args.start_time:
        start_time = pd.Timestamp(args.start_time)
    elif "start_time" in result:
        start_time = pd.Timestamp(str(result["start_time"]))
    else:
        start_time = _load_start_time(data_dir)
        print(f"Note: this file has no stored start_time; using run_config.csv's {start_time} "
              f"({len(times)} hourly steps -> ends {start_time + pd.Timedelta(hours=len(times) - 1)}). "
              "If the run started elsewhere, pass --start-time.")
    model_datetimes = _model_datetimes(times, start_time)
    if model_datetimes is not None:
        model_datetimes = pd.DatetimeIndex(model_datetimes)

    requested_dates = [d.strip() for d in args.profile_dates.split(",")] if args.profile_dates else None
    if model_datetimes is not None:
        profile_idx = pick_profile_datetimes(model_datetimes, args.n_profile_dates, requested_dates)
    else:
        profile_idx = list(np.linspace(0, len(times) - 1, args.n_profile_dates + 2, dtype=int)[1:-1])

    requested_depths = [float(d.strip()) for d in args.depths.split(",")]
    if len(requested_depths) != 2:
        parser.error("--depths expects exactly two comma-separated values, e.g. '1,30'")

    observations = _load_temperature_observations(data_dir, args.observations)

    # --- figure layout ---
    fig = plt.figure(figsize=(20, 17))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.1], hspace=0.5,
                              top=0.94, bottom=0.04, left=0.045, right=0.97)

    # Row 1: heatmaps (shared log color scale)
    top = outer[0].subgridspec(1, 2, wspace=0.25)
    vmin = max(1e-8, float(min(kz_baseline.min(), kz_hybrid.min())))
    vmax = float(max(kz_baseline.max(), kz_hybrid.max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap("viridis")

    ax_base = fig.add_subplot(top[0, 0])
    base_title = ("kz_process: closure kz along the hybrid run (before correction)" if inference
                  else "kz: process-based baseline run")
    mesh = plot_kz_heatmap(ax_base, kz_baseline, model_datetimes, depth, norm, cmap, base_title)
    ax_hyb = fig.add_subplot(top[0, 1])
    plot_kz_heatmap(ax_hyb, kz_hybrid, model_datetimes, depth, norm, cmap,
                     "kz: process + LSTM correction")
    fig.colorbar(mesh, ax=[ax_base, ax_hyb], label="kz (m^2/s, log scale)", fraction=0.025, pad=0.02)

    # Row 2: kz(depth) snapshots
    n_panels = len(profile_idx)
    mid = outer[1].subgridspec(1, n_panels, wspace=0.15)
    for i, idx in enumerate(profile_idx):
        ax = fig.add_subplot(mid[0, i])
        ax.plot(kz_baseline[idx], -depth, label="kz_process (hybrid run)" if inference else "process-based",
                color="tab:gray", lw=1.8)
        ax.plot(kz_hybrid[idx], -depth, label="process + LSTM", color="tab:red", lw=1.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(vmin, vmax)
        label = model_datetimes[idx].strftime("%Y-%m-%d %H:%M") if model_datetimes is not None else f"step {idx}"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("kz (m^2/s)")
        if i == 0:
            ax.set_ylabel("Depth (m)")
            ax.set_yticks(-depth[::max(1, n_depth // 8)])
            ax.set_yticklabels([f"{d:g}" for d in depth[::max(1, n_depth // 8)]])
            ax.legend(fontsize=8, loc="lower left")
        else:
            ax.set_yticks(-depth[::max(1, n_depth // 8)])
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    # Row 3: temperature time series at the two requested depths
    bottom = outer[2].subgridspec(1, 2, wspace=0.2)
    obs_windowed = None
    if observations is not None and model_datetimes is not None:
        obs_windowed, _ = _filter_to_modeled_period(observations, times, start_time)

    for i, target_depth in enumerate(requested_depths):
        ax = fig.add_subplot(bottom[0, i])
        model_idx = nearest_index(depth, target_depth)
        x = model_datetimes if model_datetimes is not None else np.arange(len(times))

        if model_datetimes is not None:
            shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi)



        if obs_windowed is not None and not obs_windowed.empty:
            obs_depths = np.sort(obs_windowed["Depth_meter"].unique())
            nearest_obs_depth = float(obs_depths[np.argmin(np.abs(obs_depths - target_depth))])
            obs_here = obs_windowed[obs_windowed["Depth_meter"] == nearest_obs_depth]
            ax.scatter(obs_here["datetime"], obs_here["Water_Temperature_celsius"],
                       label=f"observed ({nearest_obs_depth:g} m)", color="black", s=10, zorder=3, alpha=0.2)
        
        if temp_baseline is not None:
            nb_ = min(len(x), temp_baseline.shape[0])
            ax.plot(x[:nb_], temp_baseline[:nb_, model_idx], label="baseline (process-based kz)",
                    color="tab:gray", lw=1.2, zorder=8)
        ax.plot(x, temp_hybrid[:, model_idx], label="hybrid (process + LSTM kz)",
                color="tab:red", lw=1.0, zorder = 10)
        
        ax.set_title(f"Temperature at {depth[model_idx]:g} m (nearest to {target_depth:g} m)")
        ax.set_ylabel("Temperature (deg C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if model_datetimes is not None:
            fig.autofmt_xdate()

    fig.suptitle("Process-based vs. LSTM-corrected eddy diffusivity (kz), and resulting temperature fit",
                 fontsize=14)

    out_path = args.out or os.path.join(
        result_dir, os.path.splitext(os.path.basename(args.result_path))[0] + "_kz_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    # Second figure: PB vs MCL vs observed at every observed depth + metrics
    if obs_windowed is not None and not obs_windowed.empty:
        train = (train_lo, train_hi) if train_lo >= 0 else None
        test = (test_lo, test_hi) if test_lo >= 0 else None
        series, table = depth_comparison(obs_windowed, model_datetimes, depth, temp_baseline, temp_hybrid,
                                         train=train, test=test, min_obs=args.min_obs)
        if series:
            print_depth_table(table)
            root, ext = os.path.splitext(out_path)
            table_path = f"{root}_depth_metrics.csv"
            table.to_csv(table_path, index=False)
            print(f"Saved per-depth metrics to {table_path}")
            depth_fig = plot_depth_comparison(series, table, model_datetimes, temp_baseline, temp_hybrid, depth,
                                              train_lo, train_hi, test_lo, test_hi, ncols=args.depth_cols)
            depth_path = f"{root}_depths{ext or '.png'}"
            depth_fig.savefig(depth_path, dpi=130, bbox_inches="tight")
            print(f"Saved per-depth temperature figure to {depth_path}")
    else:
        print("No observations in the modelled period -- per-depth comparison skipped.")

    coef_fig = plot_coefficients(result, model_datetimes, train_lo, train_hi, test_lo, test_hi)
    if coef_fig is not None:
        root, ext = os.path.splitext(out_path)
        coef_path = f"{root}_coefficients{ext or '.png'}"
        coef_fig.savefig(coef_path, dpi=150, bbox_inches="tight")
        print(f"Saved coefficient figure to {coef_path}")
    plt.show()


if __name__ == "__main__":
    main()
