"""
Plot the process-based vs. MCL-corrected eddy diffusivity (kz) and the
resulting temperature fit, from any of
  * a `train_M3_mcl_jax.py` result (mcl/mcl_result_<lake>.npz, LSTM
    correction): separate process-based and hybrid runs, training/test
    windows shaded;
  * a `run_M3_mcl_jax.py` LSTM result (<lake>/res_lake1_mcl_hybrid.npz), or
  * a regime-correction run from `train_mcl_regime.py` (--apply or the final
    run; <lake>/res_lake1_mcl_regime.npz, or res_lake1_mcl_hybrid.npz when
    written by run_M3_mcl_jax.py from an mcl_regime_<lake>.npz file).
For the last two there is no separate process-based run in the file: the
"process" kz shown is `kz_process`, the closure's own kz diagnosed along the
corrected run (what the closure asked for before the correction). A
process-based temperature (PB) can be added with `--baseline
<res_lake1_jax_full.npz>` (run_M3_jax.py output, same start_time); by default
the lake folder's res_lake1_jax_full.npz is used when it covers the same run.
Without it only the corrected temperature is plotted and PB metrics are NaN.
Regime files also store the TKE head's entrainment efficiency (eta); their
coefficients are in mcl/mcl_regime_<lake>_coef.csv (the coefficient figure
below is LSTM-only).

Observations: `--temp-obs profiles` (default) uses the manual profiles only
(high_frequency == 0, as for calibration/training); `all` adds the buoy record.

Main figure (<out>):
  1. Two time x depth heatmaps on a shared log colour scale: process kz and
     corrected kz (long runs block-averaged in time, geometric means).
  2. kz(depth) snapshots at `--n-profile-dates` evenly spaced times (or
     `--profile-dates`): process vs corrected -- where and by how much the
     correction changes mixing.
  3. Temperature at two depths (`--depths`, default 1,30 m): PB, MCL and the
     observations at the nearest observed depth (lines are block means for
     long runs; metrics always use the hourly series).
Second figure (<out>_depths.png): one panel per observed depth, PB / MCL /
observed, titled with RMSE, NSE and KGE of PB and MCL (nearest model hour);
also printed and written to <out>_depth_metrics.csv (split into the training
and test windows when the file has them).
Third figure (<out>_coefficients.png, LSTM files only): the network's
correction coefficients over time.

Usage:
    python mcl/plot_mcl_kz.py Ravn/res_lake1_mcl_regime.npz
    python mcl/plot_mcl_kz.py mcl/mcl_result_Ravn.npz
        [--depths 1,30] [--n-profile-dates 5] [--profile-dates DATE,DATE,...]
        [--temp-obs profiles|all] [--out Ravn/mcl_kz_plot.png]
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


# Large (time x depth) fields neither this script nor plot_ri_diagnostics.py
# uses: not loaded (a full 1989-2024 run has ~160 MB per field; an
# mcl_result file stores each twice, baseline and hybrid).
SKIP_FIELDS = ("o2", "docr", "docl", "pocr", "pocl", "uvel", "vvel", "E_seiche")


def load_mcl_result(path, skip=SKIP_FIELDS):
    with np.load(path) as d:
        result = {}
        for k in d.files:
            base = k[:-len("_baseline")] if k.endswith("_baseline") else (k[:-len("_hybrid")] if k.endswith("_hybrid") else k)
            if base in skip:
                continue
            result[k] = np.asarray(d[k])
    return normalize_result(result)


def time_block_geomean(kz, max_cols):
    """Geometric block mean over time so at most max_cols columns remain
    (kz is shown on a log scale). Returns (reduced array, block length)."""
    n = kz.shape[0]
    if not max_cols or n <= max_cols:
        return kz, 1
    k = int(np.ceil(n / max_cols))
    n_full = (n // k) * k
    lk = np.log10(np.maximum(kz, 1e-12))
    parts = [lk[:n_full].reshape(n_full // k, k, kz.shape[1]).mean(axis=1)]
    if n_full < n:
        parts.append(lk[n_full:].mean(axis=0)[None, :])
    return 10.0 ** np.concatenate(parts, axis=0), k


def normalize_result(result):
    """Map a run_M3_mcl_jax.py inference file onto the training-result keys
    this script (and plot_ri_diagnostics.py) uses. Training results pass
    through unchanged. Sets result["_source"] to "training" or "inference"."""
    if "kz_baseline" in result or "kz_hybrid" in result:
        result["_source"] = "training"
        result["_correction"] = "LSTM"
        return result
    if "kz" not in result or "temp" not in result:
        raise KeyError("Neither a train_M3_mcl_jax.py result (kz_baseline/kz_hybrid) nor a "
                       "run_M3_mcl_jax.py result (kz/kz_process/temp) -- keys: " + ", ".join(sorted(result)))
    result["_source"] = "inference"
    # regime runs (train_mcl_regime.py) store the TKE head's eta; LSTM runs the weights path
    result["_correction"] = "regime" if ("eta" in result or "add_power_frac_wind" in result) else "LSTM"
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


LINE_MAX_POINTS = 4000


def thin_series(x, y, max_points=LINE_MAX_POINTS):
    """Block means of a long hourly line for plotting (metrics always use the
    full series). x: DatetimeIndex or array; returns (x_thin, y_thin)."""
    n = len(y)
    if n <= max_points:
        return x, y
    k = int(np.ceil(n / max_points))
    m = (n // k) * k
    yy = np.asarray(y, float)[:m].reshape(-1, k).mean(axis=1)
    xx = x[:m][k // 2::k]
    return xx, yy


def plot_kz_heatmap(ax, kz, model_datetimes, depth, norm, cmap, title, max_cols=4000):
    """Time x depth heatmap. Long runs are block-averaged (geometric mean) to
    at most `max_cols` columns and drawn with imshow: pcolormesh on every
    hour of a 36-year run needs ~1.3 GB per panel. The axis stays in
    time-step units, so the date ticks are unchanged."""
    n_time, n_depth = kz.shape
    reduced, block = time_block_geomean(kz, max_cols)
    mesh = ax.imshow(reduced.T, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
                     extent=(0, n_time, -n_depth, 0), origin="upper")
    ax.set_title(title if block == 1 else f"{title} ({block}-h geometric means)")
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
                          train_lo, train_hi, test_lo, test_hi, ncols=3, corr="MCL", obs_label="observations"):
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
            ax.plot(*thin_series(x[:n_pb], temp_pb[:n_pb, k]), color="tab:gray", lw=0.9, label="PB")
        ax.plot(*thin_series(x, temp_mcl[:, k]), color="tab:red", lw=0.8, label=f"MCL ({corr})")
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
            ax.set_ylabel("Temperature (°C)")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", markerscale=3)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(f"Temperature per observed depth: process-based (PB) vs MCL ({corr}) vs {obs_label} "
                 "(metrics over the whole modelled period, nearest model hour)", fontsize=12)
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
    parser.add_argument("result_path", help="MCL result: mcl/mcl_result_<lake>.npz (train_M3_mcl_jax.py), "
                         "<lake>/res_lake1_mcl_hybrid.npz (run_M3_mcl_jax.py) or "
                         "<lake>/res_lake1_mcl_regime.npz (train_mcl_regime.py)")
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
    parser.add_argument("--temp-obs", choices=("profiles", "all"), default="profiles",
                         help="temperature observations: 'profiles' (default; manual profiles only, as used "
                              "for calibration/training) or 'all' (adds the high-frequency buoy rows)")
    parser.add_argument("--heatmap-max-cols", type=int, default=4000,
                         help="max time columns in the kz heatmaps (block geometric means; default 4000, "
                              "0 = every step, ~1.3 GB RAM per panel for a 36-year hourly run)")
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
        print("Corrected run without a separate process-based run in the file: 'process' kz = kz_process "
              "diagnosed along the corrected run; no train/test windows stored.")
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
    if observations is not None and args.temp_obs == "profiles":
        if "high_frequency" in observations.columns:
            n_all = len(observations)
            observations = observations[observations["high_frequency"].fillna(0).astype(int) == 0]
            print(f"Temperature observations: manual profiles only ({len(observations)} of {n_all} rows)")
        else:
            print("--temp-obs profiles: no high_frequency column -- using all rows")
    obs_label = "manual profiles" if args.temp_obs == "profiles" else "all observations incl. buoy"
    corr = result.get("_correction", "LSTM")
    corr_txt = {"regime": "regime correction (kz + TKE heads)", "LSTM": "LSTM correction"}[corr]

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
    base_title = ("Process kz along the corrected run" if inference else "Process kz (process-based run)")
    mesh = plot_kz_heatmap(ax_base, kz_baseline, model_datetimes, depth, norm, cmap, base_title,
                           max_cols=args.heatmap_max_cols)
    ax_hyb = fig.add_subplot(top[0, 1])
    plot_kz_heatmap(ax_hyb, kz_hybrid, model_datetimes, depth, norm, cmap,
                     f"Corrected kz ({corr} correction)", max_cols=args.heatmap_max_cols)
    fig.colorbar(mesh, ax=[ax_base, ax_hyb], label="kz (m² s⁻¹, log scale)", fraction=0.025, pad=0.02)

    # Row 2: kz(depth) snapshots
    n_panels = len(profile_idx)
    mid = outer[1].subgridspec(1, n_panels, wspace=0.15)
    for i, idx in enumerate(profile_idx):
        ax = fig.add_subplot(mid[0, i])
        ax.plot(kz_baseline[idx], -depth, label="process kz (along corrected run)" if inference else "process kz",
                color="tab:gray", lw=1.8)
        ax.plot(kz_hybrid[idx], -depth, label="corrected kz", color="tab:red", lw=1.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(vmin, vmax)
        label = model_datetimes[idx].strftime("%Y-%m-%d %H:%M") if model_datetimes is not None else f"step {idx}"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("kz (m² s⁻¹)")
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
                       label=f"observed, {obs_label} ({nearest_obs_depth:g} m)", color="black", s=10, zorder=3,
                       alpha=0.2)
        
        if temp_baseline is not None:
            nb_ = min(len(x), temp_baseline.shape[0])
            ax.plot(*thin_series(x[:nb_], temp_baseline[:nb_, model_idx]), label="PB (process-based run)",
                    color="tab:gray", lw=1.2, zorder=8)
        ax.plot(*thin_series(x, temp_hybrid[:, model_idx]), label=f"MCL ({corr})",
                color="tab:red", lw=1.0, zorder = 10)
        
        ax.set_title(f"Temperature at {depth[model_idx]:g} m (nearest to {target_depth:g} m)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if model_datetimes is not None:
            fig.autofmt_xdate()

    fig.suptitle(f"Eddy diffusivity: process-based vs. {corr_txt}, and the resulting temperature fit "
                 f"(observations: {obs_label})", fontsize=14)

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
                                              train_lo, train_hi, test_lo, test_hi, ncols=args.depth_cols,
                                              corr=corr, obs_label=obs_label)
            depth_path = f"{root}_depths{ext or '.png'}"
            depth_fig.savefig(depth_path, dpi=130, bbox_inches="tight")
            print(f"Saved per-depth temperature figure to {depth_path}")
    else:
        print("No observations in the modelled period -- per-depth comparison skipped.")

    coef_fig = plot_coefficients(result, model_datetimes, train_lo, train_hi, test_lo, test_hi)
    if coef_fig is None and corr == "regime":
        print("Coefficient figure: LSTM files only; the regime coefficients are in "
              "mcl/mcl_regime_<lake>_coef.csv.")
    if coef_fig is not None:
        root, ext = os.path.splitext(out_path)
        coef_path = f"{root}_coefficients{ext or '.png'}"
        coef_fig.savefig(coef_path, dpi=150, bbox_inches="tight")
        print(f"Saved coefficient figure to {coef_path}")
    plt.show()


if __name__ == "__main__":
    main()
