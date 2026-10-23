import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from data_tools import lake_dir_for_result, load_run_config, load_temperature_dataframe, load_water_quality_dataframe


def to_numpy(arr):
    if isinstance(arr, pd.DataFrame):
        return arr.to_numpy()
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return np.asarray(arr)


def _resolve_input_path(path):
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "res_lake1.pkl"),
            os.path.join(path, "res_lake1_jax_full.npz"),
            os.path.join(path, "res_lake1_jax_temp.npz"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"No supported output file found in directory '{path}'. "
            "Looked for res_lake1.pkl, res_lake1_jax_full.npz, and res_lake1_jax_temp.npz."
        )
    return path


def load_result(path):
    resolved_path = _resolve_input_path(path)
    print(f"Reading file: {resolved_path}")

    if resolved_path.lower().endswith(".npz"):
        with np.load(resolved_path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}

    if resolved_path.lower().endswith(".pkl"):
        try:
            with open(resolved_path, "rb") as f:
                res = pickle.load(f)
        except TypeError as err:
            if "StringDtype" in str(err) or "pandas" in str(err):
                try:
                    res = pd.read_pickle(resolved_path)
                except Exception as err2:
                    raise RuntimeError(
                        "Failed to load the pickle file with both pickle.load and pandas.read_pickle. "
                        "This often happens when the file was created with a different pandas version."
                    ) from err2
            else:
                raise
        return res

    raise ValueError(f"Unsupported result format: '{resolved_path}'. Expected .npz or .pkl.")


def _apply_orientation(data, time_values, depth_values):
    arr = to_numpy(data)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape {arr.shape}")

    if time_values is not None and depth_values is not None:
        if time_values.size == arr.shape[0] and depth_values.size == arr.shape[1]:
            return arr, time_values, depth_values
        if time_values.size == arr.shape[1] and depth_values.size == arr.shape[0]:
            return arr.T, time_values, depth_values

    if time_values is not None and time_values.size == arr.shape[1] and arr.shape[0] != time_values.size:
        return arr.T, time_values, depth_values

    if depth_values is not None and depth_values.size == arr.shape[0] and arr.shape[1] != depth_values.size:
        return arr.T, time_values, depth_values

    return arr, time_values, depth_values


def _build_time_labels(time_values, start_time=None):
    if time_values is None:
        return None, None

    values = to_numpy(time_values)
    if values.size == 0:
        return None, None

    try:
        if start_time is not None and np.issubdtype(values.dtype, np.number):
            labels = start_time + pd.to_timedelta(values, unit="s")
        else:
            labels = pd.to_datetime(values)
        tick_idx = np.linspace(0, len(labels) - 1, min(10, len(labels)), dtype=int)
        tick_labels = [lbl.strftime("%Y-%m-%d") for lbl in labels[tick_idx]]
        return tick_idx, tick_labels
    except Exception:
        tick_idx = np.linspace(0, len(values) - 1, min(10, len(values)), dtype=int)
        tick_labels = [f"{float(v):g}" for v in values[tick_idx]]
        return tick_idx, tick_labels


def _load_temperature_observations(result_path, observations_path=None):
    """Temperature observations for plotting: `observations_path` if given
    (an explicit `--observations` override), else `run_config.csv`'s
    `u_ini_file` (falling back to `L0001-HD.csv`) via
    `data_tools.load_temperature_dataframe()` -- the same source
    `calibrate_M3_jax.py` trains/calibrates against. If the high-frequency
    buoy thermistor-chain record should be included, run
    `data_tools.py integrate-buoy` once first to fold it into `u_ini_file`
    directly; this function no longer merges it in separately (it used to,
    via its own copy of that logic, which is how `u_ini_file` ends up with
    everything already merged in the first place)."""
    result_dir = lake_dir_for_result(result_path)  # lake folder (also for results stored in mcl/)

    if observations_path is not None:
        if not os.path.exists(observations_path):
            return None
        observations = pd.read_csv(observations_path)
        required = {"datetime", "Depth_meter", "Water_Temperature_celsius"}
        missing = required.difference(observations.columns)
        if missing:
            raise ValueError(
                f"Temperature observations are missing required columns: {sorted(missing)}"
            )
        observations["datetime"] = pd.to_datetime(observations["datetime"])
        observations = observations[["datetime", "Depth_meter", "Water_Temperature_celsius"]]
    else:
        run_config = load_run_config(result_dir)
        observations = load_temperature_dataframe(result_dir, run_config)

    if observations is None:
        return None

    return observations.dropna(subset=["datetime", "Depth_meter", "Water_Temperature_celsius"])


def _load_water_quality_observations(result_path, observations_path=None):
    """Dissolved-oxygen/DOC/POC observations for plotting: `observations_path`
    if given (an explicit `--water-quality-observations` override), else
    `run_config.csv`'s `wq_ini_file` (falling back to `L0001-WQ.csv`) via
    `data_tools.load_water_quality_dataframe()`. 'poc' is compared
    against the model's combined `pocr + pocl` the same way 'doc' is
    compared against `docr + docl` -- the observation files in this
    project don't distinguish refractory vs. labile POC, only total POC."""
    result_dir = lake_dir_for_result(result_path)  # lake folder (also for results stored in mcl/)

    if observations_path is not None:
        if not os.path.exists(observations_path):
            return {}
        observations = pd.read_csv(observations_path)
    else:
        run_config = load_run_config(result_dir)
        observations = load_water_quality_dataframe(result_dir, run_config)
        if observations is None:
            return {}

    required = {"datetime", "depth", "observation", "variable"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(
            f"Water-quality observations are missing required columns: {sorted(missing)}"
        )

    observations["datetime"] = pd.to_datetime(observations["datetime"])
    observations = observations.dropna(subset=["datetime", "depth", "observation"])
    return {
        variable: observations[observations["variable"] == variable].rename(
            columns={"depth": "Depth_meter", "observation": "value"}
        )
        for variable in ("do", "doc", "poc", "kd")
    }


def _load_model_volume(result_path, n_depth):
    result_dir = lake_dir_for_result(result_path)  # lake folder (also for results stored in mcl/)
    volume_path = os.path.join(result_dir, "volume.csv")
    if not os.path.exists(volume_path):
        return None

    volume = np.loadtxt(volume_path, delimiter=",")
    if volume.ndim == 1 and volume.size == n_depth:
        return volume
    return None


def _load_start_time(result_path):
    result_dir = lake_dir_for_result(result_path)  # lake folder (also for results stored in mcl/)
    run_config_path = os.path.join(result_dir, "run_config.csv")
    if not os.path.exists(run_config_path):
        return None

    run_config = pd.read_csv(run_config_path, index_col=0)
    if run_config.empty or "start_time" not in run_config.index:
        return None
    return pd.to_datetime(run_config.loc["start_time"].iloc[0])


def _model_datetimes(time_values, start_time):
    if time_values is None:
        return None

    values = to_numpy(time_values)
    if start_time is not None and np.issubdtype(values.dtype, np.number):
        return start_time + pd.to_timedelta(values, unit="s")
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return None


def _select_temperature_depths(observations, depths=None):
    """Pick the (upper, lower) observed depths used for the time-series
    panels. By default, auto-selects the shallowest observed depth and the
    deepest depth that has data in every year the shallowest depth does.
    If `depths` is given (a 2-tuple/list of (upper, lower) depths in
    meters, e.g. from `--depths 1,30`), each requested value is instead
    snapped to the nearest depth actually present in `observations`. If
    only one depth is observed at all (e.g. a single-sensor high-frequency
    series with no companion deep profile -- Mendota's 'poc' record is
    entirely at 1 m, for instance), that same depth is used for both
    panels rather than returning `None`, so there's still something to
    plot."""
    observed_depths = np.sort(observations["Depth_meter"].unique())

    if len(observed_depths) == 1:
        return float(observed_depths[0]), float(observed_depths[0])

    if depths is not None:
        upper_requested, lower_requested = depths
        upper_depth = float(observed_depths[np.abs(observed_depths - upper_requested).argmin()])
        lower_depth = float(observed_depths[np.abs(observed_depths - lower_requested).argmin()])
        return upper_depth, lower_depth

    shallowest_depth = observed_depths[0]
    upper_observations = observations[observations["Depth_meter"] == shallowest_depth]
    upper_years = set(upper_observations["datetime"].dt.year)

    def years_with_profiles(depth):
        counts = (
            observations[observations["Depth_meter"] == depth]
            .groupby(observations["datetime"].dt.year)
            .size()
        )
        return set(counts.loc[lambda c: c >= 2].index)

    eligible_lower_depths = [
        depth for depth in observed_depths[1:] if upper_years.issubset(years_with_profiles(depth))
    ]
    if eligible_lower_depths:
        return shallowest_depth, eligible_lower_depths[-1]

    # Strict fallback found nothing -- this happens whenever the modeled
    # window doesn't fully span every calendar year the shallow depth has
    # data in (e.g. a short window, or one that starts/ends mid-year), even
    # if a perfectly usable deep comparison depth exists for the years that
    # *do* overlap. Rather than silently returning no plot at all, fall
    # back to the deepest depth with the most year-overlap with the
    # shallow depth, so there's always a best-effort default; pass
    # `--depths` explicitly to pick specific depths instead.
    overlap_counts = [
        (depth, len(upper_years & years_with_profiles(depth))) for depth in observed_depths[1:]
    ]
    overlap_counts = [(depth, n) for depth, n in overlap_counts if n > 0]
    if not overlap_counts:
        return None
    max_overlap = max(n for _, n in overlap_counts)
    best_depth = [depth for depth, n in overlap_counts if n == max_overlap][-1]
    print(
        f"  note: no depth has profile data in every year {shallowest_depth:g} m does within the "
        f"modeled period -- falling back to {best_depth:g} m (best year-overlap); pass --depths to "
        "choose explicitly."
    )
    return shallowest_depth, best_depth


def _filter_to_modeled_period(observations, time_values, start_time):
    model_datetimes = _model_datetimes(time_values, start_time)
    if model_datetimes is None:
        return observations, model_datetimes

    return (
        observations[
            observations["datetime"].between(model_datetimes[0], model_datetimes[-1])
        ],
        model_datetimes,
    )


def _match_model_to_observations(model_data, depth_values, model_datetimes, observations, value_column):
    """Nearest-neighbor match: each observation row is paired with the
    model's nearest depth level and nearest timestamp. Returns
    `(observed_values, modeled_values)` as plain numpy arrays, one entry
    per row of `observations`. Shared by the modeled-vs-observed scatter
    panel in `_plot_observed_series` and by the performance-metrics table
    (`_compute_metrics`/`_print_metrics_table`) below."""
    scatter_depths = observations["Depth_meter"].to_numpy(dtype=float)
    model_indices = np.abs(depth_values[:, None] - scatter_depths[None, :]).argmin(axis=0)
    observed_times = pd.DatetimeIndex(observations["datetime"])
    right_idx = np.searchsorted(model_datetimes, observed_times)
    right_idx = np.clip(right_idx, 0, len(model_datetimes) - 1)
    left_idx = np.maximum(right_idx - 1, 0)
    right_distance = np.abs(model_datetimes[right_idx] - observed_times)
    left_distance = np.abs(model_datetimes[left_idx] - observed_times)
    nearest_idx = np.where(right_distance < left_distance, right_idx, left_idx)
    observed_values = observations[value_column].to_numpy(dtype=float)
    modeled_values = model_data[nearest_idx, model_indices]
    return observed_values, modeled_values


def _rmse(sim, obs):
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def _r2(sim, obs):
    """Squared Pearson correlation coefficient -- the "goodness of fit"
    sense of R^2, distinct from NSE below (which also penalizes bias and
    scale errors, not just correlation). Same formula calibrate_M3_jax.py
    uses for its own evaluation metrics."""
    if len(obs) < 2 or np.std(obs) < 1e-12 or np.std(sim) < 1e-12:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    return float(r ** 2)


def _nse(sim, obs):
    """Nash-Sutcliffe Efficiency: 1 - SS_res/SS_tot. 1 = perfect,
    0 = no better than predicting the observed mean, <0 = worse."""
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1 - np.sum((obs - sim) ** 2) / denom)


def _kge(sim, obs):
    """Kling-Gupta Efficiency: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),
    with alpha = std(sim)/std(obs) (variability ratio) and
    beta = mean(sim)/mean(obs) (bias ratio). 1 = perfect."""
    if np.std(obs) < 1e-12 or abs(np.mean(obs)) < 1e-12:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _compute_metrics(model_data, time_values, depth_values, observations, value_column, start_time=None):
    """RMSE/NSE/KGE/R2 for one variable, in the spirit of
    `calibrate_M3_jax.py`'s `compute_metrics`/`extract_pairs` -- same four
    metrics, same formulas -- but matched by nearest-neighbor time & depth
    (`_match_model_to_observations`) rather than by an exact simulation
    step index, since this script works from an already-run result file
    rather than re-simulating at each observation's own step. Observations
    are first restricted to the modeled period (`_filter_to_modeled_period`),
    exactly like the scatter panel does. Returns `None` if there's nothing
    to score (no observations, no depth/time information to match against,
    or every matched pair is non-finite)."""
    if observations is None or observations.empty:
        return None
    if depth_values is None or depth_values.ndim != 1 or depth_values.size != model_data.shape[1]:
        return None

    observations, model_datetimes = _filter_to_modeled_period(observations, time_values, start_time)
    if observations.empty or model_datetimes is None:
        return None

    observed_values, modeled_values = _match_model_to_observations(
        model_data, depth_values, model_datetimes, observations, value_column,
    )
    finite = np.isfinite(observed_values) & np.isfinite(modeled_values)
    if not finite.any():
        return None
    sim = modeled_values[finite]
    obs = observed_values[finite]
    return dict(
        n=int(finite.sum()), rmse=_rmse(sim, obs), nse=_nse(sim, obs),
        kge=_kge(sim, obs), r2=_r2(sim, obs),
    )


def _match_kd(kd_model, time_values, start_time, kd_obs):
    """Pair each measured kd (one value per date, no depth) with the
    modeled kd_light at the nearest model timestamp, within the modeled
    period. Returns (obs_datetimes, observed, modeled) or None."""
    if kd_obs is None or kd_obs.empty or kd_model is None:
        return None
    kd_obs, model_datetimes = _filter_to_modeled_period(kd_obs, time_values, start_time)
    if kd_obs.empty or model_datetimes is None:
        return None
    kd_obs = kd_obs.groupby("datetime", as_index=False)["value"].mean()  # same-day duplicates
    model_ns = pd.DatetimeIndex(model_datetimes).asi8
    obs_ns = pd.DatetimeIndex(kd_obs["datetime"]).asi8
    idx = np.clip(np.searchsorted(model_ns, obs_ns), 1, len(model_ns) - 1)
    idx = np.where(np.abs(model_ns[idx - 1] - obs_ns) < np.abs(model_ns[idx] - obs_ns), idx - 1, idx)
    return kd_obs["datetime"], kd_obs["value"].to_numpy(dtype=float), np.asarray(kd_model)[idx]


def _kd_metrics(kd_model, time_values, start_time, kd_obs):
    matched = _match_kd(kd_model, time_values, start_time, kd_obs)
    if matched is None:
        return None
    _, obs, sim = matched
    finite = np.isfinite(obs) & np.isfinite(sim)
    if not finite.any():
        return None
    obs, sim = obs[finite], sim[finite]
    return dict(n=int(finite.sum()), rmse=_rmse(sim, obs), nse=_nse(sim, obs),
                kge=_kge(sim, obs), r2=_r2(sim, obs))


def _plot_kd_series(kd_model, time_values, start_time, kd_obs):
    """Time series of the model's light extinction kd_light [1/m] against
    measured kd (e.g. derived from Secchi depth), with a modeled-vs-observed
    scatter beside it."""
    model_datetimes = _model_datetimes(time_values, start_time)
    x = model_datetimes if model_datetimes is not None else np.arange(len(kd_model))
    matched = _match_kd(kd_model, time_values, start_time, kd_obs)
    fig, (ax_ts, ax_sc) = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    ax_ts.plot(x, kd_model, color="tab:blue", lw=1.0, label="Model kd_light")
    if matched is not None:
        obs_dt, obs, sim = matched
        ax_ts.scatter(obs_dt, obs, color="black", s=18, zorder=3, label="Measured kd")
        ax_sc.scatter(obs, sim, color="black", s=18)
        lo = float(np.nanmin(np.concatenate([obs, sim])))
        hi = float(np.nanmax(np.concatenate([obs, sim])))
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax_sc.legend()
    else:
        ax_sc.text(0.5, 0.5, "no measured kd\nin modeled period", ha="center", va="center",
                   transform=ax_sc.transAxes)
    ax_ts.set_title("Light extinction coefficient: modeled vs measured")
    ax_ts.set_ylabel("kd (1/m)")
    ax_ts.set_xlabel("Date" if model_datetimes is not None else "Time step")
    ax_ts.grid(alpha=0.3)
    ax_ts.legend()
    ax_sc.set_title("kd: modeled vs measured")
    ax_sc.set_xlabel("Measured kd (1/m)")
    ax_sc.set_ylabel("Modeled kd_light (1/m)")
    ax_sc.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _print_metrics_table(metrics):
    """Print RMSE/NSE/KGE/R2 for every variable with a non-`None` entry in
    `metrics` (a dict of variable name -> `_compute_metrics()` result, or
    `None` if that variable had nothing to score), in the same table style
    `calibrate_M3_jax.py`'s `print_metrics_table` uses."""
    scored = {name: m for name, m in metrics.items() if m is not None}
    if not scored:
        print("\nNo observations available to compute performance metrics against.")
        return
    print("\nPerformance metrics (modeled vs. observed, nearest-neighbor matched in time & depth; "
          "NSE/KGE/R2: 1.0 = perfect fit):")
    print(f"  {'variable':10s} {'n':>6s} {'RMSE':>12s} {'NSE':>8s} {'KGE':>8s} {'R2':>8s}")
    for name, m in scored.items():
        print(f"  {name:10s} {m['n']:6d} {m['rmse']:12.4g} {m['nse']:8.3f} {m['kge']:8.3f} {m['r2']:8.3f}")


def _plot_observed_series(
    model_data,
    time_values,
    depth_values,
    observations,
    value_column,
    title,
    y_label,
    start_time=None,
    figure_title=None,
    depths=None,
):
    if observations is None or observations.empty:
        return None
    if depth_values is None or depth_values.ndim != 1 or depth_values.size != model_data.shape[1]:
        return None

    observations, model_datetimes = _filter_to_modeled_period(
        observations, time_values, start_time
    )
    if observations.empty:
        return None

    selected_depths = _select_temperature_depths(observations, depths)
    if selected_depths is None:
        return None
    shallowest_depth, deepest_depth = selected_depths
    model_indices = [int(np.abs(depth_values - depth).argmin()) for depth in selected_depths]

    if model_datetimes is not None:
        model_times = model_datetimes
        x_label = "Date"
    else:
        model_times = np.arange(model_data.shape[0])
        x_label = "Time step"

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, width_ratios=(1.6, 1.0), wspace=0.28)
    axes = [fig.add_subplot(layout[row, 0]) for row in range(2)]
    scatter_axis = fig.add_subplot(layout[:, 1])
    for axis, observed_depth, model_index in zip(axes, selected_depths, model_indices):
        observed = observations[observations["Depth_meter"] == observed_depth]
        axis.plot(
            model_times,
            model_data[:, model_index],
            label=f"Model ({depth_values[model_index]:g} m)",
        )
        axis.scatter(
            observed["datetime"],
            observed[value_column],
            label=f"Observed ({observed_depth:g} m)",
            color="black",
            s=18,
            zorder=3,
        )
        axis.set_title(f"{title} at {observed_depth:g} m observed depth")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend()

    scatter = None
    scatter_values = []
    if model_datetimes is not None:
        observed_values, modeled_values = _match_model_to_observations(
            model_data, depth_values, model_datetimes, observations, value_column,
        )
        scatter_depths = observations["Depth_meter"].to_numpy(dtype=float)
        depth_min = scatter_depths.min()
        depth_max = scatter_depths.max()
        depth_norm = Normalize(
            vmin=depth_min if depth_min != depth_max else depth_min - 0.5,
            vmax=depth_max if depth_min != depth_max else depth_max + 0.5,
        )
        scatter = scatter_axis.scatter(
            observed_values,
            modeled_values,
            c=scatter_depths,
            cmap="viridis",
            norm=depth_norm,
            s=24,
            alpha=0.75,
        )
        scatter_values.extend((observed_values, modeled_values))
        fig.colorbar(scatter, ax=scatter_axis, label="Observed depth (m)")

    axes[-1].set_xlabel(x_label)
    has_scatter = bool(scatter_values)
    if has_scatter:
        scatter_values = np.concatenate(scatter_values)
        finite = np.isfinite(scatter_values)
        if finite.any():
            value_min = scatter_values[finite].min()
            value_max = scatter_values[finite].max()
            padding = max((value_max - value_min) * 0.05, 1e-12)
            line_min = value_min - padding
            line_max = value_max + padding
            scatter_axis.plot(
                [line_min, line_max],
                [line_min, line_max],
                color="black",
                linestyle="--",
                label="1:1",
            )
            scatter_axis.set_xlim(line_min, line_max)
            scatter_axis.set_ylim(line_min, line_max)
    scatter_axis.set_title(f"{title}: modeled vs observed")
    scatter_axis.set_xlabel("Observed")
    scatter_axis.set_ylabel("Modeled")
    scatter_axis.grid(True, alpha=0.3)
    scatter_axis.set_aspect("equal", adjustable="box")
    if has_scatter:
        scatter_axis.legend([scatter_axis.lines[-1]], ["1:1"])
    if figure_title is None and title == "Temperature":
        figure_title = (
            f"Observed depths during modeled period: upper {shallowest_depth:g} m, "
            f"lower {deepest_depth:g} m"
        )
    elif figure_title is None:
        figure_title = (
            f"{title}: observed depths during modeled period: upper {shallowest_depth:g} m, "
            f"lower {deepest_depth:g} m"
        )
    fig.suptitle(figure_title)
    fig.autofmt_xdate()
    return fig


def _plot_temperature_profiles(
    temp, time_values, depth_values, observations, start_time=None, depths=None
):
    return _plot_observed_series(
        temp,
        time_values,
        depth_values,
        observations,
        "Water_Temperature_celsius",
        "Temperature",
        "Temperature (deg C)",
        start_time,
        depths=depths,
    )


def _resolve_series_key(res, name, series=None, note=False):
    """Return the dict key holding the `name` field (e.g. "temp", "o2",
    "docr"), or `None` if it isn't present in this result at all. Plain
    model output (`run_M3_jax.py`'s `res_lake1_jax_full.npz`/
    `res_lake1.pkl`, or `run_M3_mcl_jax.py`'s `res_lake1_mcl_hybrid.npz`,
    which also uses plain field names) stores a single array per field.
    `train_M3_mcl_jax.py`'s `mcl_result.npz` instead stores a *pair* of
    series per field --
    `<name>_baseline` (process-based kz) and `<name>_hybrid` (process +
    LSTM-corrected kz) -- since the whole point of that file is comparing
    the two. If the plain key isn't present but this baseline/hybrid pair
    is, `--series` (default "hybrid") picks which one this plot treats as
    "the" model value for that field; the other one just isn't plotted here
    (see `plot_mcl_kz.py` for a side-by-side comparison of both, plus the
    kz fields this script doesn't touch)."""
    if name in res:
        return name
    baseline_key, hybrid_key = f"{name}_baseline", f"{name}_hybrid"
    if baseline_key in res or hybrid_key in res:
        chosen = series or "hybrid"
        key = f"{name}_{chosen}"
        if key not in res:
            available = [k for k in (baseline_key, hybrid_key) if k in res]
            raise KeyError(
                f"--series {chosen} requested but '{key}' isn't in this file; "
                f"available: {available}"
            )
        if note:
            print(f"No '{name}' key found -- this looks like an mcl_result.npz-style file "
                  f"(train_M3_mcl_jax.py); using '{key}' (--series {chosen}). For a side-by-side "
                  f"baseline-vs-hybrid comparison including kz, use plot_mcl_kz.py instead.")
        return key
    return None


def _resolve_temp_key(res, series=None):
    """Like `_resolve_series_key`, but "temp" is mandatory (there's nothing
    to plot without it) and always prints a note when it had to fall back
    to a baseline/hybrid pair."""
    key = _resolve_series_key(res, "temp", series, note=True)
    if key is None:
        raise KeyError(
            "No 'temp' (or 'temp_baseline'/'temp_hybrid') key found in this result -- "
            f"available keys: {sorted(res.keys())}"
        )
    return key


def _prepare_plot_arrays(res, time_values, depth_values, n_depth, volume=None, series=None):
    arrays = {}
    if "volume" in res:
        volume = to_numpy(res["volume"])

    for name in ["temp", "o2", "docr", "docl", "pocr", "pocl"]:
        key = _resolve_series_key(res, name, series)
        if key is None:
            continue

        arr = to_numpy(res[key])
        arr, _, _ = _apply_orientation(arr, time_values, depth_values)

        if name != "temp" and volume is not None and volume.ndim == 1 and volume.size == n_depth:
            arr = arr / volume

        arrays[name] = arr

    # Current speed sqrt(uvel^2 + vvel^2) -- the Goudsmit-style momentum
    # fields saved by run_M3_jax.py/run_M3_mcl_jax.py; absent entirely from
    # any output file produced before that addition, so this is skipped
    # (not an error) when either field is missing.
    # Resolved like the other fields, so train_M3_mcl_jax.py's
    # uvel_baseline/uvel_hybrid pairs work too (--series picks which).
    ukey = _resolve_series_key(res, "uvel", series)
    vkey = _resolve_series_key(res, "vvel", series)
    if ukey is not None and vkey is not None:
        uvel = to_numpy(res[ukey])
        vvel = to_numpy(res[vkey])
        uvel, _, _ = _apply_orientation(uvel, time_values, depth_values)
        vvel, _, _ = _apply_orientation(vvel, time_values, depth_values)
        arrays["speed"] = np.sqrt(uvel ** 2 + vvel ** 2)

    return arrays


def _plot_layer_bias_map(temp, time_values, depth_values, observations, start_time, obs_mode="profiles",
                         label="model", save_path=None):
    """Year x month map of the mean temperature bias (model - observed) in
    three layers: epilimnion 0-5 m, metalimnion 5-15 m, hypolimnion > 15 m
    (same layout as diagnose_temperature_errors.py's B_bias_map). Each
    observation is matched to the nearest model hour and layer.

    `obs_mode`: "profiles" (default) uses only rows with high_frequency == 0
    (sparse manual profiles) when the observation file has that flag; "all"
    uses every row."""
    if observations is None or observations.empty or depth_values is None or start_time is None:
        print("Layer bias map skipped (needs observations, depths and a start time).")
        return None
    obs = observations
    if obs_mode == "profiles":
        if "high_frequency" in obs.columns:
            obs = obs[obs["high_frequency"].fillna(0).astype(int) == 0]
        else:
            print("Layer bias map: no high_frequency column in the observations -- using all rows.")
    obs, model_datetimes = _filter_to_modeled_period(obs, time_values, start_time)
    if model_datetimes is None or obs.empty:
        print("Layer bias map skipped (no observations in the modelled period).")
        return None
    observed, modeled = _match_model_to_observations(
        temp, np.asarray(depth_values, float), pd.DatetimeIndex(model_datetimes), obs,
        "Water_Temperature_celsius")
    df = pd.DataFrame({
        "year": obs["datetime"].dt.year.to_numpy(), "month": obs["datetime"].dt.month.to_numpy(),
        "depth": obs["Depth_meter"].to_numpy(float), "resid": modeled - observed,
    })
    df["layer"] = pd.cut(df["depth"], [-1, 5, 15, 1e9], labels=["epi", "meta", "hypo"], right=False)
    years = np.arange(df["year"].min(), df["year"].max() + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, 0.17 * len(years) + 1.5)), sharey=True)
    im = None
    for ax, (layer, rng) in zip(axes, (("epi", "0-5 m"), ("meta", "5-15 m"), ("hypo", "> 15 m"))):
        m = (df[df["layer"] == layer].pivot_table(index="year", columns="month", values="resid", aggfunc="mean")
             .reindex(index=years, columns=range(1, 13)))
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                       extent=(0.5, 12.5, years[-1] + 0.5, years[0] - 0.5), interpolation="nearest")
        r = df.loc[df["layer"] == layer, "resid"]
        ax.set_title(f"{layer} ({rng})\nbias {r.mean():+.2f}, RMSE {np.sqrt(np.mean(r ** 2)):.2f} degC (n={len(r)})",
                     fontsize=10)
        ax.set_xlabel("month")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel("year")
    fig.colorbar(im, ax=axes, label="model - observed (degC)", fraction=0.02)
    fig.suptitle(f"{label}: temperature bias (model - observed) by layer, year and month"
                 + (" -- manual profiles only" if obs_mode == "profiles" else " -- all observations"), y=1.04)
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved layer bias map to {save_path}")
    return fig


def plot_result(path, observations_path=None, water_quality_observations_path=None, depths=None,
                 series=None, bias_map_obs="profiles"):
    res = load_result(path)
    if not isinstance(res, dict):
        raise TypeError(f"Expected a dictionary-like result, got {type(res).__name__}")

    temp_key = _resolve_temp_key(res, series)
    temp = to_numpy(res[temp_key])
    if temp.ndim != 2:
        raise ValueError(f"Expected 2D temperature array, got shape {temp.shape}")

    time_values = None
    for key in ("time", "times", "timelabels", "dates"):
        if key in res:
            time_values = to_numpy(res[key])
            break

    depth_values = None
    for key in ("depth", "z", "depths", "levels"):
        if key in res:
            depth_values = to_numpy(res[key])
            break

    temp, time_values, depth_values = _apply_orientation(temp, time_values, depth_values)
    n_time, n_depth = temp.shape
    # the run's own start_time (saved in newer result files) wins over the
    # lake's current run_config.csv, which may have been edited since the run
    start_time = (pd.Timestamp(str(np.asarray(res["start_time"]))) if "start_time" in res
                  else _load_start_time(path))
    volume = to_numpy(res["volume"]) if "volume" in res else _load_model_volume(path, n_depth)

    arrays = _prepare_plot_arrays(res, time_values, depth_values, n_depth, volume, series=series)

    x_ticks, x_labels = _build_time_labels(time_values, start_time)
    y_values = depth_values if depth_values is not None and depth_values.ndim == 1 and depth_values.size == n_depth else None

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    data_list = [
        (arrays.get("temp", temp), "Temperature", "Temperature"),
        (arrays.get("o2"), "DO (g/m3)", "Dissolved oxygen"),
        (arrays.get("docr"), "DOCR (g/m3)", "DOC-R"),
        (arrays.get("docl"), "DOCL (g/m3)", "DOC-L"),
        (arrays.get("pocr"), "POCR (g/m3)", "POC-R"),
        (arrays.get("pocl"), "POCL (g/m3)", "POC-L"),
        (arrays.get("speed"), "Current speed (m/s)", "Current speed sqrt(uvel^2+vvel^2)"),
    ]

    for i, (data, cbar_label, title) in enumerate(data_list):
        if data is None:
            axes[i].axis("off")
            continue

        mesh = axes[i].pcolormesh(
            np.arange(n_time + 1),
            -np.arange(n_depth + 1),
            data.T,
            cmap=plt.colormaps.get_cmap("Spectral_r"),
            shading="auto",
        )
        fig.colorbar(mesh, ax=axes[i], label=cbar_label)
        axes[i].set_title(title)
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Depth level")

        if x_ticks is not None and x_labels is not None:
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels(x_labels, rotation=45, ha="right")

        if y_values is not None:
            ytick_pos = -np.arange(n_depth)
            axes[i].set_yticks(ytick_pos)
            axes[i].set_yticklabels(y_values)

    for j in range(len(data_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    observations = _load_temperature_observations(path, observations_path)
    _plot_temperature_profiles(temp, time_values, depth_values, observations, start_time, depths=depths)
    if bias_map_obs != "none":
        resolved = _resolve_input_path(path)
        _plot_layer_bias_map(
            temp, time_values, depth_values, observations, start_time, obs_mode=bias_map_obs,
            label=os.path.splitext(os.path.basename(resolved))[0],
            save_path=os.path.splitext(resolved)[0] + "_bias_map.png" if os.path.isfile(resolved) else None)

    water_quality_observations = _load_water_quality_observations(
        path, water_quality_observations_path
    )

    # --- performance metrics (RMSE/NSE/KGE/R2), all variables with observations ---
    # Mirrors calibrate_M3_jax.py's evaluation-metrics table (same four
    # metrics/formulas -- see _rmse/_nse/_kge/_r2 above), computed here
    # against this already-run result file rather than a fresh simulation.
    # O2/DOC/POC are compared in concentration units (`arrays`, mass/volume)
    # against the mg/L observations, same units the plots above use; this
    # needs `volume` (see the `if volume is not None` guard below), which
    # is why those three are conditioned on it while temperature isn't.
    metrics = {"temp": _compute_metrics(temp, time_values, depth_values, observations,
                                         "Water_Temperature_celsius", start_time)}
    if volume is not None and volume.size == n_depth:
        o2 = arrays.get("o2")
        docr = arrays.get("docr")
        docl = arrays.get("docl")
        pocr = arrays.get("pocr")
        pocl = arrays.get("pocl")
        if o2 is not None:
            metrics["o2"] = _compute_metrics(
                o2, time_values, depth_values, water_quality_observations.get("do"),
                "value", start_time,
            )
        if docr is not None and docl is not None:
            metrics["doc"] = _compute_metrics(
                docr + docl, time_values, depth_values, water_quality_observations.get("doc"),
                "value", start_time,
            )
        if pocr is not None and pocl is not None:
            metrics["poc"] = _compute_metrics(
                pocr + pocl, time_values, depth_values, water_quality_observations.get("poc"),
                "value", start_time,
            )
    # Light extinction: modeled kd_light (saved by run_M3_jax.py and the MCL
    # scripts) against measured kd rows of the water-quality file, if both exist.
    kd_key = _resolve_series_key(res, "kd_light", series)
    kd_model = to_numpy(res[kd_key]).reshape(-1) if kd_key is not None else None
    kd_obs = water_quality_observations.get("kd")
    if kd_model is not None and kd_model.size != n_time:
        kd_model = None
    if kd_model is not None and kd_obs is not None and not kd_obs.empty:
        metrics["kd"] = _kd_metrics(kd_model, time_values, start_time, kd_obs)
    _print_metrics_table(metrics)
    if kd_model is not None:
        _plot_kd_series(kd_model, time_values, start_time, kd_obs)

    if volume is not None and volume.size == n_depth:
        o2 = arrays.get("o2")
        docr = arrays.get("docr")
        docl = arrays.get("docl")
        if o2 is not None:
            _plot_observed_series(
                o2,
                time_values,
                depth_values,
                water_quality_observations.get("do"),
                "value",
                "Dissolved oxygen",
                "O2 (mg/L)",
                start_time,
                depths=depths,
            )
        if docr is not None and docl is not None:
            _plot_observed_series(
                docr + docl,
                time_values,
                depth_values,
                water_quality_observations.get("doc"),
                "value",
                "Dissolved organic carbon",
                "DOC (mg/L)",
                start_time,
                depths=depths,
            )
        pocr = arrays.get("pocr")
        pocl = arrays.get("pocl")
        if pocr is not None and pocl is not None:
            _plot_observed_series(
                pocr + pocl,
                time_values,
                depth_values,
                water_quality_observations.get("poc"),
                "value",
                "Particulate organic carbon",
                "POC (mg/L)",
                start_time,
                depths=depths,
            )
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot lake model output from pickle or .npz files.")
    parser.add_argument("path", help="Path to a result directory, .pkl file, or .npz file.")
    parser.add_argument(
        "--observations",
        help="Optional explicit path to a temperature observations CSV; by default it is read "
        "from run_config.csv's u_ini_file, looked up beside the result.",
    )
    parser.add_argument(
        "--water-quality-observations",
        help="Optional explicit path to a water-quality observations CSV; by default it is read "
        "from run_config.csv's wq_ini_file, looked up beside the result.",
    )
    parser.add_argument(
        "--depths",
        help="Comma-separated upper,lower observed depths (meters) to use for the "
        "temperature/O2/DOC time-series panels, e.g. '--depths 1,30'. Each value is "
        "snapped to the nearest depth actually present in the observations. If "
        "omitted, depths are auto-selected as before (shallowest observed depth, "
        "and the deepest depth with data covering the same years).",
    )
    parser.add_argument(
        "--series",
        choices=["baseline", "hybrid"],
        default=None,
        help="Only relevant for an mcl_result.npz-style file (train_M3_mcl_jax.py), which "
        "stores paired 'temp_baseline'/'temp_hybrid' (and o2/docr/docl) series instead of "
        "single 'temp'/'o2'/'docr'/'docl' keys. Picks which one this script uses for all of "
        "them (default: hybrid). Ignored for a plain result file that already has plain keys.",
    )
    parser.add_argument(
        "--bias-map-obs",
        choices=["profiles", "all", "none"],
        default="profiles",
        help="Observations for the year x month layer bias map (also saved as "
        "<result>_bias_map.png): 'profiles' (default; only high_frequency == 0 rows, i.e. "
        "without the buoy record), 'all', or 'none' to skip it.",
    )
    args = parser.parse_args()

    depths = None
    if args.depths is not None:
        parts = [p.strip() for p in args.depths.split(",")]
        if len(parts) != 2:
            parser.error("--depths expects exactly two comma-separated values, e.g. '1,30'")
        try:
            depths = (float(parts[0]), float(parts[1]))
        except ValueError:
            parser.error(f"--depths values must be numeric, got '{args.depths}'")

    plot_result(
        args.path, args.observations, args.water_quality_observations, depths=depths,
        series=args.series, bias_map_obs=args.bias_map_obs,
    )


if __name__ == "__main__":
    main()
