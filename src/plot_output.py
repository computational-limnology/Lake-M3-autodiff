import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


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


def _load_buoy_temperature_observations(
    result_dir,
    json_files=("ravn_2023.json", "ravn_2024.json"),
    sensor_file="sensor_level_buoy.sen",
):
    """Load the high-frequency buoy thermistor-chain record from the raw
    JSON exports (`ravn_2023.json`/`ravn_2024.json`), if present in
    `result_dir`, average it down to hourly values, and reshape it to the
    same long-format schema as `L0001-HD.csv`. See
    `calibrate_M3_jax.load_buoy_temperature()` for the JSON schema and
    resampling details (kept in sync with this copy). Returns `None` if
    `sensor_file` or none of `json_files` are present."""
    sensor_path = os.path.join(result_dir, sensor_file)
    if not os.path.exists(sensor_path):
        return None
    sensors = pd.read_csv(sensor_path, sep=r"\s+")
    depth_by_sensor = dict(zip(sensors["sensor"].astype(int), sensors["position"].astype(float)))

    hourly_frames = []
    for json_file in json_files:
        json_path = os.path.join(result_dir, json_file)
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
    return long[["datetime", "Depth_meter", "Water_Temperature_celsius"]]


def _load_temperature_observations(result_path, observations_path=None):
    result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
    if observations_path is None:
        observations_path = os.path.join(result_dir, "L0001-HD.csv")

    observations = None
    if os.path.exists(observations_path):
        observations = pd.read_csv(observations_path)
        required = {"datetime", "Depth_meter", "Water_Temperature_celsius"}
        missing = required.difference(observations.columns)
        if missing:
            raise ValueError(
                f"Temperature observations are missing required columns: {sorted(missing)}"
            )
        observations["datetime"] = pd.to_datetime(observations["datetime"])
        observations = observations[["datetime", "Depth_meter", "Water_Temperature_celsius"]]

    buoy = _load_buoy_temperature_observations(result_dir)
    if buoy is not None:
        observations = buoy if observations is None else pd.concat(
            [observations, buoy], ignore_index=True
        )

    if observations is None:
        return None

    return observations.dropna(subset=["datetime", "Depth_meter", "Water_Temperature_celsius"])


def _load_water_quality_observations(result_path, observations_path=None):
    if observations_path is None:
        result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
        observations_path = os.path.join(result_dir, "L0001-WQ.csv")

    if not os.path.exists(observations_path):
        return {}

    observations = pd.read_csv(observations_path)
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
        for variable in ("do", "doc")
    }


def _load_model_volume(result_path, n_depth):
    result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
    volume_path = os.path.join(result_dir, "volume.csv")
    if not os.path.exists(volume_path):
        return None

    volume = np.loadtxt(volume_path, delimiter=",")
    if volume.ndim == 1 and volume.size == n_depth:
        return volume
    return None


def _load_start_time(result_path):
    result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
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
    snapped to the nearest depth actually present in `observations`."""
    observed_depths = np.sort(observations["Depth_meter"].unique())

    if depths is not None:
        upper_requested, lower_requested = depths
        upper_depth = float(observed_depths[np.abs(observed_depths - upper_requested).argmin()])
        lower_depth = float(observed_depths[np.abs(observed_depths - lower_requested).argmin()])
        return upper_depth, lower_depth

    shallowest_depth = observed_depths[0]
    upper_observations = observations[observations["Depth_meter"] == shallowest_depth]
    upper_years = set(upper_observations["datetime"].dt.year)

    eligible_lower_depths = [
        depth
        for depth in observed_depths[1:]
        if upper_years.issubset(
            set(
                observations[observations["Depth_meter"] == depth]
                .groupby(observations["datetime"].dt.year)
                .size()
                .loc[lambda counts: counts >= 2]
                .index
            )
        )
    ]
    if not eligible_lower_depths:
        return None
    return shallowest_depth, eligible_lower_depths[-1]


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
        scatter_depths = observations["Depth_meter"].to_numpy(dtype=float)
        scatter_model_indices = np.abs(
            depth_values[:, None] - scatter_depths[None, :]
        ).argmin(axis=0)
        observed_times = pd.DatetimeIndex(observations["datetime"])
        right_idx = np.searchsorted(model_datetimes, observed_times)
        right_idx = np.clip(right_idx, 0, len(model_datetimes) - 1)
        left_idx = np.maximum(right_idx - 1, 0)
        right_distance = np.abs(model_datetimes[right_idx] - observed_times)
        left_distance = np.abs(model_datetimes[left_idx] - observed_times)
        nearest_idx = np.where(right_distance < left_distance, right_idx, left_idx)
        observed_values = observations[value_column].to_numpy(dtype=float)
        modeled_values = model_data[nearest_idx, scatter_model_indices]
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


def _prepare_plot_arrays(res, time_values, depth_values, n_depth, volume=None):
    arrays = {}
    if "volume" in res:
        volume = to_numpy(res["volume"])

    for name in ["temp", "o2", "docr", "docl", "pocr", "pocl"]:
        if name not in res:
            continue

        arr = to_numpy(res[name])
        arr, _, _ = _apply_orientation(arr, time_values, depth_values)

        if name != "temp" and volume is not None and volume.ndim == 1 and volume.size == n_depth:
            arr = arr / volume

        arrays[name] = arr

    return arrays


def plot_result(path, observations_path=None, water_quality_observations_path=None, depths=None):
    res = load_result(path)
    if not isinstance(res, dict):
        raise TypeError(f"Expected a dictionary-like result, got {type(res).__name__}")

    temp = to_numpy(res["temp"])
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
    start_time = _load_start_time(path)
    volume = to_numpy(res["volume"]) if "volume" in res else _load_model_volume(path, n_depth)

    arrays = _prepare_plot_arrays(res, time_values, depth_values, n_depth, volume)

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

    water_quality_observations = _load_water_quality_observations(
        path, water_quality_observations_path
    )
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
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot lake model output from pickle or .npz files.")
    parser.add_argument("path", help="Path to a result directory, .pkl file, or .npz file.")
    parser.add_argument(
        "--observations",
        help="Optional path to L0001-HD.csv; by default it is looked up beside the result.",
    )
    parser.add_argument(
        "--water-quality-observations",
        help="Optional path to L0001-WQ.csv; by default it is looked up beside the result.",
    )
    parser.add_argument(
        "--depths",
        help="Comma-separated upper,lower observed depths (meters) to use for the "
        "temperature/O2/DOC time-series panels, e.g. '--depths 1,30'. Each value is "
        "snapped to the nearest depth actually present in the observations. If "
        "omitted, depths are auto-selected as before (shallowest observed depth, "
        "and the deepest depth with data covering the same years).",
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

    plot_result(args.path, args.observations, args.water_quality_observations, depths=depths)


if __name__ == "__main__":
    main()
