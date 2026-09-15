import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _build_time_labels(time_values):
    if time_values is None:
        return None, None

    values = to_numpy(time_values)
    if values.size == 0:
        return None, None

    try:
        labels = pd.to_datetime(values)
        tick_idx = np.linspace(0, len(labels) - 1, min(10, len(labels)), dtype=int)
        tick_labels = [lbl.strftime("%Y-%m-%d") for lbl in labels[tick_idx]]
        return tick_idx, tick_labels
    except Exception:
        tick_idx = np.linspace(0, len(values) - 1, min(10, len(values)), dtype=int)
        tick_labels = [f"{float(v):g}" for v in values[tick_idx]]
        return tick_idx, tick_labels


def _load_temperature_observations(result_path, observations_path=None):
    if observations_path is None:
        result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
        observations_path = os.path.join(result_dir, "L0001-HD.csv")

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
    return observations.dropna(subset=["datetime", "Depth_meter", "Water_Temperature_celsius"])


def _load_start_time(result_path):
    result_dir = result_path if os.path.isdir(result_path) else os.path.dirname(result_path)
    run_config_path = os.path.join(result_dir, "run_config.csv")
    if not os.path.exists(run_config_path):
        return None

    run_config = pd.read_csv(run_config_path, index_col=0)
    if run_config.empty or "start_time" not in run_config.index:
        return None
    return pd.to_datetime(run_config.loc["start_time"].iloc[0])


def _plot_temperature_profiles(
    temp, time_values, depth_values, observations, start_time=None
):
    if observations is None or observations.empty:
        return None
    if depth_values is None or depth_values.ndim != 1 or depth_values.size != temp.shape[1]:
        return None

    observed_depths = np.sort(observations["Depth_meter"].unique())
    shallowest_depth = observed_depths[0]
    deepest_depth = observed_depths[-1]
    selected_depths = [shallowest_depth, deepest_depth]
    model_indices = [int(np.abs(depth_values - depth).argmin()) for depth in selected_depths]

    if start_time is not None and time_values is not None:
        model_times = start_time + pd.to_timedelta(time_values, unit="s")
        x_label = "Date"
    else:
        model_times = np.arange(temp.shape[0])
        x_label = "Time step"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for axis, observed_depth, model_index in zip(axes, selected_depths, model_indices):
        observed = observations[observations["Depth_meter"] == observed_depth]
        axis.plot(
            model_times,
            temp[:, model_index],
            label=f"Model ({depth_values[model_index]:g} m)",
        )
        axis.scatter(
            observed["datetime"],
            observed["Water_Temperature_celsius"],
            label=f"Observed ({observed_depth:g} m)",
            color="black",
            s=18,
            zorder=3,
        )
        axis.set_title(f"Temperature at {observed_depth:g} m observed depth")
        axis.set_ylabel("Temperature (deg C)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel(x_label)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _prepare_plot_arrays(res, time_values, depth_values, n_depth):
    arrays = {}
    volume = None
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


def plot_result(path, observations_path=None):
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

    arrays = _prepare_plot_arrays(res, time_values, depth_values, n_depth)

    x_ticks, x_labels = _build_time_labels(time_values)
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
    start_time = _load_start_time(path)
    _plot_temperature_profiles(temp, time_values, depth_values, observations, start_time)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot lake model output from pickle or .npz files.")
    parser.add_argument("path", help="Path to a result directory, .pkl file, or .npz file.")
    parser.add_argument(
        "--observations",
        help="Optional path to L0001-HD.csv; by default it is looked up beside the result.",
    )
    args = parser.parse_args()
    plot_result(args.path, args.observations)


if __name__ == "__main__":
    main()
