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


def plot_result(path):
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
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot lake model output from pickle or .npz files.")
    parser.add_argument("path", help="Path to a result directory, .pkl file, or .npz file.")
    args = parser.parse_args()
    plot_result(args.path)


if __name__ == "__main__":
    main()
