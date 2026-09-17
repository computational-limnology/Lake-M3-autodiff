"""
Plot the process-based vs. LSTM-corrected eddy diffusivity (kz) from a
`run_M3_mcl_jax.py` result (`mcl_result.npz`), alongside a handful of
single-time kz(depth) profiles and a temperature check at two depths
against observations -- all in one figure.

Layout (one figure, three rows):
  1. Two heatmaps, time x depth, sharing a log color scale: kz_baseline
     (process-based `eddy_diffusivity_hendersonSellers`) and kz_hybrid
     (process-based kz corrected by the trained LSTM -- see
     `run_M3_mcl_jax.py`'s module docstring for the correction design).
  2. A row of small panels, one per snapshot date (`--profile-dates` or
     `--n-profile-dates` evenly-spaced defaults), each plotting kz(depth)
     at that single time step for both baseline and hybrid on a shared
     log-x/depth-y axis -- lets you see *where* and *by how much* the
     network is adjusting mixing, rather than only the aggregate heatmap.
  3. Temperature time series at two depths (`--depths`, default 1,30 m):
     baseline and hybrid model temperature plus the observed record
     (`L0001-HD.csv` + buoy `ravn_2023/2024.json`, via
     `plot_output._load_temperature_observations`) at the nearest
     observed depth, with the training/test windows shaded for reference.

Usage:
    python src/plot_mcl_kz.py Ravn/mcl_result.npz
        [--depths 1,30] [--n-profile-dates 5] [--profile-dates DATE,DATE,...]
        [--out Ravn/mcl_kz_plot.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from plot_output import (
    _load_temperature_observations, _load_start_time, _model_datetimes,
    _filter_to_modeled_period,
)


def load_mcl_result(path):
    with np.load(path) as d:
        return {k: np.asarray(d[k]) for k in d.files}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_path", help="Path to mcl_result.npz (from run_M3_mcl_jax.py).")
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
                         help="Optional path to L0001-HD.csv; by default it is looked up beside "
                              "mcl_result.npz (buoy ravn_2023/2024.json are picked up the same way).")
    parser.add_argument("--out", type=str, default=None,
                         help="Output image path (default: mcl_kz_plot.png beside mcl_result.npz).")
    args = parser.parse_args()

    result = load_mcl_result(args.result_path)
    result_dir = os.path.dirname(os.path.abspath(args.result_path))

    times = result["times"]
    depth = result["depth"]
    n_depth = depth.shape[0]
    kz_baseline = result["kz_baseline"]
    kz_hybrid = result["kz_hybrid"]
    temp_baseline = result["temp_baseline"]
    temp_hybrid = result["temp_hybrid"]
    train_lo, train_hi = int(result["train_lo"]), int(result["train_hi"])
    test_lo, test_hi = int(result["test_lo"]), int(result["test_hi"])

    start_time = _load_start_time(result_dir)
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

    observations = _load_temperature_observations(result_dir, args.observations)

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
    mesh = plot_kz_heatmap(ax_base, kz_baseline, model_datetimes, depth, norm, cmap,
                            "kz: process-based (Henderson-Sellers)")
    ax_hyb = fig.add_subplot(top[0, 1])
    plot_kz_heatmap(ax_hyb, kz_hybrid, model_datetimes, depth, norm, cmap,
                     "kz: process + LSTM correction")
    fig.colorbar(mesh, ax=[ax_base, ax_hyb], label="kz (m^2/s, log scale)", fraction=0.025, pad=0.02)

    # Row 2: kz(depth) snapshots
    n_panels = len(profile_idx)
    mid = outer[1].subgridspec(1, n_panels, wspace=0.15)
    for i, idx in enumerate(profile_idx):
        ax = fig.add_subplot(mid[0, i])
        ax.plot(kz_baseline[idx], -depth, label="process-based", color="tab:gray", lw=1.8)
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

        ax.plot(x, temp_baseline[:, model_idx], label="baseline (process-based kz)",
                color="tab:gray", lw=1.2)
        ax.plot(x, temp_hybrid[:, model_idx], label="hybrid (process + LSTM kz)",
                color="tab:red", lw=1.0)

        if obs_windowed is not None and not obs_windowed.empty:
            obs_depths = np.sort(obs_windowed["Depth_meter"].unique())
            nearest_obs_depth = float(obs_depths[np.argmin(np.abs(obs_depths - target_depth))])
            obs_here = obs_windowed[obs_windowed["Depth_meter"] == nearest_obs_depth]
            ax.scatter(obs_here["datetime"], obs_here["Water_Temperature_celsius"],
                       label=f"observed ({nearest_obs_depth:g} m)", color="black", s=10, zorder=3, alpha=0.6)

        ax.set_title(f"Temperature at {depth[model_idx]:g} m (nearest to {target_depth:g} m)")
        ax.set_ylabel("Temperature (deg C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if model_datetimes is not None:
            fig.autofmt_xdate()

    fig.suptitle("Process-based vs. LSTM-corrected eddy diffusivity (kz), and resulting temperature fit",
                 fontsize=14)

    out_path = args.out or os.path.join(result_dir, "mcl_kz_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
