"""
Plot the process-based vs. LSTM-corrected eddy diffusivity (kz) from a
`train_M3_mcl_jax.py` result (`mcl_result.npz`, its baseline-vs-hybrid
training-time comparison file -- not `run_M3_mcl_jax.py`'s
`res_lake1_mcl_hybrid.npz`, which has no baseline to compare against),
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
     `run_config.csv`'s `u_ini_file` -- run `integrate_buoy_temperature.py`
     once beforehand if the high-frequency buoy record should be included)
     at the nearest observed depth, with the training/test windows shaded
     for reference.

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
                              "it is read from run_config.csv's u_ini_file, looked up beside "
                              "mcl_result.npz.")
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
                            "kz: process-based (Pacanowski-Seiches)")
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



        if obs_windowed is not None and not obs_windowed.empty:
            obs_depths = np.sort(obs_windowed["Depth_meter"].unique())
            nearest_obs_depth = float(obs_depths[np.argmin(np.abs(obs_depths - target_depth))])
            obs_here = obs_windowed[obs_windowed["Depth_meter"] == nearest_obs_depth]
            ax.scatter(obs_here["datetime"], obs_here["Water_Temperature_celsius"],
                       label=f"observed ({nearest_obs_depth:g} m)", color="black", s=10, zorder=3, alpha=0.2)
        
        ax.plot(x, temp_baseline[:, model_idx], label="baseline (process-based kz)",
                color="tab:gray", lw=1.2, zorder =8)
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

    out_path = args.out or os.path.join(result_dir, "mcl_kz_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    coef_fig = plot_coefficients(result, model_datetimes, train_lo, train_hi, test_lo, test_hi)
    if coef_fig is not None:
        root, ext = os.path.splitext(out_path)
        coef_path = f"{root}_coefficients{ext or '.png'}"
        coef_fig.savefig(coef_path, dpi=150, bbox_inches="tight")
        print(f"Saved coefficient figure to {coef_path}")
    plt.show()


if __name__ == "__main__":
    main()
