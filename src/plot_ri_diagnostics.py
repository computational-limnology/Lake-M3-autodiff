"""
Plot the `--target ri` fit's learned Munk-Anderson stability parameters
(alpha, n) alongside a handful of physical drivers, from a
`run_M3_mcl_jax.py --target ri` result (`mcl_result.npz`).

Six stacked, shared-time-axis panels:
  1. alpha  (log scale -- see run_M3_mcl_jax.py's DEFAULT_RI_MEMORY_HOURS
     comment for what smooths this series: an explicit couple-hour EMA
     memory on the LSTM's raw per-step output, --ri-memory-hours).
  2. n
  3. top-bottom temperature difference (temp_hybrid[:, 0] - temp_hybrid[:, -1])
  4. wind speed (the meteorological forcing the model itself was driven by)
  5. top-bottom density difference (dens_diff_hybrid = rho_bottom - rho_top;
     the same stratification signal eddy_diffusivity_hendersonSellers's own
     Ri is built from)
  6. net surface heat flux (Q_net_hybrid -- net shortwave + longwave +
     sensible + latent, from jax_lakeModel_functions.heating_module)

This is a companion to plot_mcl_kz.py (which plots kz itself and the
resulting temperature fit); this script is about *why* alpha/n moved the
way they did, not about the resulting kz or temperature.

Usage:
    python src/plot_ri_diagnostics.py Ravn/mcl_result.npz
        [--start DATE] [--end DATE] [--out Ravn/ri_diagnostics_plot.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_output import _load_start_time, _model_datetimes
from plot_mcl_kz import load_mcl_result, shade_windows


REQUIRED_KEYS = [
    "alpha_hybrid", "n_hybrid", "temp_hybrid", "depth", "wind_speed",
    "dens_diff_hybrid", "Q_net_hybrid",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_path", help="Path to a mcl_result.npz produced with --target ri.")
    parser.add_argument("--start", type=str, default=None, help="Restrict the plot to this start date.")
    parser.add_argument("--end", type=str, default=None, help="Restrict the plot to this end date.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output image path (default: ri_diagnostics_plot.png beside mcl_result.npz).")
    args = parser.parse_args()

    result = load_mcl_result(args.result_path)
    result_dir = os.path.dirname(os.path.abspath(args.result_path))

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        raise SystemExit(
            f"{args.result_path} is missing {missing} -- this plot needs a mcl_result.npz produced "
            "with `run_M3_mcl_jax.py --target ri` (or the wrapper's --mcl-target ri), not --target kz "
            "(the default). Re-run the mcl stage with --target ri / --mcl-target ri first."
        )

    times = result["times"]
    depth = result["depth"]
    temp_hybrid = result["temp_hybrid"]
    alpha = result["alpha_hybrid"]
    n = result["n_hybrid"]
    wind_speed = result["wind_speed"]
    dens_diff = result["dens_diff_hybrid"]
    q_net = result["Q_net_hybrid"]
    train_lo, train_hi = int(result["train_lo"]), int(result["train_hi"])
    test_lo, test_hi = int(result["test_lo"]), int(result["test_hi"])
    ri_memory_hours = float(result["ri_memory_hours"]) if "ri_memory_hours" in result else None

    top_minus_bottom_temp = temp_hybrid[:, 0] - temp_hybrid[:, -1]

    start_time = _load_start_time(result_dir)
    model_datetimes = _model_datetimes(times, start_time)
    if model_datetimes is not None:
        model_datetimes = pd.DatetimeIndex(model_datetimes)
    x = model_datetimes if model_datetimes is not None else np.arange(len(times))

    window_mask = np.ones(len(times), dtype=bool)
    if model_datetimes is not None and (args.start or args.end):
        if args.start:
            window_mask &= (model_datetimes >= pd.Timestamp(args.start))
        if args.end:
            window_mask &= (model_datetimes <= pd.Timestamp(args.end))

    panels = [
        (alpha, "alpha", "tab:red", True),
        (n, "n", "tab:purple", False),
        (top_minus_bottom_temp, "Top - bottom\ntemperature (deg C)", "tab:orange", False),
        (wind_speed, "Wind speed (m/s)", "tab:green", False),
        (dens_diff, "Bottom - top\ndensity (kg/m^3)", "tab:brown", False),
        (q_net, "Net surface\nheat flux (W/m^2)", "tab:blue", False),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 16), sharex=True)
    for ax, (series, label, color, log_scale) in zip(axes, panels):
        if model_datetimes is not None:
            shade_windows(ax, model_datetimes, train_lo, train_hi, test_lo, test_hi)
        ax.plot(x[window_mask], np.asarray(series)[window_mask], color=color, lw=1.0)
        if log_scale:
            ax.set_yscale("log")
        if label in ("Net surface\nheat flux (W/m^2)", "Bottom - top\ndensity (kg/m^3)",
                     "Top - bottom\ntemperature (deg C)"):
            ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    if model_datetimes is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", fontsize=8)
        fig.autofmt_xdate()

    title = "Ri-stability-function fit: alpha/n alongside their physical drivers"
    if ri_memory_hours is not None:
        title += f"  (alpha/n EMA memory: {ri_memory_hours:g} h)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out or os.path.join(result_dir, "ri_diagnostics_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
