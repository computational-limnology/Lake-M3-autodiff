"""
Gradient scan: the loss gradient of every yearly window over the whole record,
per parameter, with an alarm for exploding (|g| > threshold) or non-finite
gradients. Diagnostic for the windowed calibration (calibrate_M3_jax.py
--windows years); uses the same pieces (src/windows.py).

What is computed
  For every year whose window fits in the record (default 1 Feb + 8016 h,
  28-day unscored spin-up) the window is started from the state of ONE
  continuous reference run with the current model_params.csv, run with the
  same truncated BPTT as the calibration (stop_gradient every --chunk-steps;
  8016 h -> 1336 h chunks), and

      loss_w = sum_v weight_v * SSE_v,w / n_v,w        (window's own mean loss)
      g_w,p  = d loss_w / d log(p)                      (= p * d loss_w / d p)

  weight_v = 1 / var(obs_v) over the record, as in the calibration
  (obs_weight). g is dimensionless and comparable across parameters and
  years: g = 1 means a 1 % increase of p changes the window's normalised
  loss by ~0.01. The calibration's (pooled) gradient is the
  observation-weighted mean of the window gradients, sum_w (n_w / N) g_w; it
  is printed as a cross-check.

Alarm
  Any |g_w,p| > --threshold (default 10) or any non-finite gradient/loss is
  listed as ALARM and the script exits with status 2 (0 otherwise), so it can
  be used in scripts. Windows without scored observations have loss 0 and
  gradient 0 (listed, n_obs = 0).

Parallelism
  Windows run in parallel over all CPU cores (pmap over --devices CPU devices,
  default = number of cores, one compute thread each; vmap within a device).

Usage (from the project root)
  python src/gradient_scan.py Ravn
  python src/gradient_scan.py Ravn --params wind_factor,seiche_alpha,k_max,sbl_drho --threshold 10
  python src/gradient_scan.py Ravn --record-end 1994-12-31 --devices 4      # quick test

Default parameters: those in <lake>/calibration_result.csv (the last
calibration), else wind_factor,seiche_alpha,k_max. Outputs in
<lake>/diagnostics/: gradient_scan.csv (one row per window) and
gradient_scan.png (gradient per year and parameter, alarm years marked).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _early_device_flags(argv):
    """Device flags must reach XLA before JAX is imported (by windows.py).
    Default: one CPU device per core, single-threaded."""
    n = int(argv[argv.index("--devices") + 1]) if "--devices" in argv else (os.cpu_count() or 1)
    flags = os.environ.get("XLA_FLAGS", "")
    if n > 1 and "xla_force_host_platform_device_count" not in flags:
        flags += f" --xla_force_host_platform_device_count={n}"
    if "--multi-thread" not in argv and "xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = flags.strip()
    if "--xla-warnings" not in argv:   # hide "may be stuck / unstuck" spam with several devices
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


if __name__ == "__main__":
    _early_device_flags(sys.argv)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import windows as W  # noqa: E402
from calibrate_M3_jax import VAR_FIELDS, obs_weight  # noqa: E402

DEFAULT_PARAMS = ("wind_factor", "seiche_alpha", "k_max")


def default_param_names(lake_dir):
    path = os.path.join(lake_dir, "calibration_result.csv")
    if os.path.exists(path):
        names = pd.read_csv(path)["parameter"].astype(str).tolist()
        if names:
            return names, f"from {os.path.relpath(path)}"
    return list(DEFAULT_PARAMS), "default"


def make_scan(setup, windows_df, variables, weights, chunk_steps, window_steps, n_devices):
    """Returns run(theta) -> (loss (B,), sse {v: (B,)}, grad {p: (B,)}),
    one entry per window, computed in parallel."""
    fields = tuple(dict.fromkeys(f for v in variables for f in VAR_FIELDS[v]))
    sim = W.make_window_simulator(setup["geom"], setup["params"], chunk_steps=chunk_steps, outputs=fields)
    F = W.window_forcing(setup["forcing"], windows_df, window_steps)
    OBS, counts = {}, {}
    for v in variables:
        o = W.window_obs(setup["obs"][v], windows_df)
        m = o["mask"] & o["valid"].reshape(o["valid"].shape + (1,) * (o["mask"].ndim - 2))
        OBS[v] = dict(idx=o["step_idx"], val=o["values"], mask=m.astype(float))
        counts[v] = OBS[v]["mask"].reshape(len(windows_df), -1).sum(axis=1)
    # each window's own mean loss: weight_v / n_v,w
    scale = {v: jnp.asarray(weights[v] / np.maximum(counts[v], 1.0)) for v in variables}

    def window_loss(theta, state0, forcing, obsw, sc):
        out = sim(theta, state0, forcing)
        total, sse = 0.0, {}
        for v in variables:
            o = obsw[v]
            pred = sum(out[f] for f in VAR_FIELDS[v])[o["idx"]]
            sse[v] = jnp.sum((pred - o["val"]) ** 2 * o["mask"])
            total = total + sc[v] * sse[v]
        return total, sse

    vg = jax.value_and_grad(window_loss, has_aux=True)

    def per_window(theta, state0, forcing, obsw, sc):
        (loss, sse), g = vg(theta, state0, forcing, obsw, sc)
        return loss, sse, g

    def run(theta, states):
        st = W.window_start_states(states, windows_df)
        f = jax.jit(lambda th, a: W.run_batched(per_window, th, a, n_devices=n_devices))
        loss, sse, g = f(theta, (st, F, OBS, scale))
        return (np.asarray(loss), {v: np.asarray(x) for v, x in sse.items()},
                {p: np.asarray(x) for p, x in g.items()})

    return run, counts


def plot_scan(df, names, threshold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.4 * len(names) + 0.8), sharex=True, squeeze=False)
    years = df["year"].to_numpy()
    for ax, p in zip(axes[:, 0], names):
        g = df[f"g_{p}"].to_numpy(float)
        bad = ~np.isfinite(g) | (np.abs(g) > threshold)
        ax.bar(years, np.where(np.isfinite(g), g, 0.0), color=np.where(bad, "tab:red", "tab:blue"))
        for y in years[~np.isfinite(g)]:
            ax.text(y, 0, "NaN", color="tab:red", ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        lim = np.nanmax(np.abs(np.where(np.isfinite(g), g, np.nan))) if np.isfinite(g).any() else 1.0
        if lim > 0.5 * threshold:
            for s in (1, -1):
                ax.axhline(s * threshold, color="tab:red", ls="--", lw=0.8)
        ax.set_ylabel(f"d loss / d log\n{p}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("window year")
    axes[-1, 0].set_xticks(years)
    axes[-1, 0].set_xticklabels([str(y) for y in years], rotation=90 if len(years) > 12 else 0, fontsize=8)
    fig.suptitle(f"Per-window gradient of the window loss (log-parameters); red = |g| > {threshold:g} or non-finite")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--params", default=None,
                    help="comma list (default: parameters of <lake>/calibration_result.csv)")
    ap.add_argument("--variables", default="temp", help="comma list from temp,o2,doc,poc,kd (default temp)")
    ap.add_argument("--temp-obs", default="profiles", help="temperature observations (default profiles)")
    ap.add_argument("--threshold", type=float, default=10.0, help="alarm if |gradient| > this (default 10)")
    ap.add_argument("--record-end", default=None, help="shorten the record (date), e.g. for a quick test")
    ap.add_argument("--years", default=None, help="restrict to these window years (comma list)")
    ap.add_argument("--window-start", default="02-01")
    ap.add_argument("--window-steps", type=int, default=8016)
    ap.add_argument("--spinup-days", type=int, default=28)
    ap.add_argument("--chunk-steps", type=int, default=2000)
    ap.add_argument("--devices", type=int, default=None, help="CPU devices (default: all cores)")
    ap.add_argument("--multi-thread", action="store_true",
                    help="let XLA use several threads per device (default: one per device)")
    ap.add_argument("--xla-warnings", action="store_true", help="show XLA 'may be stuck' messages")
    ap.add_argument("--states-cache", default=None,
                    help="optional .npz cache for the reference-run start states (reused if parameters match)")
    ap.add_argument("--out", default=None, help="output CSV (default <lake>/diagnostics/gradient_scan.csv)")
    a = ap.parse_args()

    t_all = time.time()
    lake_dir = os.path.abspath(a.lake_dir)
    names, src = (([s.strip() for s in a.params.split(",") if s.strip()], "--params") if a.params
                  else default_param_names(lake_dir))
    variables = [v.strip() for v in a.variables.split(",") if v.strip()]

    S = W.build_setup(lake_dir, record_end=a.record_end, temp_obs=a.temp_obs)
    missing = [p for p in names if S["params"].get(p) is None]
    if missing:
        raise SystemExit(f"parameters not in model params: {missing}")
    novar = [v for v in variables if S["obs"].get(v) is None]
    if novar:
        raise SystemExit(f"no observations for {novar}")
    weights = {v: obs_weight(S["obs"][v]) for v in variables}
    years = [int(y) for y in a.years.split(",")] if a.years else None
    win = W.select_windows(S, a.window_start, a.window_steps, a.spinup_days, years=years, require_obs=None)
    if not len(win):
        raise SystemExit("no windows fit the record")
    D = jax.device_count()
    theta = {p: jnp.log(jnp.asarray(float(S["params"][p]))) for p in names}
    print(f"record {S['start'].date()} + {S['n_steps']} h; {len(win)} windows {win['year'].min()}-{win['year'].max()} "
          f"({a.window_steps} h from {a.window_start}, {a.spinup_days}-day spin-up); {D} device(s)")
    print(f"parameters ({src}): " + ", ".join(f"{p}={float(S['params'][p]):.4g}" for p in names))
    print(f"variables: {variables}; alarm threshold |g| > {a.threshold:g}", flush=True)

    t = time.time()
    states = W.reference_states(S, None, cache=a.states_cache)
    print(f"reference run (start states): {time.time() - t:.0f}s", flush=True)

    run, counts = make_scan(S, win, variables, weights, a.chunk_steps, a.window_steps, D)
    t = time.time()
    loss, sse, grad = run(theta, states)
    print(f"gradients of {len(win)} windows: {time.time() - t:.0f}s", flush=True)

    df = pd.DataFrame({"year": win["year"].to_numpy()})
    for v in variables:
        df[f"n_{v}"] = counts[v].astype(int)
        df[f"rmse_{v}"] = np.where(counts[v] > 0, np.sqrt(sse[v] / np.maximum(counts[v], 1)), np.nan)
    df["loss"] = loss
    for p in names:
        df[f"g_{p}"] = grad[p]
    G = np.column_stack([grad[p] for p in names])
    bad = ~np.isfinite(G) | (np.abs(G) > a.threshold)
    df["alarm"] = bad.any(axis=1) | ~np.isfinite(loss)

    # print table
    pd.set_option("display.width", 200)
    fmt = {c: (lambda x: f"{x:+.3g}") for c in df.columns if c.startswith("g_")}
    fmt.update({c: (lambda x: f"{x:.3f}") for c in df.columns if c.startswith("rmse_") or c == "loss"})
    print("\nPer-window gradient d(loss_w)/d(log p):")
    print(df.to_string(index=False, formatters=fmt))
    # summary per parameter
    n_tot = sum(counts[v] for v in variables)
    wts = n_tot / max(n_tot.sum(), 1)
    print(f"\n{'parameter':>14} {'median|g|':>10} {'max|g|':>10} {'year(max)':>9} {'n>thr':>6} {'pooled g':>10}")
    for j, p in enumerate(names):
        g = G[:, j]
        fin = np.isfinite(g)
        ga = np.abs(np.where(fin, g, 0.0))
        pooled = float(np.sum(wts[fin] * g[fin]))
        print(f"{p:>14} {np.median(ga[fin]) if fin.any() else np.nan:10.3g} {ga.max():10.3g} "
              f"{int(df['year'][int(np.argmax(ga))]):9d} {int(bad[:, j].sum()):6d} {pooled:+10.3g}")
    print("(pooled g = observation-weighted mean over windows = the calibration's windowed gradient, "
          "all windows as training windows)")

    out = a.out or os.path.join(lake_dir, "diagnostics", "gradient_scan.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    png = os.path.splitext(out)[0] + ".png"
    plot_scan(df, names, a.threshold, png)
    print(f"\nsaved {os.path.relpath(out)} and {os.path.relpath(png)}  ({time.time() - t_all:.0f}s total)")

    if bad.any() or not np.all(np.isfinite(loss)):
        print("\n" + "!" * 72)
        print(f"ALARM: {int(bad.sum())} gradient value(s) with |g| > {a.threshold:g} or non-finite")
        for i, j in zip(*np.nonzero(bad)):
            print(f"  {int(df['year'][i])}  {names[j]:>14}  g = {G[i, j]:+.4g}")
        for i in np.nonzero(~np.isfinite(loss))[0]:
            print(f"  {int(df['year'][i])}  loss is non-finite")
        print("!" * 72)
        sys.exit(2)
    print(f"\nOK: all gradients finite and |g| <= {a.threshold:g}")


if __name__ == "__main__":
    main()
