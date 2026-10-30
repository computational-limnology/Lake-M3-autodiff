"""
Gradient-based attribution of changes in the lake's labile particulate
organic carbon (POC-labile, `pocl`) to catchment inflows, with time lags.

Two lake-model runs are compared, identical except for the two inflow-driven
forcing series the catchment model (`catchment_scs_cn.py`) produces:

  * TP     -- in-lake TP [ug/L] from L0001-WQ-inflow*.csv (drives NPP)
  * carbon -- OC load [g/h] = oc * discharge / 24 from oc_load_file*.csv
              (partitioned into DOCr/DOCl/POCr/POCl on entry)

  scenario:        attribution/oc_load_file.csv + L0001-WQ-inflow.csv
                   (with the spring agricultural TP pulse)
  counterfactual:  attribution/oc_load_file_nopulse.csv + ..._nopulse.csv
                   (or --baseline original: the lake folder's own files)

Target quantity y(t): lake-mean labile POC concentration [g/m3],
sum(pocl) / lake volume, at a set of target dates.

Method
------
1. Forward runs give the change dy(t) = y_scenario(t) - y_counterfactual(t).

2. Integrated gradients (Sundararajan et al. 2017) along the straight path
   x(a) = x_cf + a (x_sc - x_cf), a in [0, 1], over BOTH inflow series:
       IG_i = (x_sc,i - x_cf,i) * mean_a( d y(t_target) / d x_i (x(a)) )
   evaluated with --ig-steps midpoints. Every hourly input value gets its own
   contribution, so summing over input times gives each pathway's share
   (TP vs OC) and the contributions' distribution over (target - input) time
   gives the time lag. Gradients are exact reverse-mode derivatives through
   the full simulation (no gradient truncation between the inflow and the
   target date).

3. How certain is the attribution? Reported per target:
   * completeness: |sum(IG) - dy| / |dy|. IG is exact in the limit of many
     steps, so a small residual means the path integral is resolved and the
     inflow change accounts for the whole dy (only the inflows differ between
     the two runs, so by construction nothing else can).
   * split-half agreement: IG recomputed from the even and the odd path
     points separately; their relative difference is the numerical
     uncertainty of the attribution and lag estimates.
   * signal size: dy relative to y_counterfactual, and relative to the
     model's own short-term variability of y (std of y minus its 30-day
     running mean) -- a change smaller than that would be hard to detect in
     observations even if the model attributes it cleanly.
   An attribution is flagged "robust" when completeness < 10%, split-half
   difference < 25% (a conservative measure, since each half has only half
   the path points), the median lag of both halves agrees within 2 days, and
   |dy| exceeds both 5% of y and the short-term variability.

4. Time lag per target: contribution-weighted median lag and 5-95% range of
   |IG| over input days, the lag of the largest single-day contribution,
   and the same split by pathway.

Usage (from the project root; runtime grows with ig-steps x targets x
simulation length, each being one backward pass through the whole run):
    python attribution/attribute_pocl.py [--lake-dir Ravn] [--start 2023-01-01]
        [--end 2023-07-31] [--targets 2023-04-15,2023-05-15,2023-06-15,2023-07-15]
        [--ig-steps 4] [--baseline nopulse|original]
Keep the run below ~7000 hourly steps: the model's documented stable gradient
horizon is ~7500-8000 steps.
"""
import argparse
import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from processBased_lakeModel_functions import (  # noqa: E402
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile,
    provide_phosphorus, provide_carbon, get_lake_config, get_model_params, get_run_config, get_ice_and_snow,
)
from jax_lakeModel_functions import make_initial_state_full, full_step, default_params, default_geometry_wq  # noqa: E402
from run_M3_jax import build_forcing_series, add_wq_forcing_series  # noqa: E402


def setup(lake_dir, start, end):
    """Lake configuration, meteorology and initial state for [start, end]."""
    cwd = os.getcwd()
    os.chdir(lake_dir)
    try:
        lc = get_lake_config("./lake_config.csv", 1)
        mp = get_model_params("./model_params.csv", 1)
        rc = get_run_config("./run_config.csv", 1)
        ias = get_ice_and_snow("./ice_and_snow.csv", 1)
        nx = int(rc["nx"]); dt = float(rc["dt"]); dx = float(rc["dx"])
        area, depth, volume, hw = get_hypsography(hypsofile=rc["hypso_ini_file"], dx=dx, nx=nx,
                                                  outflow_depth=float(lc["outflow_depth"]))
        n_steps = int((end - start) / pd.Timedelta(seconds=dt)) + 1
        st = np.arange(1, (n_steps + 1) * dt, dt)[:n_steps]
        met = provide_meteorology(meteofile=rc["meteo_ini_file"], windfactor=1.0, lat=lc["Latitude"],
                                  lon=lc["Longitude"], elev=lc["Elevation"], startDate=start)
        u_ini = initial_profile(initfile=rc["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=start)
        wq = wq_initial_profile(initfile=rc["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=start)
        forcing_met = build_forcing_series(met, st)
        geom = default_geometry_wq(area=area, depth=depth, volume=volume, dx=dx, dt=dt,
                                   latitude=float(lc["Latitude"]), altitude=float(lc["Elevation"]),
                                   hypso_weight=hw, mean_depth=float(np.sum(volume) / np.max(area)),
                                   hydro_res_time_hr=float(mp["hydro_res_time"]) * 8760)
        params = default_params(mp, ias, diffusion_method=str(rc.get("diffusion_method", "hendersonSellers")))
        tb = lambda x: x.strip().lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)
        ice = dict(ice=tb(ias["ice"]), Hi=ias["Hi"], Hs=ias["Hs"], Hsi=ias["Hsi"], iceT=ias["iceT"],
                   rho_snow=ias["rho_snow"])
        state0 = make_initial_state_full(
            jnp.asarray(deepcopy(u_ini)), jnp.asarray(wq[0]), jnp.asarray(wq[1] * 0.75),
            jnp.asarray(wq[1] * 0.25), jnp.asarray(0.5 * volume), jnp.asarray(0.5 * volume), nx, **ice)
        orig_files = (os.path.abspath(rc["oc_load_file"]), os.path.abspath(rc["tp_ini_file"]))
    finally:
        os.chdir(cwd)
    return dict(st=st, dt=dt, start=start, forcing_met=forcing_met, geom=geom, params=params,
                state0=state0, volume=np.asarray(volume), orig_files=orig_files,
                diffusion_method=params["diffusion_method"])


def inflow_series(oc_file, tp_file, start, st):
    """Hourly TP [ug/L] and OC load [g/h] forcing, built exactly as the lake
    model's own drivers build them (provide_* + add_wq_forcing_series)."""
    tp = provide_phosphorus(tpfile=tp_file, startingDate=start, startTime=1)
    oc = provide_carbon(ocloadfile=oc_file, startingDate=start, startTime=1).dropna(subset=["oc"])
    f = add_wq_forcing_series({}, tp, oc, st)
    return jnp.asarray(f["TP"]), jnp.asarray(f["carbon"])


def make_simulator(S):
    vol_total = float(np.sum(S["volume"]))
    geom, params = S["geom"], S["params"]

    @jax.checkpoint
    def body(s, f):
        ns, _ = full_step(s, f, geom, params)
        return ns, jnp.sum(ns.pocl) / vol_total

    def simulate(tp, carbon):
        forcing = dict(S["forcing_met"], TP=tp, carbon=carbon)
        _, y = lax.scan(body, S["state0"], forcing)
        return y  # lake-mean labile POC [g/m3], one value per hourly step

    return simulate


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return np.nan
    return float(np.interp(q * c[-1], c, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake-dir", default=os.path.join(ROOT, "Ravn"))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-07-31")
    ap.add_argument("--targets", default="2023-04-15,2023-05-15,2023-06-15,2023-07-15")
    ap.add_argument("--ig-steps", type=int, default=4, help="path points (even number, >= 2)")
    ap.add_argument("--baseline", choices=["nopulse", "original"], default="nopulse",
                    help="counterfactual inflows: catchment model without the spring pulse (default), "
                         "or the lake folder's original oc_load/TP files")
    ap.add_argument("--scenario-oc", default=os.path.join(HERE, "oc_load_file.csv"))
    ap.add_argument("--scenario-tp", default=os.path.join(HERE, "L0001-WQ-inflow.csv"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    if args.ig_steps < 2 or args.ig_steps % 2:
        ap.error("--ig-steps must be an even number >= 2 (split-half check)")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    targets = [pd.Timestamp(t.strip()) for t in args.targets.split(",") if t.strip()]
    if any(t > end or t <= start for t in targets):
        ap.error("all --targets must lie inside (start, end]")

    S = setup(os.path.abspath(args.lake_dir), start, end)
    st, dt = S["st"], S["dt"]
    n = len(st)
    if n > 7000:
        print(f"WARNING: {n} steps exceeds the ~7000-step stable gradient horizon; gradients may degrade.")
    if args.baseline == "nopulse":
        cf_oc, cf_tp = os.path.join(HERE, "oc_load_file_nopulse.csv"), os.path.join(HERE, "L0001-WQ-inflow_nopulse.csv")
    else:
        cf_oc, cf_tp = S["orig_files"]
    tp_sc, c_sc = inflow_series(args.scenario_oc, args.scenario_tp, start, st)
    tp_cf, c_cf = inflow_series(cf_oc, cf_tp, start, st)
    d_tp, d_c = tp_sc - tp_cf, c_sc - c_cf

    times = start + pd.to_timedelta(st, unit="s")
    t_idx = np.array([int(np.argmin(np.abs(times - t))) for t in targets])
    print(f"Lake run {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({n} hourly steps), closure {S['diffusion_method']}; "
          f"counterfactual: {args.baseline}; targets: {', '.join(f'{t:%Y-%m-%d}' for t in targets)}")

    simulate = make_simulator(S)
    sim_jit = jax.jit(simulate)
    t0 = time.time()
    y_sc = np.asarray(sim_jit(tp_sc, c_sc)); y_cf = np.asarray(sim_jit(tp_cf, c_cf))
    print(f"  forward runs done ({time.time() - t0:.0f}s)")
    dy = y_sc - y_cf

    f_targets = lambda tp, c: simulate(tp, c)[t_idx]
    jac = jax.jit(jax.jacrev(f_targets, argnums=(0, 1)))
    m = args.ig_steps
    alphas = (np.arange(m) + 0.5) / m
    g_tp = np.zeros((m, len(t_idx), n)); g_c = np.zeros((m, len(t_idx), n))
    for k, a in enumerate(alphas):
        t0 = time.time()
        jt, jc = jac(tp_cf + a * d_tp, c_cf + a * d_c)
        g_tp[k], g_c[k] = np.asarray(jt), np.asarray(jc)
        bad = ~np.isfinite(g_tp[k]).all() or ~np.isfinite(g_c[k]).all()
        print(f"  path point {k + 1}/{m} (alpha={a:.3f}) done ({time.time() - t0:.0f}s)"
              + ("  WARNING: non-finite gradients" if bad else ""))

    dtp, dc = np.asarray(d_tp), np.asarray(d_c)
    ig_tp = g_tp.mean(0) * dtp; ig_c = g_c.mean(0) * dc                       # (targets, steps)
    ig_half_t = [g_tp[h::2].mean(0) * dtp + g_c[h::2].mean(0) * dc for h in (0, 1)]   # (targets, steps)
    ig_half = [x.sum(1) for x in ig_half_t]
    if not np.any(dc):
        print("  note: the OC load is identical in both runs, so the OC pathway is zero by construction "
              "(use --baseline original to compare against the lake folder's own OC load)")

    # daily aggregation for lags/plots
    day = pd.DatetimeIndex(times).normalize()
    days = day.unique()
    agg = lambda x: pd.DataFrame(x.T, index=day).groupby(level=0).sum().to_numpy().T   # (targets, days)
    ig_tp_d, ig_c_d = agg(ig_tp), agg(ig_c)
    ig_half_d = [agg(x) for x in ig_half_t]

    y_series = pd.Series(y_cf, index=times)
    short_var = float((y_series - y_series.rolling("30D", center=True).mean()).std())

    rows = []
    for j, (t, ti) in enumerate(zip(targets, t_idx)):
        dyt = float(dy[ti]); yt = float(y_cf[ti])
        tot_tp, tot_c = float(ig_tp[j].sum()), float(ig_c[j].sum())
        tot = tot_tp + tot_c
        completeness = abs(tot - dyt) / max(abs(dyt), 1e-30)
        split = abs(ig_half[0][j] - ig_half[1][j]) / max(abs(tot), 1e-30)
        lag_days = np.asarray((t.normalize() - days).days, dtype=float)
        keep = lag_days >= 0
        w_all = np.abs(ig_tp_d[j] + ig_c_d[j])[keep]
        lags = lag_days[keep]
        row = dict(
            target=f"{t:%Y-%m-%d}", y_counterfactual=yt, y_scenario=float(y_sc[ti]), dy=dyt,
            dy_rel=dyt / max(abs(yt), 1e-30), dy_over_shortterm_var=abs(dyt) / max(short_var, 1e-30),
            attributed_total=tot, share_TP=tot_tp / tot if tot else np.nan, share_OC=tot_c / tot if tot else np.nan,
            completeness_residual=completeness, split_half_diff=split,
            lag_median_d=weighted_quantile(lags, w_all, 0.5),
            lag_p05_d=weighted_quantile(lags, w_all, 0.05), lag_p95_d=weighted_quantile(lags, w_all, 0.95),
            lag_peak_d=float(lags[np.argmax(w_all)]) if w_all.size else np.nan,
            lag_median_halves_d=[weighted_quantile(lags, np.abs(h[j])[keep], 0.5) for h in ig_half_d],
            lag_median_TP_d=weighted_quantile(lags, np.abs(ig_tp_d[j])[keep], 0.5),
            lag_median_OC_d=weighted_quantile(lags, np.abs(ig_c_d[j])[keep], 0.5),
        )
        lh = row.pop("lag_median_halves_d")
        row["lag_median_split_half_diff_d"] = abs(lh[0] - lh[1])
        # The split-half difference compares two half-resolution estimates, so it
        # overstates the error of the full estimate (midpoint rule: roughly by a
        # factor 2-4); completeness is the decisive check of the attribution sum.
        row["robust"] = bool(completeness < 0.1 and split < 0.25 and row["lag_median_split_half_diff_d"] <= 2
                             and abs(row["dy_rel"]) > 0.05 and row["dy_over_shortterm_var"] > 1.0)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "attribution_pocl_summary.csv"), index=False)

    print("\nPOC-labile (lake mean) change attributed to catchment inflows:")
    for r in rows:
        print(f"  {r['target']}: y_cf {r['y_counterfactual']:.4f} -> y_sc {r['y_scenario']:.4f} g/m3 "
              f"(dy {r['dy']:+.4f}, {r['dy_rel']:+.1%}; {r['dy_over_shortterm_var']:.1f}x short-term variability)")
        print(f"      attributed {r['attributed_total']:+.4f} = TP {r['share_TP']:.0%} + OC {r['share_OC']:.0%}; "
              f"completeness residual {r['completeness_residual']:.1%}, split-half diff {r['split_half_diff']:.1%}"
              f" -> {'ROBUST' if r['robust'] else 'not robust'}")
        print(f"      lag: median {r['lag_median_d']:.0f} d (5-95%: {r['lag_p05_d']:.0f}-{r['lag_p95_d']:.0f} d), "
              f"peak {r['lag_peak_d']:.0f} d; median by pathway TP {r['lag_median_TP_d']:.0f} d, "
              f"OC {r['lag_median_OC_d']:.0f} d; split-half lag diff {r['lag_median_split_half_diff_d']:.1f} d")

    np.savez(os.path.join(args.out_dir, "attribution_pocl_results.npz"),
             times=np.asarray(times.astype("datetime64[s]")), y_scenario=y_sc, y_counterfactual=y_cf,
             targets=np.array([str(t) for t in targets]), days=np.asarray(days.astype("datetime64[s]")),
             ig_tp_daily=ig_tp_d, ig_oc_daily=ig_c_d, dtp=dtp, dcarbon=dc,
             grad_tp_mean=g_tp.mean(0), grad_carbon_mean=g_c.mean(0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(times, y_cf, label=f"counterfactual ({args.baseline})"); ax[0].plot(times, y_sc, label="scenario")
    ax[0].set_ylabel("lake-mean POC-labile (g/m3)"); ax[0].legend()
    ax[1].plot(times, dy, color="k"); ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_ylabel("dy scenario - counterfactual")
    for t in targets:
        for a_ in ax[:2]:
            a_.axvline(t, color="tab:red", ls=":", lw=0.8)
    ax[2].plot(times, np.asarray(tp_sc), label="TP scenario"); ax[2].plot(times, np.asarray(tp_cf), label="TP counterfactual")
    ax2 = ax[2].twinx(); ax2.plot(times, np.asarray(c_sc) * 24 / 1e3, color="tab:green", lw=0.7, label="OC load scenario")
    ax2.plot(times, np.asarray(c_cf) * 24 / 1e3, color="tab:olive", lw=0.7, label="OC load cf")
    ax[2].set_ylabel("in-lake TP (ug/L)"); ax2.set_ylabel("OC load (kg/day)"); ax[2].legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "attribution_pocl_timeseries.png"), dpi=120)

    def draw_contributions(a_, j, t, legend):
        a_.bar(days, ig_tp_d[j], width=1.0, color="tab:blue", label="via TP")
        a_.bar(days, ig_c_d[j], width=1.0, bottom=ig_tp_d[j], color="tab:green", label="via OC load")
        a_.axvline(t, color="tab:red", ls=":", lw=1.0)
        lo = min(0.0, float(np.min(ig_tp_d[j])), float(np.min(ig_tp_d[j] + ig_c_d[j])))
        hi = max(0.0, float(np.max(ig_tp_d[j])), float(np.max(ig_tp_d[j] + ig_c_d[j])))
        a_.set_ylim(lo * 1.1, hi * 1.1 if hi > 0 else 1e-12)
        r = rows[j]
        a_.set_title(f"Target {t:%Y-%m-%d}: dy {r['dy']:+.4f} g/m3, attributed {r['attributed_total']:+.4f} "
                     f"(TP {r['share_TP']:.0%}); median lag {r['lag_median_d']:.0f} d", fontsize=10)
        a_.set_ylabel("contribution (g/m3)")
        if legend:
            a_.legend(fontsize=8)

    # all targets stacked in one figure
    fig, axs = plt.subplots(len(targets), 1, figsize=(13, 2.8 * len(targets)), sharex=True, squeeze=False)
    for j, t in enumerate(targets):
        draw_contributions(axs[j, 0], j, t, legend=(j == 0))
    axs[-1, 0].set_xlabel("input (inflow) date")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "attribution_pocl_lags.png"), dpi=120)

    # one file per target, same panel as above
    lag_files = []
    for j, t in enumerate(targets):
        fig, a_ = plt.subplots(figsize=(13, 3.6))
        draw_contributions(a_, j, t, legend=True)
        a_.set_xlabel("input (inflow) date")
        fname = f"attribution_pocl_lags_{t:%Y-%m-%d}.png"
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, fname), dpi=120)
        plt.close(fig)
        lag_files.append(fname)
    print(f"\nSaved attribution_pocl_summary.csv, attribution_pocl_results.npz, "
          f"attribution_pocl_timeseries.png, attribution_pocl_lags.png, {', '.join(lag_files)} to {args.out_dir}")


if __name__ == "__main__":
    main()
