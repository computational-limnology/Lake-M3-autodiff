"""
Where does the temperature error come from? Profile-based error diagnosis of
one or more model runs (e.g. process-based and MCL) against the sparse manual
temperature profiles of a lake. High-frequency (buoy) rows are excluded by
default (`--temp-obs profiles`).

Usage:
    python src/diagnose_temperature_errors.py Ravn \
        --run PB=Ravn/res_lake1_jax_full.npz --run MCL=Ravn/res_lake1_mcl_hybrid.npz \
        [--match daily_mean|midnight|noon] [--out-dir Ravn/diagnostics]

Analyses (each writes a CSV and most a figure into --out-dir):

A  Heat content vs vertical distribution. Each profile with full-column
   coverage is interpolated onto the model layers it spans. With volume
   weights V_i, the volume-weighted MSE splits exactly into
       MSE_V = b**2 + mean_V(r'**2),
   b = Tbar_model - Tbar_obs (lake-mean temperature bias = heat-content error)
   and r'(z) = r(z) - b (distribution error). Reported overall, by season,
   by period and per profile; plus the heat-content time series.
B  Bias / RMSE by layer (epi 0-5 m, meta 5-15 m, hypo >15 m) x season x period,
   and a year x month map of the layer biases.
C  Stratification: surface-bottom difference, thermocline depth, Schmidt
   stability per profile; stratification onset/end per year (top-bottom
   difference crossing 1 degC; observed dates are intervals between
   consecutive profiles); hypolimnion bias at onset vs during summer; winter
   inverse stratification as an ice proxy (no ice observations available).
D  Drift: annual residual per layer with linear trends (degC/decade), and
   annual means of the meteorological forcing with the best single step
   change (max t-statistic split).
E  Weather links: residuals vs the preceding 7-day mean wind, air temperature
   and shortwave; spring (Mar-May) wind vs hypolimnion bias at onset.
F  Observation checks: profiles with outlying heat-content bias or
   distribution error (robust z > 3), and the sensitivity to the time of day
   the model is sampled at (profiles carry dates only, 00:00).

The summary at the end applies the decision rules of the analysis plan.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_tools import load_run_config, load_temperature_dataframe, resolve_data_path  # noqa: E402

LAYERS = (("epi", 0.0, 5.0), ("meta", 5.0, 15.0), ("hypo", 15.0, 1e9))
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
STRAT_DT = 1.0      # degC top-bottom difference defining stratification
HYPO_TOP = 15.0     # m


def period_of(year):
    return "1989-1999" if year < 2000 else ("2000-2009" if year < 2010 else "2010-2024")


def density(t):
    t = np.asarray(t, float)
    return (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2 + 1.001685e-4 * t ** 3
            - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_run(path, match):
    with np.load(path) as d:
        temp = np.asarray(d["temp"], float)
        depth = np.asarray(d["depth"], float)
        volume = np.asarray(d["volume"], float) if "volume" in d.files else None
        start = pd.Timestamp(str(d["start_time"])) if "start_time" in d.files else None
    n_days = temp.shape[0] // 24
    daily = temp[: n_days * 24].reshape(n_days, 24, -1)
    if match == "daily_mean":
        tday = daily.mean(axis=1)
    elif match == "midnight":
        tday = daily[:, 0, :]
    else:
        tday = daily[:, 12, :]
    return dict(path=path, depth=depth, volume=volume, start=start, daily=tday,
                daily_mean=daily.mean(axis=1), daily_00=daily[:, 0, :], daily_12=daily[:, 12, :])


def day_index(dates, start):
    return ((pd.DatetimeIndex(dates).normalize() - start.normalize()) / pd.Timedelta(days=1)).astype(int)


# ---------------------------------------------------------------------------
# per-point and per-profile matching
# ---------------------------------------------------------------------------

def match_points(obs, run, tday=None):
    tday = run["daily"] if tday is None else tday
    di = day_index(obs["datetime"], run["start"])
    ok = (di >= 0) & (di < tday.shape[0])
    o = obs[ok].copy()
    di = np.asarray(di[ok])
    k = np.abs(run["depth"][None, :] - o["Depth_meter"].to_numpy()[:, None]).argmin(axis=1)
    o["model"] = tday[di, k]
    o["resid"] = o["model"] - o["Water_Temperature_celsius"]
    o["layer"] = pd.cut(o["Depth_meter"], [-1, 5, 15, 1e9], labels=["epi", "meta", "hypo"], right=False)
    o["season"] = o["datetime"].dt.month.map(SEASON)
    o["year"] = o["datetime"].dt.year
    o["period"] = o["year"].map(period_of)
    return o


def profile_table(obs, run, min_depths=5, min_cov_frac=0.6):
    """One row per profile with full-column coverage: interpolated obs and
    model on the model layers inside [min obs depth, max obs depth]."""
    depth, vol = run["depth"], run["volume"]
    zmax = depth[-1]
    rows, store = [], {}
    for dt, g in obs.groupby("datetime"):
        g = g.dropna(subset=["Water_Temperature_celsius"]).groupby("Depth_meter")["Water_Temperature_celsius"].mean()
        if len(g) < min_depths or g.index.min() > 1.0 or g.index.max() < min_cov_frac * zmax:
            continue
        di = int(day_index([dt], run["start"])[0])
        if di < 0 or di >= run["daily"].shape[0]:
            continue
        sel = (depth >= g.index.min() - 0.25) & (depth <= g.index.max() + 0.25)
        z = depth[sel]
        t_obs = np.interp(z, g.index.to_numpy(), g.to_numpy())
        t_mod = run["daily"][di, sel]
        v = vol[sel]
        w = v / v.sum()
        r = t_mod - t_obs
        b = float(np.sum(w * r))
        rp = r - b
        rows.append(dict(datetime=dt, n_obs=len(g), zmax_obs=float(g.index.max()),
                         Tbar_obs=float(np.sum(w * t_obs)), Tbar_mod=float(np.sum(w * t_mod)),
                         bias_hc=b, mse_v=float(np.sum(w * r ** 2)), mse_hc=b ** 2,
                         mse_dist=float(np.sum(w * rp ** 2))))
        store[dt] = (z, t_obs, t_mod, v)
    pt = pd.DataFrame(rows)
    pt["season"] = pt["datetime"].dt.month.map(SEASON)
    pt["year"] = pt["datetime"].dt.year
    pt["period"] = pt["year"].map(period_of)
    return pt, store


def split_summary(pt, by=None):
    def agg(g):
        hc, dist = g["mse_hc"].mean(), g["mse_dist"].mean()
        return pd.Series(dict(n=len(g), rmse_v=np.sqrt(hc + dist), mean_bias_hc=g["bias_hc"].mean(),
                              mse_hc=hc, mse_dist=dist, share_hc=hc / (hc + dist)))
    if by is None:
        return agg(pt).to_frame().T.assign(group="all")
    return pt.groupby(by).apply(agg).reset_index()


# ---------------------------------------------------------------------------
# stratification metrics
# ---------------------------------------------------------------------------

def strat_metrics(z, t, v, g=9.81):
    rho = density(t)
    dT = float(np.mean(t[z <= 1.0]) - t[-1]) if np.any(z <= 1.0) else float(t[0] - t[-1])
    n2 = np.abs(np.diff(rho)) / np.diff(z)
    zt = float(0.5 * (z[np.argmax(n2)] + z[np.argmax(n2) + 1])) if dT > STRAT_DT else np.nan
    area0 = v[0] / (z[1] - z[0]) if len(z) > 1 else 1.0
    zv = np.sum(z * v) / np.sum(v)
    schmidt = float(g / area0 * np.sum((z - zv) * rho * v))
    hypo = float(np.mean(t[z >= HYPO_TOP])) if np.any(z >= HYPO_TOP) else np.nan
    return dT, zt, max(schmidt, 0.0), hypo


def strat_table(store):
    rows = []
    for dt, (z, to, tm, v) in store.items():
        o = strat_metrics(z, to, v)
        m = strat_metrics(z, tm, v)
        rows.append(dict(datetime=dt, dT_obs=o[0], dT_mod=m[0], zth_obs=o[1], zth_mod=m[1],
                         schmidt_obs=o[2], schmidt_mod=m[2], hypo_obs=o[3], hypo_mod=m[3]))
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def strat_timing(st, run):
    """Observed onset/end intervals per year from consecutive profiles, and the
    model's own dates from its daily top(<=1 m)-bottom difference."""
    depth = run["depth"]
    top = run["daily"][:, depth <= 1.0].mean(axis=1)
    dT_model = top - run["daily"][:, -1]
    days = run["start"].normalize() + pd.to_timedelta(np.arange(len(dT_model)), unit="D")
    s_model = pd.Series(dT_model > STRAT_DT, index=days)
    rows = []
    for year, g in st.groupby(st["datetime"].dt.year):
        g = g.sort_values("datetime")
        strat = (g["dT_obs"] > STRAT_DT).to_numpy()
        dts = g["datetime"].to_numpy()
        on_lo = on_hi = end_lo = end_hi = pd.NaT
        for i in range(1, len(g)):
            m = pd.Timestamp(dts[i]).month
            if not strat[i - 1] and strat[i] and 2 <= m <= 7 and pd.isna(on_hi):
                on_lo, on_hi = pd.Timestamp(dts[i - 1]), pd.Timestamp(dts[i])
            if strat[i - 1] and not strat[i] and 8 <= m <= 12:
                end_lo, end_hi = pd.Timestamp(dts[i - 1]), pd.Timestamp(dts[i])
        sy = s_model[str(year)]
        sy_s = sy.rolling(7, center=True, min_periods=1).mean() > 0.5   # ignore 1-3 day flips
        spring = sy_s[(sy_s.index.month >= 2) & (sy_s.index.month <= 7)]
        autumn = sy_s[(sy_s.index.month >= 8)]
        on_mod = spring.index[np.argmax(spring.to_numpy())] if spring.any() else pd.NaT
        end_mod = pd.NaT
        if autumn.any() and not autumn.iloc[-1]:
            last_true = autumn[autumn].index[-1]
            end_mod = last_true + pd.Timedelta(days=1)

        def err(mod, lo, hi):
            if pd.isna(mod) or pd.isna(lo):
                return np.nan
            if lo <= mod <= hi:
                return 0.0
            return float((mod - (lo if mod < lo else hi)) / pd.Timedelta(days=1))
        rows.append(dict(year=year, onset_obs_from=on_lo, onset_obs_to=on_hi, onset_mod=on_mod,
                         onset_err_days=err(on_mod, on_lo, on_hi),
                         end_obs_from=end_lo, end_obs_to=end_hi, end_mod=end_mod,
                         end_err_days=err(end_mod, end_lo, end_hi)))
    return pd.DataFrame(rows)


def hypo_onset_persistence(st, timing):
    """Per year: hypolimnion bias at the first stratified profile vs the mean
    hypolimnion bias over the stratified season (Jun-Sep profiles)."""
    rows = []
    st = st.assign(hb=st["hypo_mod"] - st["hypo_obs"])
    for _, r in timing.iterrows():
        if pd.isna(r["onset_obs_to"]):
            continue
        y = st[st["datetime"].dt.year == r["year"]]
        first = y[y["datetime"] == r["onset_obs_to"]]
        summer = y[(y["datetime"].dt.month >= 6) & (y["datetime"].dt.month <= 9) & (y["dT_obs"] > STRAT_DT)]
        if first.empty or summer.empty:
            continue
        rows.append(dict(year=r["year"], hypo_bias_onset=float(first["hb"].iloc[0]),
                         hypo_bias_summer=float(summer["hb"].mean()),
                         hypo_bias_summer_std=float(summer["hb"].std()), n_summer=len(summer)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# drift / forcing
# ---------------------------------------------------------------------------

def trend(years, values):
    ok = np.isfinite(values)
    x, y = np.asarray(years, float)[ok], np.asarray(values, float)[ok]
    if len(x) < 5:
        return np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(x) - 2
    s2 = float(np.sum((y - A @ coef) ** 2) / dof)
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    return 10 * coef[0], 10 * se


def best_step(years, values, min_seg=4):
    y = np.asarray(values, float)
    best = (np.nan, 0.0, np.nan)
    for i in range(min_seg, len(y) - min_seg + 1):
        a, b = y[:i], y[i:]
        s = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = abs(b.mean() - a.mean()) / s if s > 0 else 0.0
        if t > best[1]:
            best = (int(years[i]), t, b.mean() - a.mean())
    return best


def load_met(data_dir, rc):
    path = resolve_data_path(data_dir, rc, "meteo_ini_file", "L0001-MET.csv")
    m = pd.read_csv(path)
    m["datetime"] = pd.to_datetime(m["datetime"])
    cols = {c: c for c in m.columns}
    pick = lambda key: next(c for c in m.columns if key.lower() in c.lower())
    out = pd.DataFrame({
        "datetime": m["datetime"],
        "Tair": m[pick("Air_Temperature")], "RH": m[pick("Relative_Humidity")],
        "Wind": m[pick("Wind_Speed")], "SW": m[pick("Shortwave")], "LW": m[pick("Longwave")],
        "Precip": m[pick("Precipitation")],
    })
    return out.set_index("datetime")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--run", action="append", required=True, help="LABEL=path/to/result.npz (repeatable)")
    ap.add_argument("--temp-obs", choices=("profiles", "all"), default="profiles")
    ap.add_argument("--match", choices=("daily_mean", "midnight", "noon"), default="daily_mean",
                    help="model value compared with a date-only profile (default: daily mean)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = os.path.abspath(args.data_dir)
    out = args.out_dir or os.path.join(data_dir, "diagnostics")
    os.makedirs(out, exist_ok=True)
    rc = load_run_config(data_dir)
    obs = load_temperature_dataframe(data_dir, rc, temp_obs=args.temp_obs)
    obs = obs.dropna(subset=["Water_Temperature_celsius"])
    met = load_met(data_dir, rc)
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = load_run(path, args.match)
        print(f"{label}: {path}  ({runs[label]['daily'].shape[0]} days from {runs[label]['start'].date()})")
    labels = list(runs)
    first = runs[labels[0]]
    obs = obs[(obs["datetime"] >= first["start"].normalize())
              & (obs["datetime"] < first["start"].normalize() + pd.Timedelta(days=first["daily"].shape[0]))]
    print(f"Observations: {args.temp_obs}, {obs['datetime'].nunique()} profile dates, {len(obs)} values; "
          f"model sampled as {args.match}\n")

    summary = {}
    pts, pts_all, sts, tims, hyps = {}, {}, {}, {}, {}
    for lab, run in runs.items():
        P = match_points(obs, run)
        pt, store = profile_table(obs, run)
        st = strat_table(store)
        tim = strat_timing(st, run)
        hyp = hypo_onset_persistence(st, tim)
        pts[lab], pts_all[lab], sts[lab], tims[lab], hyps[lab] = pt, P, st, tim, hyp
        pt.to_csv(f"{out}/A_profile_split_{lab}.csv", index=False)
        split = pd.concat([split_summary(pt), split_summary(pt, "season").rename(columns={"season": "group"}),
                           split_summary(pt, "period").rename(columns={"period": "group"})], ignore_index=True)
        split.to_csv(f"{out}/A_split_{lab}.csv", index=False)
        lay = P.groupby(["layer", "season", "period"], observed=True)["resid"].agg(
            n="size", bias="mean", rmse=lambda r: np.sqrt(np.mean(r ** 2))).reset_index()
        lay.to_csv(f"{out}/B_layer_season_period_{lab}.csv", index=False)
        st.to_csv(f"{out}/C_stratification_{lab}.csv", index=False)
        tim.to_csv(f"{out}/C_timing_{lab}.csv", index=False)
        hyp.to_csv(f"{out}/C_hypo_onset_{lab}.csv", index=False)
        summary[lab] = dict(split=split, layer=lay)

    # ---- D: drift + forcing ---------------------------------------------------
    drift_rows = []
    for lab in labels:
        P = pts_all[lab]
        ann = P.groupby(["year", "layer"], observed=True)["resid"].mean().unstack()
        for layer in ann.columns:
            s, se = trend(ann.index, ann[layer].to_numpy())
            yr, t, d = best_step(ann.index.to_numpy(), ann[layer].to_numpy())
            drift_rows.append(dict(run=lab, series=f"resid_{layer}", trend_per_decade=s, se=se,
                                   step_year=yr, step_t=t, step_size=d))
        hc = pts[lab].groupby("year")["bias_hc"].mean()
        s, se = trend(hc.index, hc.to_numpy())
        yr, t, d = best_step(hc.index.to_numpy(), hc.to_numpy())
        drift_rows.append(dict(run=lab, series="bias_heat_content", trend_per_decade=s, se=se,
                               step_year=yr, step_t=t, step_size=d))
    met_ann = met.resample("YE").mean()
    met_ann.index = met_ann.index.year
    met_ann = met_ann[(met_ann.index >= first["start"].year) & (met_ann.index <= 2024)]
    for c in met_ann.columns:
        s, se = trend(met_ann.index, met_ann[c].to_numpy())
        yr, t, d = best_step(met_ann.index.to_numpy(), met_ann[c].to_numpy())
        drift_rows.append(dict(run="forcing", series=c, trend_per_decade=s, se=se, step_year=yr, step_t=t,
                               step_size=d))
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(f"{out}/D_drift_and_forcing.csv", index=False)

    # ---- E: weather links ------------------------------------------------------
    met_d = met.resample("D").mean()
    w7 = met_d.rolling(7, min_periods=5).mean().shift(1)  # 7 days before the profile date
    weather_rows = []
    for lab in labels:
        P = pts_all[lab].copy()
        P["date"] = P["datetime"].dt.normalize()
        pm = P.groupby(["date", "layer"], observed=True)["resid"].mean().unstack()
        pm = pm.join(w7, how="left")
        for season in ("MAM", "JJA", "SON", "DJF"):
            q = pm[pm.index.month.map(SEASON) == season]
            for layer in ("epi", "meta", "hypo"):
                if layer not in q:
                    continue
                for var in ("Wind", "Tair", "SW"):
                    ok = q[[layer, var]].dropna()
                    if len(ok) >= 10:
                        rho = ok[layer].rank().corr(ok[var].rank())
                        weather_rows.append(dict(run=lab, season=season, layer=layer, driver=f"{var}_7d",
                                                 n=len(ok), spearman=rho))
        hyp = hyps[lab]
        if len(hyp) >= 8:
            spring_wind = met_d["Wind"].groupby(met_d.index.year).apply(
                lambda s: s[(s.index.month >= 3) & (s.index.month <= 5)].mean())
            h = hyp.set_index("year").join(spring_wind.rename("spring_wind"))
            ok = h[["hypo_bias_onset", "spring_wind"]].dropna()
            weather_rows.append(dict(run=lab, season="MAM", layer="hypo@onset", driver="spring_wind_mean",
                                     n=len(ok), spearman=ok.iloc[:, 0].rank().corr(ok.iloc[:, 1].rank())))
    weather = pd.DataFrame(weather_rows)
    weather.to_csv(f"{out}/E_weather_links.csv", index=False)

    # ---- F: observation checks -------------------------------------------------
    flag_rows = []
    for lab in labels:
        pt = pts[lab]
        zb, zd = robust_z(pt["bias_hc"]), robust_z(np.sqrt(pt["mse_dist"]))
        f = pt.assign(z_bias=zb, z_dist=zd)
        f = f[(np.abs(f["z_bias"]) > 3) | (f["z_dist"] > 3)]
        flag_rows.append(f.assign(run=lab))
    flags = pd.concat(flag_rows, ignore_index=True) if flag_rows else pd.DataFrame()
    flags.to_csv(f"{out}/F_flagged_profiles.csv", index=False)
    tod_rows = []
    for lab, run in runs.items():
        for name, arr in (("daily_mean", run["daily_mean"]), ("00:00", run["daily_00"]), ("12:00", run["daily_12"])):
            P = match_points(obs, run, arr)
            for layer in ("epi", "hypo"):
                for season in ("JJA", "DJF"):
                    r = P[(P["layer"] == layer) & (P["season"] == season)]["resid"]
                    tod_rows.append(dict(run=lab, model_time=name, layer=layer, season=season,
                                         bias=r.mean(), rmse=np.sqrt(np.mean(r ** 2))))
    tod = pd.DataFrame(tod_rows)
    tod.to_csv(f"{out}/F_time_of_day.csv", index=False)

    # ---- figures ----------------------------------------------------------------
    colors = dict(zip(labels, ["tab:gray", "tab:red", "tab:blue", "tab:green"]))
    # A1: split by season/period
    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 4.2), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        sp = summary[lab]["split"]
        x = np.arange(len(sp))
        ax.bar(x, sp["mse_hc"], color="tab:orange", label="heat content (b^2)")
        ax.bar(x, sp["mse_dist"], bottom=sp["mse_hc"], color="tab:blue", label="distribution")
        for xi, (sh, rm) in enumerate(zip(sp["share_hc"], sp["rmse_v"])):
            ax.text(xi, (rm ** 2) * 1.02, f"{100 * sh:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(x, sp["group"], rotation=30)
        ax.set_ylabel("volume-weighted MSE (degC^2)")
        ax.set_title(f"{lab}: error split (label = heat-content share)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/A_error_split.png", dpi=130); plt.close(fig)

    # A2: heat content
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    pt0 = pts[labels[0]]
    axes[0].plot(pt0["datetime"], pt0["Tbar_obs"], "k.", ms=3, label="observed")
    for lab in labels:
        axes[0].plot(pts[lab]["datetime"], pts[lab]["Tbar_mod"], ".", ms=2.5, color=colors[lab], label=lab)
        axes[1].plot(pts[lab]["datetime"], pts[lab]["bias_hc"], ".", ms=3, color=colors[lab], label=lab)
        yr = pts[lab].groupby("year").agg(d=("datetime", "median"), b=("bias_hc", "mean"))
        axes[1].plot(yr["d"], yr["b"], "-", color=colors[lab], lw=1.2)
    axes[0].set_ylabel("volume-mean T (degC)"); axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.6); axes[1].set_ylabel("heat-content bias b (degC)")
    axes[1].set_title("per profile (dots) and annual mean (line)")
    fig.tight_layout(); fig.savefig(f"{out}/A_heat_content.png", dpi=130); plt.close(fig)

    # B: year x month maps of layer bias
    for lab in labels:
        P = pts_all[lab]
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
        for ax, layer in zip(axes, ("epi", "meta", "hypo")):
            m = P[P["layer"] == layer].pivot_table(index="year", columns=P["datetime"].dt.month,
                                                   values="resid", aggfunc="mean")
            im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                           extent=(0.5, 12.5, m.index.max() + 0.5, m.index.min() - 0.5))
            ax.set_title(f"{lab}: {layer} bias (model - obs)")
            ax.set_xlabel("month")
        axes[0].set_ylabel("year")
        fig.colorbar(im, ax=axes, label="degC", fraction=0.02)
        fig.savefig(f"{out}/B_bias_map_{lab}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # C: timing + hypolimnion persistence
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for lab in labels:
        t = tims[lab]
        axes[0].plot(t["year"], t["onset_err_days"], "o-", color=colors[lab], ms=4, label=lab)
        axes[1].plot(t["year"], t["end_err_days"], "o-", color=colors[lab], ms=4, label=lab)
        h = hyps[lab]
        axes[2].plot(h["hypo_bias_onset"], h["hypo_bias_summer"], "o", color=colors[lab], label=lab)
    for ax, ttl in zip(axes[:2], ("stratification onset", "end of stratification (autumn turnover)")):
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("model date - observed interval (days)")
        ax.set_title(f"{ttl}: 0 = inside observed interval"); ax.legend(fontsize=8)
    lim = np.nanmax(np.abs(np.concatenate([hyps[l][["hypo_bias_onset", "hypo_bias_summer"]].to_numpy().ravel()
                                           for l in labels] + [np.array([1.0])])))
    axes[2].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
    axes[2].set_xlabel("hypolimnion bias at first stratified profile (degC)")
    axes[2].set_ylabel("mean hypolimnion bias Jun-Sep (degC)")
    axes[2].set_title("is the summer hypolimnion error set at onset?"); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/C_stratification_timing.png", dpi=130); plt.close(fig)

    # C2: structure scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for lab in labels:
        st = sts[lab]
        axes[0].plot(st["dT_obs"], st["dT_mod"], ".", color=colors[lab], ms=4, label=lab)
        axes[1].plot(st["zth_obs"], st["zth_mod"], ".", color=colors[lab], ms=4, label=lab)
        axes[2].plot(st["schmidt_obs"], st["schmidt_mod"], ".", color=colors[lab], ms=4, label=lab)
    for ax, ttl in zip(axes, ("surface - bottom (degC)", "thermocline depth (m)", "Schmidt stability (J/m2)")):
        lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlabel("observed"); ax.set_ylabel("model"); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/C_structure.png", dpi=130); plt.close(fig)

    # D: drift + forcing
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for lab in labels:
        ann = pts_all[lab].groupby(["year", "layer"], observed=True)["resid"].mean().unstack()
        for layer, ls in zip(("epi", "meta", "hypo"), ("-", "--", ":")):
            if layer in ann:
                axes[0].plot(ann.index, ann[layer], ls, marker=".", color=colors[lab], label=f"{lab} {layer}")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].set_ylabel("annual mean residual (degC)")
    axes[0].legend(fontsize=7, ncol=3)
    z = (met_ann - met_ann.mean()) / met_ann.std()
    for c in ("Tair", "Wind", "SW", "LW", "RH"):
        axes[1].plot(z.index, z[c], marker=".", label=c)
    axes[1].set_ylabel("annual forcing mean (z-score)"); axes[1].legend(fontsize=8, ncol=5)
    fig.tight_layout(); fig.savefig(f"{out}/D_drift_forcing.png", dpi=130); plt.close(fig)

    # ---- printed summary ---------------------------------------------------------
    print("=" * 78)
    print("A  Volume-weighted error split (profiles with full-column coverage)")
    for lab in labels:
        sp = summary[lab]["split"]
        print(f"  {lab}:")
        for _, r in sp.iterrows():
            print(f"    {str(r['group']):10s} n={int(r['n']):4d}  RMSE_V={r['rmse_v']:.2f}  "
                  f"mean heat-content bias={r['mean_bias_hc']:+.2f}  heat-content share={100 * r['share_hc']:.0f}%")
    print("\nB  Bias / RMSE by layer and season (all periods)")
    for lab in labels:
        P = pts_all[lab]
        t = P.groupby(["layer", "season"], observed=True)["resid"].agg(
            bias="mean", rmse=lambda r: np.sqrt(np.mean(r ** 2))).unstack()
        print(f"  {lab}  bias:\n" + t["bias"][["DJF", "MAM", "JJA", "SON"]].round(2).to_string().replace("\n", "\n    "))
        print(f"  {lab}  rmse:\n" + t["rmse"][["DJF", "MAM", "JJA", "SON"]].round(2).to_string().replace("\n", "\n    "))
    print("\nC  Stratification")
    for lab in labels:
        t, h, st = tims[lab], hyps[lab], sts[lab]
        oe, ee = t["onset_err_days"].dropna(), t["end_err_days"].dropna()
        print(f"  {lab}: onset error {oe.mean():+.1f} d (|.| {oe.abs().mean():.1f} d, inside interval "
              f"{(oe == 0).mean() * 100:.0f}% of {len(oe)} yrs); end error {ee.mean():+.1f} d "
              f"(|.| {ee.abs().mean():.1f} d, inside {(ee == 0).mean() * 100:.0f}% of {len(ee)} yrs)")
        if len(h) > 3:
            c = np.corrcoef(h["hypo_bias_onset"], h["hypo_bias_summer"])[0, 1]
            print(f"       hypolimnion bias at onset {h['hypo_bias_onset'].mean():+.2f}, Jun-Sep "
                  f"{h['hypo_bias_summer'].mean():+.2f} degC; corr(onset, summer) = {c:.2f} over {len(h)} yrs")
        summer = st[st["datetime"].dt.month.isin([6, 7, 8, 9])]
        print(f"       Jun-Sep: surface-bottom dT bias {np.nanmean(summer['dT_mod'] - summer['dT_obs']):+.2f} degC, "
              f"thermocline depth bias {np.nanmean(summer['zth_mod'] - summer['zth_obs']):+.2f} m, "
              f"Schmidt ratio model/obs {np.nanmedian(summer['schmidt_mod'] / summer['schmidt_obs']):.2f}")
        win = st[st["datetime"].dt.month.isin([1, 2])]
        if len(win):
            inv_o, inv_m = (win["dT_obs"] < -0.5), (win["dT_mod"] < -0.5)
            print(f"       Jan-Feb inverse stratification (top colder by >0.5): obs {inv_o.mean() * 100:.0f}%, "
                  f"model {inv_m.mean() * 100:.0f}% of {len(win)} profiles; agree {(inv_o == inv_m).mean() * 100:.0f}%")
    print("\nD  Drift (degC/decade +- SE) and best single step change")
    for _, r in drift.iterrows():
        mark = "  <-- " if (abs(r["trend_per_decade"]) > 2 * r["se"]) or r["step_t"] > 4 else ""
        print(f"  {r['run']:8s} {r['series']:18s} trend {r['trend_per_decade']:+.3f} +- {r['se']:.3f}   "
              f"step {r['step_year']} (t={r['step_t']:.1f}, {r['step_size']:+.3f}){mark}")
    print("\nE  Strongest weather links (|Spearman| >= 0.3)")
    strong = weather[weather["spearman"].abs() >= 0.3].sort_values("spearman", key=np.abs, ascending=False)
    print(strong.head(15).to_string(index=False) if len(strong) else "  none")
    print("\nF  Observation checks")
    for lab in labels:
        f = flags[flags["run"] == lab] if len(flags) else flags
        print(f"  {lab}: {len(f)} outlying profiles (robust z>3): "
              + ", ".join(f"{d:%Y-%m-%d}" for d in f["datetime"].head(12)) + (" ..." if len(f) > 12 else ""))
    print("  model time of day vs date-only profiles (bias epi JJA):",
          ", ".join(f"{r.run} {r.model_time} {r.bias:+.2f}" for r in tod[(tod.layer == 'epi') & (tod.season == 'JJA')].itertuples()))

    print("\nDecision rules")
    for lab in labels:
        sp = summary[lab]["split"].set_index("group")
        share = sp.loc["all", "share_hc"]
        h = hyps[lab]
        c = np.corrcoef(h["hypo_bias_onset"], h["hypo_bias_summer"])[0, 1] if len(h) > 3 else np.nan
        verdicts = []
        verdicts.append(("heat budget / forcing dominates (b^2 share > 60%)" if share > 0.6 else
                         "vertical distribution dominates (b^2 share < 40%)" if share < 0.4 else
                         "mixed heat-content and distribution error") + f" [{100 * share:.0f}%]")
        if np.isfinite(c) and c > 0.6:
            verdicts.append(f"summer hypolimnion error largely set at onset (corr {c:.2f}) -> spring mixing / ice-off")
        dr = drift[(drift["run"] == lab)]
        if ((dr["trend_per_decade"].abs() > 2 * dr["se"]) | (dr["step_t"] > 4)).any():
            verdicts.append("significant drift or step change in residuals -> check forcing record")
        if len(flags) and (flags["run"] == lab).sum() > 0.05 * len(pts[lab]):
            verdicts.append("more than 5% of profiles are outliers -> QC the observations")
        print(f"  {lab}: " + "; ".join(verdicts))
    print(f"\nTables and figures written to {out}")


if __name__ == "__main__":
    main()
