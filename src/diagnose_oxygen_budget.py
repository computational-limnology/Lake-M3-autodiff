"""
Oxygen budget diagnostic: what controls modelled O2 in each layer, month by
month, compared with the observations -- in particular whether the spring
surface O2 deficit can be closed by primary production (NPP) or not.

Runs the JAX model (full_step, current model_params.csv) continuously from
run_config's start_time and adds up, per day and layer, the O2 change of every
process (full_step diagnostics o2_* in g O2, which sum exactly to the O2
change of each step):

  atm   gas exchange with the atmosphere (surface layer only)
  sod   sediment oxygen demand
  mix   wind mixing (mixing_step)
  diff  turbulent diffusion
  npp   gross O2 production of NPP (32 g O2 per mol C)
  resp  respiration = net reactions (o2_bio) - npp

Layers (volume-weighted): epilimnion 0-`--epi` m, metalimnion to `--meta` m,
hypolimnion below. Rates are in mg/L/day (= g/m3/day, layer mass / layer
volume). Observations (L0001-WQ do, manual profiles, interpolated to the model
grid as in calibrate_M3_jax.py) are averaged over the same layer volume on the
observation dates, and the model is sampled on the same dates and grid cells.

Spring check (--spring-months, default 4,5): the model's surface layer relaxes
towards saturation at the gas-exchange rate k = atm / (O2sat - O2model)
[1/day]. Holding the layer higher by the observed bias (obs - model) against
that outgassing needs an extra net source of about k * bias [mg/L/day]; this
is compared with the modelled gross NPP and with the NPP change a factor-2
increase of npp_pmax could at most provide.

Usage (project root):
  python src/diagnose_oxygen_budget.py Ravn
  python src/diagnose_oxygen_budget.py Ravn --end 1994-12-31          # shorter record
Outputs in <lake>/diagnostics/: o2_budget_daily.csv, o2_budget_monthly.csv,
o2_budget.png. A full 1989-2024 run takes about as long as one continuous
forward run.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import windows as W  # noqa: E402
from jax_lakeModel_functions import full_step, do_sat_calc  # noqa: E402

TERMS = ("atm", "sod", "mix", "diff", "npp", "resp")
LAYERS = ("epi", "meta", "hypo")


def layer_masks(depth, epi, meta):
    depth = np.asarray(depth)
    return {"epi": depth < epi, "meta": (depth >= epi) & (depth < meta), "hypo": depth >= meta}


def run_budget(S, masks, days_per_call=365):
    """Daily per-layer sums of the O2 terms [g], end-of-day layer O2 mass [g],
    daily mean surface O2 saturation [mg/L] and surface temperature."""
    geom, params = S["geom"], S["params"]
    M = jnp.asarray(np.stack([masks[l] for l in LAYERS]).astype(float))   # (3, nx)
    alt = geom["altitude"]

    @jax.jit
    def run_days(s, f_days):
        def day(s, fd):
            def step(s, f):
                ns, d = full_step(s, f, geom, params)
                t = jnp.stack([d["o2_atm"], d["o2_sod"], d["o2_mix"], d["o2_diff"], d["o2_npp"],
                               d["o2_bio"] - d["o2_npp"]])                     # (6, nx)
                sat = do_sat_calc(ns["u"][0], baro=f["Pa"], altitude=alt)
                return ns, (t @ M.T, sat, ns["u"][0])                         # (6, 3)
            s, (t, sat, ts) = lax.scan(step, s, fd)
            return s, (t.sum(0), sat.mean(), ts.mean(), M @ s["o2"], s["o2"])
        return lax.scan(day, s, f_days)

    F = S["forcing"]
    n_days = S["n_steps"] // 24
    s = S["s0"]
    mass0 = np.asarray(M @ s["o2"])
    out = {k: [] for k in ("terms", "sat", "tsurf", "mass", "o2")}
    d = 0
    while d < n_days:
        nd = min(days_per_call, n_days - d)
        f = {k: jnp.asarray(v[24 * d:24 * (d + nd)].reshape((nd, 24) + v.shape[1:])) for k, v in F.items()}
        s, (t, sat, ts, mass, o2) = run_days(s, f)
        for k, v in zip(out, (t, sat, ts, mass, o2)):
            out[k].append(np.asarray(v))
        d += nd
        print(f"  simulated {d}/{n_days} days", flush=True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["mass0"] = mass0
    return res


def daily_table(S, masks, res):
    vol = np.asarray(S["volume"])
    Vl = np.array([vol[masks[l]].sum() for l in LAYERS])
    days = pd.date_range(S["start"].normalize(), periods=len(res["sat"]), freq="D")
    rows = []
    for j, l in enumerate(LAYERS):
        df = pd.DataFrame({"date": days, "layer": l})
        for i, t in enumerate(TERMS):
            df[t] = res["terms"][:, i, j] / Vl[j]                    # mg/L/day
        prev = np.concatenate([[res["mass0"][j]], res["mass"][:-1, j]])
        df["dO2_dt"] = (res["mass"][:, j] - prev) / Vl[j]
        df["closure"] = df["dO2_dt"] - df[list(TERMS)].sum(axis=1)
        df["o2_model"] = res["mass"][:, j] / Vl[j]
        if l == "epi":
            df["o2_sat"] = res["sat"]
            df["t_surface"] = res["tsurf"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True), Vl


def observed_layer_means(S, masks, res, min_cover=0.5):
    """Volume-weighted observed and modelled O2 [mg/L] per layer on the
    observation dates (same grid cells for both)."""
    o = S["obs"]["o2"]
    if o is None:
        return pd.DataFrame()
    idx = np.asarray(o["step_idx"]); val = np.asarray(o["values"]); m = np.asarray(o["mask"])
    vol = np.asarray(S["volume"])
    day = idx // 24
    ok = day < len(res["o2"])
    rows = []
    for k in np.nonzero(ok)[0]:
        mod = res["o2"][day[k]] / vol
        date = (S["start"] + pd.Timedelta(hours=int(idx[k]))).normalize()
        for l in LAYERS:
            sel = masks[l] & m[k]
            if vol[sel].sum() < min_cover * vol[masks[l]].sum():
                continue
            w = vol[sel]
            rows.append(dict(date=date, layer=l, o2_obs=float(np.sum(val[k][sel] * w) / w.sum()),
                             o2_model_at_obs=float(np.sum(mod[sel] * w) / w.sum()),
                             o2_obs_surface=float(val[k][0]) if m[k][0] else np.nan))
    return pd.DataFrame(rows)


def monthly_table(daily, obs):
    daily = daily.assign(month=daily["date"].dt.month)
    cols = list(TERMS) + ["o2_model"]
    mon = daily.groupby(["layer", "month"])[cols + ["o2_sat"]].mean()
    if len(obs):
        ob = obs.assign(month=obs["date"].dt.month).groupby(["layer", "month"])[
            ["o2_obs", "o2_model_at_obs"]].mean()
        ob["bias_model_minus_obs"] = ob["o2_model_at_obs"] - ob["o2_obs"]
        mon = mon.join(ob)
    return mon.reindex(pd.MultiIndex.from_product([LAYERS, range(1, 13)], names=["layer", "month"]))


def spring_check(daily, obs, months, npp_factor=2.0):
    d = daily[(daily["layer"] == "epi") & daily["date"].dt.month.isin(months)]
    o = obs[(obs["layer"] == "epi") & obs["date"].dt.month.isin(months)] if len(obs) else obs
    deficit = d["o2_sat"] - d["o2_model"]
    use = deficit.abs() > 0.05
    k = float(d.loc[use, "atm"].sum() / deficit[use].sum()) if use.any() else np.nan   # 1/day
    bias = float((o["o2_obs"] - o["o2_model_at_obs"]).mean()) if len(o) else np.nan
    sat_model = float((d["o2_model"] / d["o2_sat"]).mean() * 100)
    sat_obs = np.nan
    if len(o):
        sat_by_day = d.set_index("date")["o2_sat"]
        s = o.set_index("date")["o2_obs_surface"] / sat_by_day.reindex(o["date"]).to_numpy()
        sat_obs = float(s.mean() * 100)
    need = k * bias if np.isfinite(k) and np.isfinite(bias) else np.nan
    npp = float(d["npp"].mean()); resp = float(d["resp"].mean())
    return dict(months=months, k_gas_per_day=k, bias_obs_minus_model=bias, sat_model_pct=sat_model,
                sat_obs_pct=sat_obs, extra_source_needed=need, npp_gross=npp, resp=resp,
                atm=float(d["atm"].mean()), mix_diff=float((d["mix"] + d["diff"]).mean()),
                npp_factor=npp_factor,
                npp_ratio_needed=(npp + need) / npp if npp > 0 and np.isfinite(need) else np.nan)


def plot_monthly(mon, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(atm="tab:blue", sod="tab:brown", mix="tab:purple", diff="tab:gray",
                  npp="tab:green", resp="tab:red")
    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(11, 3.4 * len(LAYERS)), sharex=True)
    x = np.arange(1, 13)
    for ax, l in zip(axes, LAYERS):
        m = mon.loc[l]
        pos = np.zeros(12); neg = np.zeros(12)
        for t in TERMS:
            v = m[t].to_numpy(float)
            v = np.nan_to_num(v)
            ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=colors[t], label=t, width=0.7)
            ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=colors[t], width=0.7)
            pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
        ax.plot(x, m[list(TERMS)].sum(axis=1), "k.-", lw=1, label="net")
        span = max(pos.max(), -neg.min(), 1e-6)
        ax.set_ylim(neg.min() - 0.08 * span, pos.max() + 0.08 * span)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"{l}\nO2 rate (mg/L/day)")
        ax2 = ax.twinx()
        ax2.plot(x, m["o2_model"], color="tab:orange", lw=2, label="model O2")
        if "o2_obs" in m:
            ax2.plot(x, m["o2_obs"], "o", color="k", label="observed O2")
        if l == "epi":
            ax2.plot(x, m["o2_sat"], "--", color="tab:cyan", label="saturation")
        ax2.set_ylabel("O2 (mg/L)")
        if l == "epi":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2
    axes[-1].set_xticks(x)
    axes[-1].set_xlabel("month (mean over all years)")
    fig.suptitle("Modelled O2 budget by layer (bars) and O2 concentration (lines)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, ncol=10, fontsize=8, loc="lower center")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--end", default=None, help="end of the simulated record (date; default run_config end_time)")
    ap.add_argument("--epi", type=float, default=5.0, help="epilimnion bottom depth [m] (default 5)")
    ap.add_argument("--meta", type=float, default=15.0, help="metalimnion bottom depth [m] (default 15)")
    ap.add_argument("--spring-months", default="4,5", help="months for the spring check (default 4,5)")
    ap.add_argument("--out-dir", default=None, help="default <lake>/diagnostics")
    a = ap.parse_args()

    t0 = time.time()
    S = W.build_setup(a.lake_dir, record_end=a.end, temp_obs="profiles", load_obs=True)
    masks = layer_masks(S["depth"], a.epi, a.meta)
    print(f"record {S['start'].date()} + {S['n_steps'] // 24} days; layers: epi < {a.epi:g} m, "
          f"meta {a.epi:g}-{a.meta:g} m, hypo > {a.meta:g} m", flush=True)
    res = run_budget(S, masks)
    daily, Vl = daily_table(S, masks, res)
    obs = observed_layer_means(S, masks, res)
    mon = monthly_table(daily, obs)

    clos = float(np.max(np.abs(daily["closure"]))) / max(float(np.max(np.abs(daily["dO2_dt"]))), 1e-300)
    print(f"budget closure: max |dO2/dt - sum(terms)| / max|dO2/dt| = {clos:.1e}")

    pd.set_option("display.width", 220)
    show = list(TERMS) + ["o2_model"] + [c for c in ("o2_obs", "bias_model_minus_obs", "o2_sat") if c in mon]
    for l in LAYERS:
        print(f"\n{l}: monthly mean O2 budget (mg/L/day) and O2 (mg/L), all years")
        print(mon.loc[l][[c for c in show if c != "o2_sat" or l == "epi"]].round(3).to_string())

    months = [int(m) for m in a.spring_months.split(",")]
    sc = spring_check(daily, obs, months)
    print(f"\nSpring check, epilimnion, months {months}:")
    print(f"  O2 saturation: model {sc['sat_model_pct']:.0f} %, observed {sc['sat_obs_pct']:.0f} % "
          f"(observed surface O2 vs modelled saturation)")
    print(f"  bias obs - model: {sc['bias_obs_minus_model']:+.2f} mg/L")
    print(f"  modelled terms (mg/L/day): NPP gross {sc['npp_gross']:+.3f}, respiration {sc['resp']:+.3f}, "
          f"gas exchange {sc['atm']:+.3f}, mixing+diffusion {sc['mix_diff']:+.3f}")
    print(f"  gas-exchange relaxation rate k = {sc['k_gas_per_day']:.3f} /day "
          f"(time scale {1 / sc['k_gas_per_day']:.1f} days)" if sc["k_gas_per_day"] > 0 else
          "  gas-exchange rate could not be estimated")
    print(f"  extra net O2 source needed to hold the observed level: ~{sc['extra_source_needed']:.3f} mg/L/day "
          f"= NPP x {sc['npp_ratio_needed']:.1f}" if np.isfinite(sc["npp_ratio_needed"]) else "")
    if np.isfinite(sc["npp_ratio_needed"]):
        if sc["npp_ratio_needed"] > 3:
            print("  -> the missing source is several times the modelled NPP: not reachable by tuning "
                  "npp_pmax/npp_alpha within plausible ranges; check the NPP structure "
                  "(areal NPP spread over layers by area, TP limitation, light/PAR) or slower gas "
                  "exchange (kgas_factor).")
        elif sc["npp_ratio_needed"] > 1.2:
            print("  -> a moderate NPP increase (or slower gas exchange) could close the spring deficit.")
        else:
            print("  -> the spring deficit is small relative to NPP.")

    out_dir = a.out_dir or os.path.join(os.path.abspath(a.lake_dir), "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    daily.to_csv(os.path.join(out_dir, "o2_budget_daily.csv"), index=False)
    mon.to_csv(os.path.join(out_dir, "o2_budget_monthly.csv"))
    plot_monthly(mon, os.path.join(out_dir, "o2_budget.png"))
    print(f"\nsaved o2_budget_daily.csv, o2_budget_monthly.csv, o2_budget.png in {out_dir} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
