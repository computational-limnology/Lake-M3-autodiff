"""
Thermocline-deepening diagnostic (Step 1 of the mixing_step MCL plan): where
and why the summer thermocline is too deep, and which process controls it.

A. Thermocline-depth bias (model - observed) by month and by year, from a
   continuous result file (default <lake>/res_lake1_jax_full.npz), on manual
   profiles reaching --zmin m, split into training years and --test-years.
   Thermocline depth = depth of the maximum dT/dz between grid cells. Storm
   effect: for consecutive summer profiles <= 30 days apart, change of the
   depth bias vs the wind energy input in between (Spearman correlation and
   mean change for the calmest / windiest third of the intervals).

B. Process attribution: all yearly windows are run in parallel from the start
   states of a continuous reference run, once with the process model and with
   one process changed at a time (through full_step_mcl's hooks, so nothing
   else changes):
     entrainment     wind energy in mixing_step x 1.2 and / 1.2  (mix_override ke_scale)
     surface diff.   Kz in the surface mixed layer x 1.5 and / 1.5  (kz_override, weight = the
                     closure's own surface-layer weight)
     interior diff.  Kz below the surface layer x 1.5 and / 1.5
     light ext.      background light extinction light_water x 1.3 and / 1.3 (sunlight
                     absorbed nearer the surface / deeper; tests whether the depth bias
                     is heating- rather than wind-driven)
   Splits: training / test years, and the calmest / windiest third of
   summers (May-Aug mean U^3 of the forcing).
   Per process: elasticity of the summer thermocline depth, maximum dT/dz and
   metalimnion bias per unit log-factor, and the factor that would remove the
   mean depth bias if the response were linear (indicative only).

C. Wind-mixing energy log of the process-model windows (mixing_step
   diagnostics), by month: wind energy input [J m-2 d-1], layers fully merged
   per hour, share of hours with a full merge, mean mixed-layer depth after
   mixing, and hours with deepening > 1 m.

Usage (project root; parts B/C use all CPU cores by default):
  python src/diagnose_mixing.py Ravn
  python src/diagnose_mixing.py Ravn --run Ravn/res_lake1_jax_full.npz --test-years 2016,2018,2020,2022,2024
  python src/diagnose_mixing.py Ravn --record-end 1994-12-31 --skip-a      # quick test of B/C
Outputs: <lake>/diagnostics/mixing_A_depth_bias.csv, mixing_B_attribution.csv,
mixing_C_energy.csv.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _early_device_flags(argv):
    n = int(argv[argv.index("--devices") + 1]) if "--devices" in argv else (os.cpu_count() or 1)
    flags = os.environ.get("XLA_FLAGS", "")
    if n > 1 and "xla_force_host_platform_device_count" not in flags:
        flags += f" --xla_force_host_platform_device_count={n}"
    if "--multi-thread" not in argv and "xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = flags.strip()
    if n > 1 and "--xla-warnings" not in argv:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import xla_log_filter
        xla_log_filter.install()


if __name__ == "__main__":
    _early_device_flags(sys.argv)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import windows as W  # noqa: E402
from jax_lakeModel_functions import full_step_mcl, calc_dens, surface_layer_weight  # noqa: E402

PERTURB = (("entrainment", 0, 1.2), ("surface diffusion", 1, 1.5), ("interior diffusion", 2, 1.5),
           ("light extinction", 3, 1.3))
MIX_KEYS = ("mix_KE0", "mix_KE_used", "mix_n_full", "mix_kp_partial", "mix_mld")


def years_split(years, test_spec):
    test = [int(y) for y in test_spec.split(",") if y.strip()] if test_spec else []
    return {"train": [y for y in years if y not in test], "test": [y for y in years if y in test]}


def wind_split(S, years):
    """Calmest / windiest third of the given years by May-Aug mean U^3."""
    t = S["start"] + pd.to_timedelta(np.arange(S["n_steps"]), unit="h")
    u3 = pd.Series(np.asarray(S["forcing"]["Uw"]) ** 3, index=t)
    u3 = u3[u3.index.month.isin([5, 6, 7, 8])].groupby(u3[u3.index.month.isin([5, 6, 7, 8])].index.year).mean()
    u3 = u3.reindex([y for y in years if y in u3.index])
    if len(u3) < 3:
        return {}
    q = u3.quantile([1 / 3, 2 / 3]).to_numpy()
    return {"calm third": u3.index[u3 <= q[0]].tolist(), "windy third": u3.index[u3 >= q[1]].tolist()}


def thermocline(prof, depth):
    """Depth of maximum dT/dz and the maximum, per profile (rows)."""
    dz = np.diff(depth[:prof.shape[1]])
    g = -np.diff(prof, axis=1) / dz[None]
    j = np.argmax(g, axis=1)
    return 0.5 * (depth[j] + depth[j + 1]), g[np.arange(len(g)), j]


def summer_profiles(obs, start, depth, zmin, months=(6, 7, 8, 9)):
    idx = np.asarray(obs["step_idx"]); val = np.asarray(obs["values"]); m = np.asarray(obs["mask"])
    t = start + pd.to_timedelta(idx, unit="h")
    k = int(np.searchsorted(depth, zmin))
    sel = np.isin(t.month, months) & m[:, :k].all(axis=1)
    return sel, idx, val, m, t, k


# ---------------------------------------------------------------------------
# A. depth bias from a continuous result file
# ---------------------------------------------------------------------------

def part_a(S, run_path, zmin, split):
    with np.load(run_path) as z:
        u = z["temp"]
        rstart = pd.Timestamp(str(z["start_time"])) if "start_time" in z.files else S["start"]
    if rstart != S["start"]:
        raise SystemExit(f"{run_path} starts {rstart}, run_config starts {S['start']}")
    depth = np.asarray(S["depth"])
    o = S["obs"]["temp"]
    sel, idx, val, m, t, k = summer_profiles(o, S["start"], depth, zmin, months=range(5, 11))
    sel &= idx < len(u)
    zo, go = thermocline(val[sel][:, :k], depth)
    zm, gm = thermocline(u[idx[sel]][:, :k], depth)
    df = pd.DataFrame(dict(date=t[sel].normalize(), year=t[sel].year, month=t[sel].month, z_obs=zo, z_mod=zm,
                           bias=zm - zo, g_obs=go, g_mod=gm))
    df["split"] = np.where(df["year"].isin(split["test"]), "test", "train")
    by_month = df.groupby(["split", "month"]).agg(n=("bias", "size"), z_obs=("z_obs", "mean"), z_mod=("z_mod", "mean"),
                                                  depth_bias=("bias", "mean"), dTdz_obs=("g_obs", "mean"),
                                                  dTdz_mod=("g_mod", "mean"))
    js = df[df["month"].isin([6, 7, 8, 9])]
    by_year = js.groupby("year").agg(n=("bias", "size"), depth_bias=("bias", "mean"), dTdz_obs=("g_obs", "mean"),
                                     dTdz_mod=("g_mod", "mean"))
    # storms: wind energy between consecutive profiles in the same summer
    pw = 1.225 * float(S["params"]["Cd"]) * (np.asarray(S["forcing"]["Uw"]) * float(S["params"]["wind_factor"])) ** 3
    cpw = np.concatenate([[0.0], np.cumsum(pw)]) * 3600.0          # J m-2 since start
    js = js.sort_values("date").reset_index(drop=True)
    rows = []
    for a, b in zip(js.index[:-1], js.index[1:]):
        r0, r1 = js.loc[a], js.loc[b]
        days = (r1["date"] - r0["date"]).days
        if r0["year"] != r1["year"] or not (0 < days <= 30):
            continue
        h0 = int((r0["date"] - S["start"]) / pd.Timedelta(hours=1)); h1 = int((r1["date"] - S["start"]) / pd.Timedelta(hours=1))
        rows.append(dict(year=r1["year"], days=days, wind_J_m2_per_day=(cpw[min(h1, len(cpw) - 1)] - cpw[min(h0, len(cpw) - 1)]) / days,
                         d_bias=r1["bias"] - r0["bias"], d_zobs=r1["z_obs"] - r0["z_obs"], d_zmod=r1["z_mod"] - r0["z_mod"]))
    st = pd.DataFrame(rows)
    storm = {}
    if len(st) > 5:
        rho = st[["wind_J_m2_per_day", "d_bias"]].rank().corr().iloc[0, 1]
        q = st["wind_J_m2_per_day"].quantile([1 / 3, 2 / 3]).to_numpy()
        storm = dict(n_intervals=len(st), spearman_wind_vs_bias_change=float(rho),
                     calm_third_mean_bias_change=float(st.loc[st["wind_J_m2_per_day"] <= q[0], "d_bias"].mean()),
                     windy_third_mean_bias_change=float(st.loc[st["wind_J_m2_per_day"] >= q[1], "d_bias"].mean()),
                     windy_third_obs_deepening=float(st.loc[st["wind_J_m2_per_day"] >= q[1], "d_zobs"].mean()),
                     windy_third_model_deepening=float(st.loc[st["wind_J_m2_per_day"] >= q[1], "d_zmod"].mean()))
    return by_month, by_year, storm


# ---------------------------------------------------------------------------
# B/C. windowed runs with one process changed at a time
# ---------------------------------------------------------------------------

def make_sim(S):
    geom, params = S["geom"], S["params"]

    def sim(fac, state0, forcing):
        """fac = [ke_scale, surface-layer Kz factor, interior Kz factor, light_water factor] (traced)."""
        lsb, lin = jnp.log(fac[1]), jnp.log(fac[2])
        p = dict(params)
        p["light_water"] = params["light_water"] * fac[3]

        def kz_override(kz_process, u, ice, dens_u, forcing_t, ri):
            w = surface_layer_weight(dens_u, params["sbl_drho"], params["sbl_width"])
            return kz_process * jnp.exp(lsb * w + lin * (1.0 - w))

        def mix_override(state, forcing_t, u, ice):
            return dict(ke_scale=fac[0])

        def step(s, f):
            ns, d = full_step_mcl(s, f, geom, p, kz_override=kz_override, mix_override=mix_override)
            return ns, (ns["u"], jnp.stack([d[k] for k in MIX_KEYS]))

        _, (u, mix) = lax.scan(step, state0, forcing)
        return u, mix
    return sim


def window_metrics(S, win, U, zmin):
    """Per window: summer profiles reaching zmin after the spin-up -> model and
    observed thermocline depth / max dT/dz, metalimnion bias, RMSE."""
    depth = np.asarray(S["depth"])
    o = W.window_obs(S["obs"]["temp"], win)
    k = int(np.searchsorted(depth, zmin))
    meta = (depth >= 5) & (depth < 15)
    rows = []
    for b, r in enumerate(win.itertuples()):
        v = o["valid"][b]
        idx = o["step_idx"][b][v]; val = o["values"][b][v]; m = o["mask"][b][v]
        t = r.start + pd.to_timedelta(idx, unit="h")
        sel = np.isin(t.month, [6, 7, 8, 9]) & m[:, :k].all(axis=1)
        if not sel.any():
            continue
        mod = U[b][idx[sel]]
        zo, go = thermocline(val[sel][:, :k], depth); zm, gm = thermocline(mod[:, :k], depth)
        e = mod - val[sel]
        mm = m[sel]
        rows.append(dict(year=r.year, n=int(sel.sum()), z_obs=zo.mean(), z_mod=zm.mean(), g_obs=go.mean(), g_mod=gm.mean(),
                         meta_bias=float(e[:, meta][mm[:, meta]].mean()), rmse=float(np.sqrt(np.mean(e[mm] ** 2)))))
    return pd.DataFrame(rows)


def parts_bc(S, zmin, split, window_start, window_steps, spinup_days):
    win = W.select_windows(S, window_start, window_steps, spinup_days, require_obs="temp")
    D = jax.device_count()
    print(f"\nB/C: {len(win)} windows {win['year'].min()}-{win['year'].max()} on {D} device(s); "
          "reference run for the start states ...", flush=True)
    t = time.time()
    daily = W.reference_states(S, None)
    st = W.window_start_states(daily, win)
    F = W.window_forcing(S["forcing"], win, window_steps)
    print(f"  reference run {time.time() - t:.0f}s", flush=True)
    sim = make_sim(S)
    run = jax.jit(lambda fac, st, F: W.run_batched(sim, fac, (st, F), n_devices=D))

    configs = [("process", jnp.ones(4))]
    for name, j, f in PERTURB:
        for lab, x in (("+", f), ("-", 1.0 / f)):
            fac = np.ones(4); fac[j] = x
            configs.append((f"{name} {lab}", jnp.asarray(fac)))
    res, mix_base = {}, None
    for lab, fac in configs:
        t = time.time()
        U, mix = run(fac, st, F)
        U = np.asarray(U)
        res[lab] = window_metrics(S, win, U, zmin)
        if lab == "process":
            mix_base = np.asarray(mix)
        print(f"  {lab:24s} summer thermocline depth {res[lab]['z_mod'].mean():6.2f} m "
              f"(obs {res[lab]['z_obs'].mean():.2f})  ({time.time() - t:.0f}s)", flush=True)

    base = res["process"]
    rows = []
    for name, j, f in PERTURB:
        p, m = res[f"{name} +"], res[f"{name} -"]
        dl = 2 * np.log(f)
        for split_name, ys in split.items():
            sb = base["year"].isin(ys)
            if not sb.any():
                continue
            pick = lambda d, c: d.loc[d["year"].isin(ys), c].mean()
            e_z = (pick(p, "z_mod") - pick(m, "z_mod")) / dl
            bias = pick(base, "z_mod") - pick(base, "z_obs")
            rows.append(dict(process=name, split=split_name, depth_bias_m=bias,
                             dz_per_dlogf=e_z, d_dTdz_per_dlogf=(pick(p, "g_mod") - pick(m, "g_mod")) / dl,
                             d_meta_bias_per_dlogf=(pick(p, "meta_bias") - pick(m, "meta_bias")) / dl,
                             d_rmse_per_dlogf=(pick(p, "rmse") - pick(m, "rmse")) / dl,
                             factor_to_remove_depth_bias=float(np.exp(-bias / e_z)) if abs(e_z) > 1e-6 else np.nan))
    attr = pd.DataFrame(rows).set_index(["process", "split"])

    # C. energy log (process windows, after the spin-up)
    area0 = float(np.asarray(S["geom"]["area"])[0])
    recs = []
    for b, r in enumerate(win.itertuples()):
        tt = r.start + pd.to_timedelta(np.arange(window_steps), unit="h")
        mx = mix_base[b][24 * spinup_days:].T          # (len(MIX_KEYS), hours after spin-up)
        tt = tt[24 * spinup_days:]
        mld = mx[4]
        recs.append(pd.DataFrame(dict(month=tt.month, KE0=mx[0], used=mx[1], n_full=mx[2], kp=mx[3], mld=mld,
                                      deepen=np.concatenate([[0.0], np.diff(mld)]))))
    e = pd.concat(recs)
    energy = e.groupby("month").agg(
        wind_energy_J_m2_day=("KE0", lambda x: 24 * x.mean() / area0),
        used_share=("used", "sum"), n_full_per_hour=("n_full", "mean"),
        share_hours_full_merge=("n_full", lambda x: (x > 0).mean()), mean_partial=("kp", "mean"),
        mld_m=("mld", "mean"), hours_deepening_gt1m=("deepen", lambda x: (x > 1.0).sum() / len(win)))
    tot = e.groupby("month")["KE0"].sum()
    energy["used_share"] = energy["used_share"] / tot.where(tot > 0, np.nan)
    return attr, energy, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--run", default=None, help="continuous result for part A (default <lake>/res_lake1_jax_full.npz)")
    ap.add_argument("--test-years", default="2016,2018,2020,2022,2024")
    ap.add_argument("--zmin", type=float, default=25.0, help="profiles must reach this depth (default 25 m)")
    ap.add_argument("--record-end", default=None)
    ap.add_argument("--window-start", default="02-01")
    ap.add_argument("--window-steps", type=int, default=8016)
    ap.add_argument("--spinup-days", type=int, default=28)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--multi-thread", action="store_true")
    ap.add_argument("--xla-warnings", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    lake = os.path.abspath(a.lake_dir)
    out = a.out_dir or os.path.join(lake, "diagnostics")
    os.makedirs(out, exist_ok=True)
    S = W.build_setup(lake, record_end=a.record_end, temp_obs="profiles")
    years = sorted(set((S["start"] + pd.to_timedelta(np.asarray(S["obs"]["temp"]["step_idx"]), unit="h")).year))
    split = years_split(years, a.test_years)
    pd.set_option("display.width", 200)

    if not a.skip_a:
        run_path = a.run or os.path.join(lake, "res_lake1_jax_full.npz")
        bm, by, storm = part_a(S, run_path, a.zmin, split)
        print(f"A. thermocline depth (max dT/dz) from {os.path.relpath(run_path)}; profiles reaching {a.zmin:g} m, May-Oct")
        print(bm.round(2).to_string())
        print("\nby year (Jun-Sep):")
        print(by.round(2).T.to_string())
        if storm:
            print("\nstorm effect (consecutive summer profiles <= 30 d apart):")
            for k, v in storm.items():
                print(f"  {k:32s} {v:.3g}")
        pd.concat({"by_month": bm.reset_index(), "by_year": by.reset_index()}, names=["table"]).to_csv(
            os.path.join(out, "mixing_A_depth_bias.csv"))

    if not a.skip_bc:
        split_b = dict(split, **wind_split(S, years))
        attr, energy, _ = parts_bc(S, a.zmin, split_b, a.window_start, a.window_steps, a.spinup_days)
        print("\nB. process attribution (summer, windowed runs from the same start states):")
        print("   d.../dlogf = change per unit log-factor (x e); factor_to_remove = linear estimate, indicative only")
        print(attr.round(3).to_string())
        print("\nC. wind-mixing energy (process model, windows after spin-up), by month:")
        print(energy.round(3).to_string())
        attr.to_csv(os.path.join(out, "mixing_B_attribution.csv"))
        energy.to_csv(os.path.join(out, "mixing_C_energy.csv"))
    print(f"\nsaved in {out}")


if __name__ == "__main__":
    main()
