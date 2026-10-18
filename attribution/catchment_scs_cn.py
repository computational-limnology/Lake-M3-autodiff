"""
Simple daily catchment model (SCS Curve Number runoff + soil/groundwater
buckets) written in JAX, used to build *catchment-driven* inflow forcing for
the Lake-M3 JAX lake model:

  * `oc_load_file.csv`      -- daily discharge [m3/day] and flow-weighted
                               organic-carbon concentration `oc` [g/m3 = mg/L]
  * `L0001-WQ-inflow.csv`   -- daily in-lake (epilimnion) TP [ug/L], from a
                               simple lake TP mass balance driven by the
                               catchment TP load

Both are written to this `attribution/` folder (the files in the lake folder
are not touched). A counterfactual pair without the spring agricultural TP
pulse is written alongside (`*_nopulse.csv`), so `attribute_pocl.py` can
attribute changes in the lake's labile POC to the catchment inflow.

Usage (from the project root):
    python attribution/catchment_scs_cn.py [--lake-dir Ravn] [--area-km2 45]
        [--spring-amplitude 5] [--out-dir attribution]

---------------------------------------------------------------------------
Model (all fluxes in mm/day over the area of each land-use class, then
converted to m3/day with the class area):

1. Precipitation P: daily sum of L0001-MET.csv's hourly
   "Precipitation_millimeterPerDay" column. The column holds hourly amounts
   despite its name: its long-term mean is 0.082 per hour = ~720 mm/yr,
   typical for Denmark, whereas reading it as mm/day would give ~30 mm/yr.

2. Surface runoff per land use, SCS-CN (USDA-SCS 1972; AMC II):
       S  = 25400 / CN - 254           [mm]  potential maximum retention
       Ia = 0.2 S                      [mm]  initial abstraction
       Q  = (P - Ia)^2 / (P - Ia + S)  for P > Ia, else 0
   written with relu(P - Ia) so it is differentiable in CN.

3. Soil bucket per land use (capacity SOIL_MAX): infiltration P - Q enters,
   actual ET = PET * min(1, soil / (0.7 SOIL_MAX)) leaves, and storage above
   capacity drains. A land-use-specific fraction of that drainage leaves as
   fast drain flow (tile drains: high on Danish arable land), the rest
   recharges a shared groundwater store.
   PET is a deliberately simple temperature index,
   PET = clip(0.12 T_air + 0.3, 0, 4) mm/day (~500 mm/yr for Ravn).
   No snow routine (mild Danish winters; a known simplification).

4. Routing: surface runoff + drain flow enter a fast linear reservoir
   (K_QUICK days); recharge enters a slow groundwater reservoir (K_GW days).
   Discharge = outflow of both, times area.

5. Loads: each land use has a TP and OC concentration for its fast flow
   (surface runoff + drains); groundwater/baseflow has its own (low)
   concentrations. Agricultural fast-flow TP is multiplied by a spring
   factor 1 + A exp(-0.5 ((doy - SPRING_DOY) / SPRING_WIDTH)^2) representing
   manure/fertiliser application on wet soils in late winter/spring (A =
   --spring-amplitude; A = 0 is the counterfactual). Solute mass is routed
   through the same reservoirs as the water, so concentrations stay
   consistent.

6. In-lake TP (the lake model's TP forcing is an in-lake concentration, used
   directly in its NPP limitation term, not an inflow load): a one-box
   Vollenweider-type balance
       V dC/dt = L_TP - Q C - sigma(doy) V C
   with a seasonal net loss rate sigma (settling/uptake, highest in summer)
   whose magnitude sigma0 is set so the long-term mean matches the
   observed epilimnion TP in the original L0001-WQ-inflow.csv.

Default catchment: 61 km2 (~36x Ravn's 1.7 km2 surface; Danish lake
catchments are typically 10-50x lake area). With this water balance
(~240 mm/yr runoff) 61 km2 reproduces the mean discharge of Ravn's original
oc_load_file.csv.
Land use 55% arable / 27% forest / 13% grassland & nature / 5% urban,
roughly Danish (about 60% of Denmark is farmed). All values are assumptions
for a synthetic scenario, not a calibrated model of Ravn's catchment.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# --- land use: fraction of catchment, curve number (HSG B, AMC II), fraction
# of soil drainage leaving as fast drain flow, fast-flow TP [g/m3] and OC [g/m3]
LAND_USE = {
    #               frac   CN   f_drain  TP_fast  OC_fast
    "arable":      (0.55, 78.0, 0.50,    0.25,    9.0),
    "forest":      (0.27, 55.0, 0.05,    0.03,    20.0),
    "grassland":   (0.13, 61.0, 0.10,    0.08,    14.0),
    "urban":       (0.05, 85.0, 0.30,    0.30,    7.0),
}
SOIL_MAX = 120.0     # mm, plant-available soil storage
K_QUICK = 2.0        # days, fast reservoir time constant
K_GW = 90.0          # days, groundwater reservoir time constant
TP_BASE = 0.04       # g/m3, groundwater TP
OC_BASE = 8.0        # g/m3, groundwater OC
SPRING_DOY = 90.0    # day of year of the spring TP application peak (~31 Mar)
SPRING_WIDTH = 20.0  # days (1 sd)
DEFAULT_SPRING_AMPLITUDE = 5.0
DEFAULT_AREA_KM2 = 61.0
SIGMA_SEASONAL_AMP = 0.8   # sigma(doy) = sigma0 * (1 + amp * cos(2 pi (doy - 200)/365))


def scs_runoff(P, CN):
    """SCS-CN direct runoff [mm] for daily precipitation P [mm]."""
    S = 25400.0 / CN - 254.0
    pe = jax.nn.relu(P - 0.2 * S)
    return pe ** 2 / (pe + S + 1e-9)


def simulate_catchment(P, T, doy, area_m2, spring_amplitude, lu=LAND_USE):
    """Run the catchment model over daily series. Returns a dict of daily
    series: Q [m3/day], TP_load [g/day], OC_load [g/day], plus diagnostics.
    Pure JAX (lax.scan), differentiable in its parameters."""
    names = list(lu)
    frac = jnp.array([lu[n][0] for n in names])
    CN = jnp.array([lu[n][1] for n in names])
    f_drain = jnp.array([lu[n][2] for n in names])
    tp_fast = jnp.array([lu[n][3] for n in names])
    oc_fast = jnp.array([lu[n][4] for n in names])
    is_arable = jnp.array([1.0 if n == "arable" else 0.0 for n in names])
    a_lu = frac * area_m2  # m2 per land use

    def step(carry, x):
        soil, gw, quick, gw_tp, gw_oc, q_tp, q_oc = carry
        p, temp, d = x
        pet = jnp.clip(0.12 * temp + 0.3, 0.0, 4.0)
        q_surf = scs_runoff(p, CN)                                  # mm, per land use
        soil = soil + (p - q_surf)
        aet = pet * jnp.minimum(1.0, soil / (0.7 * SOIL_MAX))
        soil = jnp.maximum(soil - aet, 0.0)
        excess = jax.nn.relu(soil - SOIL_MAX)
        soil = soil - excess
        drain = f_drain * excess
        recharge = excess - drain
        fast_mm = q_surf + drain                                    # mm per land use
        fast_m3 = fast_mm * 1e-3 * a_lu
        spring = 1.0 + spring_amplitude * jnp.exp(-0.5 * ((d - SPRING_DOY) / SPRING_WIDTH) ** 2)
        tp_conc = tp_fast * (1.0 + is_arable * (spring - 1.0))
        # reservoirs (water m3, solute g)
        quick = quick + jnp.sum(fast_m3)
        q_tp = q_tp + jnp.sum(fast_m3 * tp_conc)
        q_oc = q_oc + jnp.sum(fast_m3 * oc_fast)
        rech_m3 = jnp.sum(recharge * 1e-3 * a_lu)
        gw = gw + rech_m3
        gw_tp = gw_tp + rech_m3 * TP_BASE
        gw_oc = gw_oc + rech_m3 * OC_BASE
        out_q, out_g = quick / K_QUICK, gw / K_GW
        out_qtp, out_gtp = q_tp / K_QUICK, gw_tp / K_GW
        out_qoc, out_goc = q_oc / K_QUICK, gw_oc / K_GW
        quick, gw = quick - out_q, gw - out_g
        q_tp, gw_tp = q_tp - out_qtp, gw_tp - out_gtp
        q_oc, gw_oc = q_oc - out_qoc, gw_oc - out_goc
        out = dict(Q=out_q + out_g, Q_fast=out_q, TP_load=out_qtp + out_gtp,
                   OC_load=out_qoc + out_goc, aet=jnp.sum(aet * frac), surf=jnp.sum(q_surf * frac))
        return (soil, gw, quick, gw_tp, gw_oc, q_tp, q_oc), out

    n_lu = len(names)
    # spin-up start: soils half full, groundwater at a store giving ~ mean baseflow
    gw0 = 0.6 * jnp.mean(P) * 1e-3 * area_m2 * 0.35 * K_GW
    carry0 = (jnp.full(n_lu, 0.5 * SOIL_MAX), gw0, 0.0, gw0 * TP_BASE, gw0 * OC_BASE, 0.0, 0.0)
    _, out = lax.scan(step, carry0, (P, T, doy))
    return out


def lake_tp_balance(load_g_day, Q_m3_day, doy, volume_m3, sigma0, C0):
    """Daily in-lake TP [g/m3]: V dC/dt = L - Q C - sigma(doy) V C (implicit Euler)."""
    def step(C, x):
        L, Q, d = x
        sigma = sigma0 * (1.0 + SIGMA_SEASONAL_AMP * jnp.cos(2 * jnp.pi * (d - 200.0) / 365.0))
        C = (C + L / volume_m3) / (1.0 + Q / volume_m3 + sigma)
        return C, C
    _, C = lax.scan(step, C0, (load_g_day, Q_m3_day, doy))
    return C


def fit_sigma0(load, Q, doy, volume, C_obs_mean, C0):
    """sigma0 [1/day] so that the simulated long-term mean TP matches the
    observed mean (bisection on log sigma0; the mean decreases monotonically
    with sigma0)."""
    lo, hi = np.log(1e-6), np.log(1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = float(jnp.mean(lake_tp_balance(load, Q, doy, volume, np.exp(mid), C0)))
        lo, hi = (mid, hi) if c > C_obs_mean else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake-dir", default=os.path.join(ROOT, "Ravn"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--area-km2", type=float, default=DEFAULT_AREA_KM2)
    ap.add_argument("--spring-amplitude", type=float, default=DEFAULT_SPRING_AMPLITUDE,
                    help="multiplier peak of the arable spring TP pulse (0 = no pulse)")
    args = ap.parse_args()

    from processBased_lakeModel_functions import get_hypsography, get_lake_config, get_run_config

    lake_dir = os.path.abspath(args.lake_dir)
    rc = get_run_config(os.path.join(lake_dir, "run_config.csv"), 1)
    lc = get_lake_config(os.path.join(lake_dir, "lake_config.csv"), 1)
    cwd = os.getcwd(); os.chdir(lake_dir)
    area, depth, volume, _ = get_hypsography(hypsofile=rc["hypso_ini_file"], dx=float(rc["dx"]),
                                            nx=int(rc["nx"]), outflow_depth=float(lc["outflow_depth"]))
    os.chdir(cwd)
    lake_volume = float(np.sum(volume)); lake_area = float(np.max(area))

    met = pd.read_csv(os.path.join(lake_dir, rc["meteo_ini_file"]),
                      usecols=["datetime", "Precipitation_millimeterPerDay", "Air_Temperature_celsius"])
    met["datetime"] = pd.to_datetime(met["datetime"])
    daily = met.set_index("datetime").resample("D").agg(
        {"Precipitation_millimeterPerDay": "sum", "Air_Temperature_celsius": "mean"})
    daily = daily.dropna()
    dates = daily.index
    P = jnp.asarray(daily["Precipitation_millimeterPerDay"].to_numpy())   # hourly amounts summed = mm/day
    T = jnp.asarray(daily["Air_Temperature_celsius"].to_numpy())
    doy = jnp.asarray(dates.dayofyear.to_numpy(), dtype=jnp.float64)
    area_m2 = args.area_km2 * 1e6

    oc_orig = pd.read_csv(os.path.join(lake_dir, rc["oc_load_file"]), parse_dates=["datetime"])
    tp_orig = pd.read_csv(os.path.join(lake_dir, rc["tp_ini_file"]), parse_dates=["datetime"])
    C_obs_mean = float(tp_orig["tp"].mean()) * 1e-3   # ug/L -> g/m3
    C0 = float(tp_orig["tp"].iloc[0]) * 1e-3

    results = {}
    for label, amp in (("scenario", args.spring_amplitude), ("nopulse", 0.0)):
        out = simulate_catchment(P, T, doy, area_m2, amp)
        results[label] = out
    # sigma0 fitted once on the scenario and reused for the counterfactual,
    # so the two lake-TP series differ only through the catchment load
    sc = results["scenario"]
    sigma0 = fit_sigma0(sc["TP_load"], sc["Q"], doy, lake_volume, C_obs_mean, C0)

    os.makedirs(args.out_dir, exist_ok=True)
    for label, out in results.items():
        Q = np.asarray(out["Q"]); oc = np.asarray(out["OC_load"]) / np.maximum(Q, 1e-9)
        C = np.asarray(lake_tp_balance(out["TP_load"], out["Q"], doy, lake_volume, sigma0, C0)) * 1e3
        suffix = "" if label == "scenario" else "_nopulse"
        pd.DataFrame({"datetime": dates.strftime("%Y-%m-%d"), "oc": oc, "discharge": Q}).to_csv(
            os.path.join(args.out_dir, f"oc_load_file{suffix}.csv"), index=False)
        pd.DataFrame({"datetime": dates.strftime("%Y-%m-%d"), "layer": "epi", "tp": C}).to_csv(
            os.path.join(args.out_dir, f"L0001-WQ-inflow{suffix}.csv"), index=False)
        results[label] = dict(out, oc=oc, lakeTP=C)

    # --- summary ---
    s, n = results["scenario"], results["nopulse"]
    years = (dates[-1] - dates[0]).days / 365.25
    Qs = np.asarray(s["Q"]); tpl_s = np.asarray(s["TP_load"]); tpl_n = np.asarray(n["TP_load"])
    spring = (dates.month >= 2) & (dates.month <= 5)
    print(f"Catchment {args.area_km2:g} km2 ({args.area_km2 * 1e6 / lake_area:.0f}x lake area {lake_area / 1e6:.2f} km2), "
          f"{years:.1f} years")
    print(f"  precipitation {float(jnp.sum(P)) / years:.0f} mm/yr, AET {float(np.sum(s['aet'])) / years:.0f} mm/yr, "
          f"runoff {Qs.sum() / area_m2 * 1e3 / years:.0f} mm/yr")
    print(f"  mean discharge {Qs.mean():.0f} m3/day (original oc_load_file: {oc_orig['discharge'].mean():.0f})")
    print(f"  TP load {tpl_s.sum() / years / 1e3:.0f} kg/yr ({tpl_s.sum() / years / area_m2 * 1e4 * 1e-3:.2f} kg/ha/yr); "
          f"Feb-May share {tpl_s[spring].sum() / tpl_s.sum():.0%} (no-pulse: {tpl_n[spring].sum() / tpl_n.sum():.0%})")
    print(f"  flow-weighted OC {np.asarray(s['OC_load']).sum() / Qs.sum():.1f} g/m3 (original oc {oc_orig['oc'].mean():.1f})")
    print(f"  in-lake TP: sigma0 {sigma0 * 365:.2f} /yr, mean {np.mean(s['lakeTP']):.1f} ug/L "
          f"(observed {C_obs_mean * 1e3:.1f})")
    print(f"Wrote oc_load_file.csv, L0001-WQ-inflow.csv (+ *_nopulse.csv counterfactual) to {args.out_dir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = (dates >= "2022-01-01")
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax[0].bar(dates[w], np.asarray(P)[w], color="tab:blue", width=1.0); ax[0].set_ylabel("P (mm/day)")
    ax[1].plot(dates[w], Qs[w], label="catchment model"); ax[1].set_ylabel("Q (m3/day)")
    o = oc_orig[oc_orig.datetime >= "2022-01-01"]; ax[1].plot(o.datetime, o.discharge, color="gray", lw=0.8, label="original file")
    ax[1].legend()
    ax[2].plot(dates[w], tpl_s[w] / 1e3, label="scenario (spring pulse)"); ax[2].plot(dates[w], tpl_n[w] / 1e3, label="no pulse")
    ax[2].set_ylabel("TP load (kg/day)"); ax[2].legend()
    ax[3].plot(dates[w], np.asarray(s["lakeTP"])[w], label="scenario"); ax[3].plot(dates[w], np.asarray(n["lakeTP"])[w], label="no pulse")
    t = tp_orig[tp_orig.datetime >= "2022-01-01"]; ax[3].plot(t.datetime, t.tp, "ko", ms=3, label="original (observed) TP")
    ax[3].set_ylabel("in-lake TP (ug/L)"); ax[3].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "catchment_summary.png"), dpi=120)
    print(f"Saved {os.path.join(args.out_dir, 'catchment_summary.png')}")


if __name__ == "__main__":
    main()
