"""
Train the small "regime" corrections (mcl/mcl_regime.py) on sparse profile
data with yearly windows run in parallel -- the sparse-data alternative to the
hourly-LSTM train_M3_mcl_jax.py.

Two heads (--heads kz,tke = both, the default; ablations: --heads kz / --heads tke):
  kz head   interior mixing, mainly the thermocline's sharpness: multiplicative
            factor per depth band (default thermocline bands without the mixed-
            layer band, which had no effect on depth or sharpness) plus an
            additive term in the above-core / core / below-core bands.
  TKE head  mixed-layer deepening, mainly the thermocline's depth: efficiency
            eta in [0.3, 1.3] on the wind energy in mixing_step (optional
            dissipation length, --tke-dissipation).
diagnose_mixing.py showed the split of roles: entrainment energy sets the depth
(+3..4 m per log-factor), interior diffusion the sharpness (-1.3..-1.6 degC/m
per log-factor), surface-layer diffusion neither. With --energy-link (default
on) the additive thermocline mixing can only use (gamma x) the wind energy the
TKE head keeps from deepening in the same step (and at most --f-max of the
24-h wind power), so the two heads can't both add energy.

Data split (by window year, default): test 2016-2024 even years (reported
only), validation every 4th remaining year (stopping, best weights), training
the rest. Manual profiles only by default (--temp-obs profiles).

Loss (training windows, after the 28-day unscored spin-up):
  MSE(T) + --grad-weight MSE(dT_1m) + --depth-weight MSE(z_core)
         + --l2 sum(coef^2) + --energy-penalty mean(P_add / P_wind)
  dT_1m = T(z + 1 m) - T(z) on the observation grid (sharpness);
  z_core = N^2-weighted thermocline core depth of profiles reaching --zmin m
  with max dT/dz > 1 degC/m (observed vs modelled at the same time).
Validation score = sqrt(MSE(T) + grad_weight MSE(dT_1m) + depth_weight MSE(z_core)).

Optimiser: Adam, separate cosine-decaying learning rates per head (--lr-kz
0.02, --lr-tke 0.01 -> x --lr-final-frac 0.1), gradient clipping.
Noise-aware stopping as before (tol = max(--tol, 2 x measured round-off noise);
--worsen-patience 3, --plateau-patience 8, --min-iters 5).

Report per split: process vs corrected on the same start states (RMSE all /
per layer, RMSE dT_1m, RMSE z_core, summer eta, added mixing power), then
(unless --no-continuous) continuous runs: RMSE per split, summer thermocline
metrics, summer thermocline-depth bias per year; the corrected run is saved
like run_M3_jax.py output.

Usage (project root):
  python mcl/train_mcl_regime.py Ravn                         # both heads
  python mcl/train_mcl_regime.py Ravn --heads tke             # ablation: TKE head only
  python mcl/train_mcl_regime.py Ravn --apply mcl/mcl_regime_Ravn.npz
  python mcl/run_M3_mcl_jax.py Ravn --nn-params mcl_regime.npz
Outputs: mcl/mcl_regime_<lake>.npz with _coef.csv, _history.csv,
_windowed_rmse.csv, _continuous_rmse.csv, _thermocline.csv, _depth_by_year.csv;
<lake>/res_lake1_mcl_regime.npz.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SRC_DIR = _os.path.join(_os.path.dirname(_HERE), "src")
for _p in (_SRC_DIR, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


def _early_device_flags(argv):
    n = int(argv[argv.index("--devices") + 1]) if "--devices" in argv else (_os.cpu_count() or 1)
    flags = _os.environ.get("XLA_FLAGS", "")
    if n > 1 and "xla_force_host_platform_device_count" not in flags:
        flags += f" --xla_force_host_platform_device_count={n}"
    if "--multi-thread" not in argv and "xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    _os.environ["XLA_FLAGS"] = flags.strip()
    if n > 1 and "--xla-warnings" not in argv:
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import xla_log_filter
        xla_log_filter.install()


if __name__ == "__main__":
    _early_device_flags(_sys.argv)

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
import optax  # noqa: E402

import windows as W  # noqa: E402
import jax_lakeModel_functions as JM  # noqa: E402
from jax_lakeModel_functions import full_step_mcl, calc_dens  # noqa: E402
from data_tools import mcl_path  # noqa: E402
import mcl_regime as R  # noqa: E402

LAYERS = (("epi", 0.0, 5.0), ("meta", 5.0, 15.0), ("hypo", 15.0, 1e9))
DIFF_CELLS = 2        # dT over 2 grid cells (= 1 m at dx = 0.5 m)
SUMMER_DOY = (152, 274)
HEAD_OF = {"mult": "kz", "add": "kz", "tke": "tke", "tke_L": "tke"}


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def add_forcing(S):
    """Day of year and 24-h / 7-day mean wind power as extra forcing series."""
    t = S["start"] + pd.to_timedelta(np.arange(S["n_steps"]), unit="h")
    S["forcing"]["doy"] = np.asarray(t.dayofyear, dtype=float) + np.asarray(t.hour, dtype=float) / 24.0
    wf, cd = float(S["params"]["wind_factor"]), float(S["params"]["Cd"])
    S["forcing"]["Pw24"] = R.wind_power_mean(S["forcing"]["Uw"], wf, cd, 24)
    S["forcing"]["Pw7d"] = R.wind_power_mean(S["forcing"]["Uw"], wf, cd, 24 * 7)
    return S


def parse_years(spec, years):
    if not spec:
        return []
    if spec.startswith("every:"):
        n = int(spec.split(":", 1)[1])
        return [y for i, y in enumerate(sorted(years)) if i % n == n - 1]
    return [int(y) for y in spec.split(",") if y.strip()]


def split_windows(win, test_spec, val_spec):
    years = win["year"].tolist()
    test = [y for y in parse_years(test_spec, years) if y in years]
    rest = [y for y in years if y not in test]
    val = [y for y in parse_years(val_spec, rest) if y in rest]
    train = [y for y in rest if y not in val]
    pick = lambda ys: win[win["year"].isin(ys)].reset_index(drop=True)
    return pick(train), pick(val), pick(test)


def feature_norms(S, daily, windows_df):
    """Mean/sd of the z-scored features over the days of the given windows
    (daily states of a process-model run; wind from the forcing)."""
    geom, params, F = S["geom"], S["params"], S["forcing"]
    days = np.unique(np.concatenate([np.arange(r.i0 // 24, r.i1 // 24) for r in windows_df.itertuples()]))
    days = days[days < len(daily["u"])]
    u = jnp.asarray(daily["u"][days])
    pw24 = jnp.asarray(F["Pw24"][np.minimum(days * 24, len(F["Pw24"]) - 1)])
    raw = jax.vmap(lambda uu, p: R.raw_features(0.0, calc_dens(uu), False, {"Pw24": p}, geom, params))(u, pw24)
    hours = np.concatenate([np.arange(24 * d, 24 * d + 24) for d in days])
    hours = hours[hours < len(F["Pw7d"])]
    vals = dict(schmidt=np.asarray(raw["schmidt"]), lake_number=np.asarray(raw["lake_number"]),
                wind7d=np.log10(F["Pw7d"][hours] + 1e-6))
    norm = {}
    for k, v in vals.items():
        norm[f"{k}_mu"] = float(np.mean(v))
        norm[f"{k}_sd"] = float(max(np.std(v), 1e-6))
    return norm


def mix_diag_of(d):
    return dict(KE0=d["mix_KE0"], ke_scale=d["mix_ke_scale"], KE_diss=d["mix_KE_diss"])


def step_energy(coef, s_new, f, d, geom, params, cfg, heads):
    """(P_add / P_wind, cap share s) for one step, from the end-of-step state
    (diagnostic/penalty; the cap itself acts inside the step)."""
    if not (heads and cfg.get("kz", True) and cfg.get("additive") and len(R.ADD_BANDS[cfg["band_mode"]])):
        return jnp.asarray(0.0), jnp.asarray(1.0)
    _, aux = R.correction_terms(coef, s_new["kz"], calc_dens(s_new["u"]), s_new["ice"], f, geom, params, cfg,
                                additive=True, mix_diag=mix_diag_of(d))
    return aux["P_add"] / (aux["P_wind"] + 1e-30), aux["s"]


# ---------------------------------------------------------------------------
# continuous run (start states / final run)
# ---------------------------------------------------------------------------

def run_continuous(S, coef, cfg, outputs=(), heads=True, days_per_call=365, energy=False):
    """Daily states (leading day axis; day 0 = initial) and per-step outputs.
    heads=False runs the process model. energy=True adds 'e_ratio', 's', 'eta'."""
    geom, params = S["geom"], S["params"]
    coef = jax.tree_util.tree_map(jnp.asarray, coef)
    kz_ov, mix_ov = R.make_overrides(coef, geom, params, cfg, heads)
    outputs = tuple(outputs)

    @jax.jit
    def run_days(s, f_days):
        def day(s, fd):
            def step(s, f):
                ns, d = full_step_mcl(s, f, geom, params, kz_override=kz_ov, mix_override=mix_ov)
                o = {k: (d[k] if k in d else ns[k]) for k in outputs}
                if energy:
                    o["e_ratio"], o["s"] = step_energy(coef, ns, f, d, geom, params, cfg, heads)
                    o["eta"] = d["mix_ke_scale"]
                return ns, o
            s, o = lax.scan(step, s, fd)
            return s, (s, o)
        return lax.scan(day, s, f_days)

    F = S["forcing"]
    n_days = S["n_steps"] // 24
    s = S["s0"]
    states = {k: [np.asarray(v)[None]] for k, v in s.items()}
    keys = outputs + (("e_ratio", "s", "eta") if energy else ())
    outs = {k: [] for k in keys}
    d = 0
    while d < n_days:
        nd = min(days_per_call, n_days - d)
        f = {k: jnp.asarray(v[24 * d:24 * (d + nd)].reshape((nd, 24) + v.shape[1:])) for k, v in F.items()}
        s, (ss, oo) = run_days(s, f)
        for k in states:
            states[k].append(np.asarray(ss[k]))
        for k in keys:
            v = np.asarray(oo[k])
            outs[k].append(v.reshape((-1,) + v.shape[2:]))
        d += nd
    states = {k: np.concatenate(v, axis=0) for k, v in states.items()}
    outs = {k: np.concatenate(v, axis=0) for k, v in outs.items()}
    return states, outs


# ---------------------------------------------------------------------------
# windowed loss
# ---------------------------------------------------------------------------

def obs_core_depth(val, mask, depth, k, min_grad=1.0):
    """Observed thermocline core depth (as R.thermocline_core) and eligibility
    (profile complete to cell k and max dT/dz > min_grad) per profile."""
    complete = mask[..., :k].all(axis=-1)
    v = np.where(mask, val, 0.0)[..., :k]
    flat = v.reshape(-1, k)
    zc = np.asarray(jax.vmap(lambda p: R.thermocline_core(calc_dens(p), jnp.asarray(depth[:k]))[0])(jnp.asarray(flat)))
    g = (-np.diff(flat, axis=1) / np.diff(depth[:k])[None]).max(axis=1)
    elig = complete.reshape(-1) & (g > min_grad)
    return zc.reshape(v.shape[:-1]), elig.reshape(v.shape[:-1])


class RegimeWindows:
    """Windowed errors of the corrected model for one window set. Per window:
    SSE of T per layer (3), SSE of dT_1m, SSE of z_core, mean energy ratio,
    share of hours at the cap, summer mean eta."""

    NOUT = 8

    def __init__(self, S, win, cfg, window_steps, chunk_steps, n_devices, energy_penalty=0.0, zmin=25.0):
        self.S, self.win, self.cfg, self.D = S, win.reset_index(drop=True), cfg, n_devices
        self.energy_penalty = energy_penalty
        depth = np.asarray(S["depth"])
        self.F = W.window_forcing(S["forcing"], self.win, window_steps)
        o = W.window_obs(S["obs"]["temp"], self.win)
        m = o["mask"] & o["valid"][:, :, None]
        lm = np.stack([(depth >= a) & (depth < b) for _, a, b in LAYERS])
        m3 = (m[:, None, :, :] & lm[None, :, None, :]).astype(float)
        kd = DIFF_CELLS
        md = (m[:, :, kd:] & m[:, :, :-kd]).astype(float)
        vd = o["values"][:, :, kd:] - o["values"][:, :, :-kd]
        self.kz_cells = int(np.searchsorted(depth, zmin))
        zc, elig = obs_core_depth(o["values"], m, depth, self.kz_cells)
        elig = elig & o["valid"]
        self.O = dict(idx=o["step_idx"], val=o["values"], m3=m3, md=md, vd=vd, zc=zc, ez=elig.astype(float))
        self.counts = m3.sum(axis=(2, 3))
        self.n = float(self.counts.sum())
        self.nd = float(md.sum())
        self.nz = float(elig.sum())
        self.states = None
        self.c = W.fit_chunk(window_steps, chunk_steps)
        self._fns = {}

    def _build(self, heads, with_energy):
        S, cfg, c = self.S, self.cfg, self.c
        geom, params = S["geom"], S["params"]
        kd, kc = DIFF_CELLS, self.kz_cells
        dk = jnp.asarray(np.asarray(S["depth"])[:kc])

        def sim(coef, state0, forcing):
            kz_ov, mix_ov = R.make_overrides(coef, geom, params, cfg, heads)

            @jax.checkpoint
            def step(s, f):
                ns, d = full_step_mcl(s, f, geom, params, kz_override=kz_ov, mix_override=mix_ov)
                er, sc = step_energy(coef, ns, f, d, geom, params, cfg, heads) if with_energy else (0.0, 1.0)
                return ns, (ns["u"], jnp.asarray(er), jnp.asarray(sc), d["mix_ke_scale"], f["doy"])

            @jax.checkpoint
            def chunk(s, fc):
                s, out = lax.scan(step, s, fc)
                return jax.tree_util.tree_map(lax.stop_gradient, s), out

            n = forcing["Uw"].shape[0]
            fch = jax.tree_util.tree_map(lambda v: v.reshape((n // c, c) + v.shape[1:]), forcing)
            _, out = lax.scan(chunk, state0, fch)
            return jax.tree_util.tree_map(lambda v: v.reshape((n,) + v.shape[2:]), out)

        def per_window(coef, state0, forcing, ob):
            u, er, sc, eta, doy = sim(coef, state0, forcing)
            ui = u[ob["idx"]]
            sse3 = jnp.einsum("lkn,kn->l", ob["m3"], (ui - ob["val"]) ** 2)
            sse_d = jnp.sum(ob["md"] * ((ui[:, kd:] - ui[:, :-kd]) - ob["vd"]) ** 2)
            zc = jax.vmap(lambda p: R.thermocline_core(calc_dens(p[:kc]), dk)[0])(ui)
            sse_z = jnp.sum(ob["ez"] * (zc - ob["zc"]) ** 2)
            summ = ((doy >= SUMMER_DOY[0]) & (doy < SUMMER_DOY[1])).astype(float)
            eta_s = jnp.sum(eta * summ) / jnp.maximum(jnp.sum(summ), 1.0)
            return jnp.concatenate([sse3, jnp.stack([sse_d, sse_z, jnp.mean(er), jnp.mean(sc < 0.5), eta_s])])

        def batched(coef, st, F, O):
            return W.run_batched(per_window, coef, (st, F, O), n_devices=self.D)
        return batched

    def _fn(self, heads, with_energy):
        key = (bool(heads), bool(with_energy))
        if key not in self._fns:
            self._fns[key] = jax.jit(self._build(*key))
        return self._fns[key]

    def make_value_and_grad(self, grad_weight, depth_weight, l2):
        with_energy = self.energy_penalty > 0
        batched = self._build(True, with_energy)
        n, nd, nz = max(self.n, 1.0), max(self.nd, 1.0), max(self.nz, 1.0)

        def loss(coef, st, F, O):
            r = batched(coef, st, F, O)
            mse, mse_d, mse_z = jnp.sum(r[:, :3]) / n, jnp.sum(r[:, 3]) / nd, jnp.sum(r[:, 4]) / nz
            reg = l2 * sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(coef))
            pen = self.energy_penalty * jnp.mean(r[:, 5])
            return mse + grad_weight * mse_d + depth_weight * mse_z + reg + pen, (mse, mse_d, mse_z)

        vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
        return lambda coef: vg(coef, self.states, self.F, self.O)

    def set_states(self, daily):
        self.states = W.window_start_states(daily, self.win)

    def scores(self, coef, heads=True, with_energy=True, grad_weight=0.0, depth_weight=0.0):
        r = np.asarray(self._fn(heads, with_energy)(coef, self.states, self.F, self.O))
        mse = r[:, :3].sum() / max(self.n, 1.0)
        mse_d = r[:, 3].sum() / max(self.nd, 1.0)
        mse_z = r[:, 4].sum() / max(self.nz, 1.0)
        out = dict(rmse=float(np.sqrt(mse)), rmse_dT1m=float(np.sqrt(mse_d)), rmse_zcore=float(np.sqrt(mse_z)),
                   score=float(np.sqrt(mse + grad_weight * mse_d + depth_weight * mse_z)),
                   eta_summer=float(r[:, 7].mean()), energy_pct_wind=float(100 * r[:, 5].mean()),
                   share_capped=float(r[:, 6].mean()))
        for j, (name, _, _) in enumerate(LAYERS):
            out[name] = float(np.sqrt(r[:, j].sum() / max(self.counts[:, j].sum(), 1.0)))
        return out


# ---------------------------------------------------------------------------
# continuous evaluation
# ---------------------------------------------------------------------------

def continuous_scores(S, u, years_by_split):
    o = S["obs"]["temp"]
    idx = np.asarray(o["step_idx"]); val = np.asarray(o["values"]); m = np.asarray(o["mask"])
    yrs = (S["start"] + pd.to_timedelta(idx, unit="h")).year
    depth = np.asarray(S["depth"])
    rows = {}
    for split, ys in years_by_split.items():
        sel = np.isin(yrs, ys) & (idx < len(u))
        e = (u[idx[sel]] - val[sel])
        mm = m[sel]
        r = {"all": float(np.sqrt(np.mean(e[mm] ** 2))) if mm.any() else np.nan}
        for name, a, b in LAYERS:
            ml = mm & ((depth >= a) & (depth < b))[None, :]
            r[name] = float(np.sqrt(np.mean(e[ml] ** 2))) if ml.any() else np.nan
        rows[split] = r
    return rows


def thermocline_metrics(S, u, years, months=(6, 7, 8, 9), zmin=25.0):
    o = S["obs"]["temp"]
    idx = np.asarray(o["step_idx"]); val = np.asarray(o["values"]); m = np.asarray(o["mask"])
    t = S["start"] + pd.to_timedelta(idx, unit="h")
    depth = np.asarray(S["depth"])
    kz = int(np.searchsorted(depth, zmin))
    sel = np.isin(t.year, years) & np.isin(t.month, months) & m[:, :kz].all(axis=1) & (idx < len(u))
    if not sel.any():
        return None
    obs = val[sel][:, :kz]; mod = u[idx[sel]][:, :kz]
    dz = np.diff(depth[:kz])

    def tc(p):
        g = -np.diff(p, axis=1) / dz[None]
        j = np.argmax(g, axis=1)
        return 0.5 * (depth[j] + depth[j + 1]), g[np.arange(len(g)), j]

    zo, go = tc(obs); zm, gm = tc(mod)
    rel = depth[:kz][None, :] - zo[:, None]
    e = mod - obs
    out = dict(n_profiles=int(sel.sum()), max_dTdz_obs=float(go.mean()), max_dTdz_model=float(gm.mean()),
               thermocline_depth_bias_m=float((zm - zo).mean()))
    for lo in range(-6, 8, 2):
        mm = (rel >= lo) & (rel < lo + 2)
        out[f"bias_{lo:+d}_{lo + 2:+d}m"] = float(e[mm].mean()) if mm.any() else np.nan
    return out


def coef_tables(coef, cfg):
    names = R.BAND_NAMES[cfg["band_mode"]]
    parts = []
    if cfg.get("kz", True):
        mask = np.asarray(R.band_mask(cfg))
        t = pd.DataFrame(np.asarray(coef["mult"]), index=list(names), columns=list(cfg["kz_features"]))
        t = t[mask > 0]
        t.insert(0, "part", "kz mult (log factor)")
        parts.append(t)
        if cfg.get("additive") and len(R.ADD_BANDS[cfg["band_mode"]]):
            ta = pd.DataFrame(np.asarray(coef["add"]), index=[names[i] for i in R.ADD_BANDS[cfg["band_mode"]]],
                              columns=list(cfg["kz_features"]))
            ta.insert(0, "part", "kz add")
            parts.append(ta)
    if cfg.get("tke"):
        tt = pd.DataFrame([np.asarray(coef["tke"])], index=["entrainment efficiency"], columns=list(cfg["tke_features"]))
        tt.insert(0, "part", "tke eta")
        parts.append(tt)
        if cfg.get("tke_dissipation"):
            tl = pd.DataFrame([np.asarray(coef["tke_L"])], index=["dissipation"], columns=list(cfg["tke_features"]))
            tl.insert(0, "part", "tke 1/L")
            parts.append(tl)
    t = pd.concat(parts) if parts else pd.DataFrame()
    t.index.name = "row"
    return t


def continuous_report(S, coef, cfg, years, res_out=None, save_fields=True):
    t = time.time()
    # kz (corrected) and kz_process (the closure's own value along the same run) make the
    # file readable by mcl/plot_mcl_kz.py (same keys as a run_M3_mcl_jax.py result)
    fields = (("u", "o2", "docr", "docl", "pocr", "pocl", "kd_light", "kz", "kz_process")
              if save_fields else ("u",))
    _, out_c = run_continuous(S, coef, cfg, outputs=fields, energy=True)
    _, out_b = run_continuous(S, coef, cfg, outputs=("u",), heads=False)
    n = min(len(out_c["u"]), S["n_steps"])
    uc, ub = out_c["u"][:n], out_b["u"][:n]
    sb, sc = continuous_scores(S, ub, years), continuous_scores(S, uc, years)
    rows, trows, yrows = [], [], []
    for split in years:
        rows.append(dict(split=split, model="process", **sb[split]))
        rows.append(dict(split=split, model="corrected", **sc[split]))
        for lab, uu in (("process", ub), ("corrected", uc)):
            tm = thermocline_metrics(S, uu, years[split])
            if tm:
                trows.append(dict(split=split, model=lab, **tm))
        for y in years[split]:
            tb, tcm = thermocline_metrics(S, ub, [y]), thermocline_metrics(S, uc, [y])
            if tb and tcm:
                yrows.append(dict(year=y, split=split, n=tb["n_profiles"], process=tb["thermocline_depth_bias_m"],
                                  corrected=tcm["thermocline_depth_bias_m"]))
    ct = pd.DataFrame(rows).set_index(["split", "model"])
    tt = pd.DataFrame(trows).set_index(["split", "model"]) if trows else pd.DataFrame()
    yt = pd.DataFrame(yrows).sort_values("year").set_index("year") if yrows else pd.DataFrame()
    tim = S["start"] + pd.to_timedelta(np.arange(n), unit="h")
    summer = np.isin(tim.month, (6, 7, 8, 9))
    en = dict(eta_summer=float(out_c["eta"][:n][summer].mean()),
              energy_pct_wind_summer=float(100 * out_c["e_ratio"][:n][summer].mean()),
              share_capped_summer=float((out_c["s"][:n][summer] < 0.5).mean()))
    print(f"\nContinuous temperature RMSE (degC) per split ({time.time() - t:.0f}s):")
    print(ct.round(4).to_string())
    if len(tt):
        print("\nSummer thermocline (Jun-Sep profiles reaching 25 m): max dT/dz [degC/m], depth bias [m], "
              "bias [degC] by depth relative to the observed thermocline:")
        print(tt.round(3).to_string())
    if len(yt):
        print("\nSummer thermocline-depth bias [m] per year:")
        print(yt.round(2).T.to_string())
    print(f"summer: mean entrainment efficiency eta {en['eta_summer']:.3f}; added mixing power "
          f"{en['energy_pct_wind_summer']:.4f} % of the 24-h wind power, cap active {100 * en['share_capped_summer']:.1f} % "
          "of hours")
    if save_fields:
        np.savez(res_out, temp=uc, o2=out_c["o2"][:n], docr=out_c["docr"][:n], docl=out_c["docl"][:n],
                 pocr=out_c["pocr"][:n], pocl=out_c["pocl"][:n], kd_light=out_c["kd_light"][:n],
                 kz=out_c["kz"][:n], kz_process=out_c["kz_process"][:n],
                 eta=out_c["eta"][:n], add_power_frac_wind=out_c["e_ratio"][:n],
                 start_time=np.array(str(S["start"])), times=S["step_times"][:n],
                 depth=np.asarray(S["depth"]), volume=np.asarray(S["volume"]))
        print(f"saved continuous corrected run to {res_out}")
    return ct, tt, yt, en


def apply_file(lake_dir, coef_path, res_out=None, record_end=None, temp_obs="profiles", report=True):
    """Run a trained correction file continuously (--apply and run_M3_mcl_jax.py)."""
    lake_dir = os.path.abspath(lake_dir)
    coef, cfg, years = R.load_coef_file(coef_path)
    S = add_forcing(W.build_setup(lake_dir, record_end=record_end, temp_obs=temp_obs))
    print(f"{coef_path}: heads kz={cfg.get('kz', True)} tke={cfg.get('tke', False)}, band mode {cfg['band_mode']}, "
          f"additive {cfg['additive']}, energy link {cfg.get('energy_link', False)}")
    print(coef_tables(coef, cfg).round(3).to_string())
    if not years:
        yrs = sorted(set((S["start"] + pd.to_timedelta(np.asarray(S["obs"]["temp"]["step_idx"]), unit="h")).year))
        years = {"all": yrs}
    res_out = res_out or os.path.join(lake_dir, "res_lake1_mcl_regime.npz")
    return continuous_report(S, coef, cfg, years, res_out=res_out)


def save_coef_file(path, coef, cfg, extra):
    c = jax.tree_util.tree_map(np.asarray, coef)
    np.savez(path, coef_mult=c["mult"], coef_add=c["add"], coef_tke=c["tke"], coef_tke_L=c["tke_L"],
             band_mode=np.array(cfg["band_mode"]), additive=np.array(cfg["additive"]), kz=np.array(cfg.get("kz", True)),
             tke=np.array(cfg.get("tke", False)), tke_dissipation=np.array(cfg.get("tke_dissipation", False)),
             energy_link=np.array(cfg.get("energy_link", False)), gamma=cfg.get("gamma", 1.0),
             tke_gate=np.array(cfg.get("tke_gate", True)),
             kz_features=np.array(cfg["kz_features"]), tke_features=np.array(cfg["tke_features"]),
             features=np.array(cfg["kz_features"]),
             kz_bands=np.array(cfg["kz_bands"] if cfg.get("kz_bands") is not None else [], dtype=int),
             bands=np.array(R.BAND_NAMES[cfg["band_mode"]]),
             max_log_correction=cfg["max_log_correction"], min_kz=cfg["min_kz"], max_kz=cfg["max_kz"],
             k_ref=cfg["k_ref"], add_max_factor=cfg["add_max_factor"], f_max=cfg["f_max"],
             **cfg["norm"], **extra)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--heads", default="kz,tke", help="kz, tke or kz,tke (default)")
    ap.add_argument("--band-mode", choices=tuple(R.BAND_NAMES), default="thermocline")
    ap.add_argument("--kz-bands", default="1,2,3,4",
                    help="kz bands with a multiplicative factor (thermocline mode: 0 mixed layer, 1 above core, "
                         "2 core, 3 below core, 4 hypolimnion; default 1,2,3,4 = without the mixed layer; 'all')")
    ap.add_argument("--no-additive", action="store_true", help="kz head multiplicative only")
    ap.add_argument("--kz-features", default="const,sin_doy,cos_doy,schmidt,ice,wind7d,lake_number")
    ap.add_argument("--tke-features", default="const,sin_doy,cos_doy,schmidt,wind7d,lake_number,qnet")
    ap.add_argument("--tke-dissipation", action="store_true", help="TKE head: also learn a dissipation length")
    ap.add_argument("--no-energy-link", action="store_true",
                    help="additive kz mixing capped by f_max x wind power only (not by the energy the TKE head withholds)")
    ap.add_argument("--gamma", type=float, default=1.0, help="energy link: usable share of the withheld energy")
    ap.add_argument("--k-ref", type=float, default=5e-6)
    ap.add_argument("--add-max-factor", type=float, default=8.0)
    ap.add_argument("--f-max", type=float, default=0.002)
    ap.add_argument("--energy-penalty", type=float, default=0.0)
    ap.add_argument("--grad-weight", type=float, default=0.5, help="weight of MSE(dT_1m) (default 0.5)")
    ap.add_argument("--depth-weight", type=float, default=0.25, help="weight of MSE(thermocline core depth) [m2] (default 0.25)")
    ap.add_argument("--zmin", type=float, default=25.0, help="depth-loss profiles must reach this depth")
    ap.add_argument("--temp-obs", default="profiles", choices=("profiles", "all"))
    ap.add_argument("--test-years", default="2016,2018,2020,2022,2024")
    ap.add_argument("--val-years", default="every:4")
    ap.add_argument("--record-end", default=None)
    ap.add_argument("--window-start", default="02-01")
    ap.add_argument("--window-steps", type=int, default=8016)
    ap.add_argument("--spinup-days", type=int, default=28)
    ap.add_argument("--chunk-steps", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr-kz", type=float, default=0.02)
    ap.add_argument("--lr-tke", type=float, default=0.01)
    ap.add_argument("--lr-final-frac", type=float, default=0.1, help="final / initial learning rate (cosine)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad-ri-max", type=float, default=1e4,
                    help="bound on the derivative of the surface-layer Richardson number used while training "
                         "(model default 1, see _ri_safe_div). With 1 the TKE-head gradient has the wrong sign "
                         "in summer (AD/FD -6.7 over 56 days); 1e4 gives ~1.1 at moderate size. Forward runs "
                         "are unaffected (derivative-only bound).")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--worsen-patience", type=int, default=3)
    ap.add_argument("--plateau-patience", type=int, default=8)
    ap.add_argument("--min-iters", type=int, default=5)
    ap.add_argument("--refresh-every", type=int, default=5)
    ap.add_argument("--max-log-correction", type=float, default=1.0)
    ap.add_argument("--min-kz", type=float, default=1e-9)
    ap.add_argument("--max-kz", type=float, default=10.0)
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--multi-thread", action="store_true")
    ap.add_argument("--xla-warnings", action="store_true")
    ap.add_argument("--no-continuous", action="store_true")
    ap.add_argument("--apply", default=None)
    ap.add_argument("--out", default="mcl_regime.npz")
    ap.add_argument("--res-out", default=None)
    a = ap.parse_args()

    lake_dir = os.path.abspath(a.lake_dir)
    t_all = time.time()
    JM._EDDY_DIFFUSIVITY_GRAD_MAX = a.grad_ri_max      # derivative-only (training); forward unchanged
    if a.apply:
        apply_file(lake_dir, a.apply, a.res_out, a.record_end, a.temp_obs)
        print(f"total {time.time() - t_all:.0f}s")
        return

    heads = {h.strip() for h in a.heads.split(",") if h.strip()}
    if not heads or heads - {"kz", "tke"}:
        ap.error("--heads: kz, tke or kz,tke")
    parse_f = lambda spec: tuple(f for f in R.FEATURES_ALL if f in {s.strip() for s in spec.split(",")})
    kzf, tkf = parse_f(a.kz_features), parse_f(a.tke_features)
    bad = [s for s in (a.kz_features + "," + a.tke_features).split(",") if s.strip() and s.strip() not in R.FEATURES_ALL]
    if bad:
        ap.error(f"unknown features {bad}; choose from {R.FEATURES_ALL}")
    if "qnet" in kzf:
        ap.error("qnet is only available to the TKE head")
    nb = len(R.BAND_NAMES[a.band_mode])
    kz_bands = None if a.kz_bands == "all" else tuple(int(b) for b in a.kz_bands.split(",") if b.strip())
    if kz_bands is not None and any(b >= nb for b in kz_bands):
        kz_bands = tuple(b for b in kz_bands if b < nb)
    additive = (not a.no_additive) and len(R.ADD_BANDS[a.band_mode]) > 0 and "kz" in heads
    out_path = mcl_path(a.out, lake_dir)
    stem = os.path.splitext(out_path)[0]

    S = add_forcing(W.build_setup(lake_dir, record_end=a.record_end, temp_obs=a.temp_obs))
    if S["obs"].get("temp") is None:
        raise SystemExit("no temperature observations")
    D = jax.device_count()
    win = W.select_windows(S, a.window_start, a.window_steps, a.spinup_days, require_obs="temp")
    tr, va, te = split_windows(win, a.test_years, a.val_years)
    cfg = R.default_cfg(band_mode=a.band_mode, additive=additive, kz="kz" in heads, tke="tke" in heads,
                        tke_dissipation=a.tke_dissipation, energy_link=(not a.no_energy_link) and {"kz", "tke"} <= heads,
                        gamma=a.gamma, kz_bands=kz_bands, kz_features=kzf, tke_features=tkf,
                        max_log_correction=a.max_log_correction, min_kz=a.min_kz, max_kz=a.max_kz,
                        k_ref=a.k_ref, add_max_factor=a.add_max_factor, f_max=a.f_max)
    print(f"record {S['start'].date()} + {S['n_steps']} h; {D} device(s); temperature obs: {a.temp_obs}")
    print(f"heads: {sorted(heads)}; kz bands {kz_bands if kz_bands is not None else 'all'}, additive {additive}, "
          f"energy link {cfg['energy_link']} (gamma {a.gamma:g}), TKE dissipation {a.tke_dissipation}")
    print(f"  kz features {kzf}\n  TKE features {tkf}")
    print(f"loss MSE(T) + {a.grad_weight:g} MSE(dT_1m) + {a.depth_weight:g} MSE(z_core)")
    print(f"windows ({a.window_steps} h from {a.window_start}, {a.spinup_days}-day spin-up): "
          f"train {tr['year'].tolist()}\n  validation {va['year'].tolist()}\n  test {te['year'].tolist()}")
    if not len(tr) or not len(va):
        raise SystemExit("need at least one training and one validation window")
    zero = R.zero_coef(cfg)

    t = time.time()
    daily, _ = run_continuous(S, zero, cfg, heads=False)
    cfg["norm"] = feature_norms(S, daily, tr)
    print(f"reference run (process model): {time.time() - t:.0f}s; feature norms "
          + ", ".join(f"{k} {v:.3g}" for k, v in cfg["norm"].items()), flush=True)

    sets = {name: RegimeWindows(S, w, cfg, a.window_steps, a.chunk_steps, D, a.energy_penalty, a.zmin)
            for name, w in (("train", tr), ("val", va), ("test", te)) if len(w)}
    for ws in sets.values():
        ws.set_states(daily)
    print("scored values: " + ", ".join(f"{k} {int(v.n)} T / {int(v.nz)} core depths" for k, v in sets.items()),
          flush=True)
    gw, dw = a.grad_weight, a.depth_weight
    vg = sets["train"].make_value_and_grad(gw, dw, a.l2)
    val_score = lambda c: sets["val"].scores(c, with_energy=False, grad_weight=gw, depth_weight=dw)

    coef = zero
    t = time.time()
    s0 = val_score(coef)
    s1 = val_score(jax.tree_util.tree_map(lambda v: v + 1e-9, coef))
    v0 = s0["score"]
    noise = abs(s1["score"] - v0) / max(v0, 1e-12)
    tol = max(a.tol, 2 * noise)
    print(f"validation score {v0:.4f} (RMSE {s0['rmse']:.4f}, dT_1m {s0['rmse_dT1m']:.4f}, core depth "
          f"{s0['rmse_zcore']:.3f} m); round-off noise {100 * noise:.2f} % -> tolerance {100 * tol:.2f} %"
          + ("  (raised above --tol by the noise)" if tol > a.tol else "") + f"  ({time.time() - t:.0f}s)", flush=True)

    def sched(lr):
        return optax.cosine_decay_schedule(lr, max(a.iters, 1), alpha=a.lr_final_frac)
    txs = {"kz": optax.adam(sched(a.lr_kz)) if "kz" in heads else optax.set_to_zero(),
           "tke": optax.adam(sched(a.lr_tke)) if "tke" in heads else optax.set_to_zero()}
    labels = {k: HEAD_OF[k] for k in coef}
    if not a.tke_dissipation:
        txs["off"] = optax.set_to_zero()
        labels["tke_L"] = "off"
    if not additive:
        txs.setdefault("off", optax.set_to_zero())
        labels["add"] = "off"
    opt = optax.chain(optax.clip_by_global_norm(a.clip) if a.clip > 0 else optax.identity(),
                      optax.multi_transform(txs, labels))
    state = opt.init(coef)

    best, best_val, ref_val = coef, v0, v0
    worse, since_improve = 0, 0
    hist = [dict(iter=0, val_score=v0, val_rmse=s0["rmse"], val_rmse_dT=s0["rmse_dT1m"], val_rmse_z=s0["rmse_zcore"])]
    reason = f"reached --iters {a.iters}"
    for it in range(1, a.iters + 1):
        t = time.time()
        refreshed = False
        if a.refresh_every and it > 1 and (it - 1) % a.refresh_every == 0:
            daily, _ = run_continuous(S, coef, cfg)
            for ws in sets.values():
                ws.set_states(daily)
            best_val = val_score(best)["score"]
            ref_val = best_val
            refreshed = True
            print(f"  [start states refreshed; best re-scored: validation score {best_val:.4f}]", flush=True)
        (L, (mse, mse_d, mse_z)), g = vg(coef)
        if not (np.isfinite(float(L)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))):
            reason = "non-finite loss or gradient"
            break
        gmax = max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(g))
        if gmax > 10:
            print(f"  WARNING: max |gradient| {gmax:.3g} > 10 (clipped to global norm {a.clip:g}); "
                  "if this repeats, lower --grad-ri-max", flush=True)
        upd, state = opt.update(g, state, coef)
        coef_eval = coef
        coef = optax.apply_updates(coef, upd)
        sv = val_score(coef_eval)
        v = sv["score"]
        hist.append(dict(iter=it, train_mse=float(mse), train_mse_dT=float(mse_d), train_mse_z=float(mse_z),
                         val_score=v, val_rmse=sv["rmse"], val_rmse_dT=sv["rmse_dT1m"], val_rmse_z=sv["rmse_zcore"],
                         val_eta_summer=sv["eta_summer"], refreshed=refreshed))
        flag = ""
        if v < best_val:
            best, best_val = coef_eval, v
            flag = " *"
        if v < ref_val * (1 - tol):
            ref_val, since_improve = v, 0
        else:
            since_improve += 1
        worse = worse + 1 if v > best_val * (1 + tol) else 0
        print(f"  iter {it:3d}: train RMSE {np.sqrt(float(mse)):.4f} (dT_1m {np.sqrt(float(mse_d)):.4f}, core "
              f"{np.sqrt(float(mse_z)):.2f} m)  validation {v:.4f} (RMSE {sv['rmse']:.4f}, dT_1m {sv['rmse_dT1m']:.4f}, "
              f"core {sv['rmse_zcore']:.2f} m, eta {sv['eta_summer']:.3f}){flag}  max|g| {gmax:.2g}  "
              f"({time.time() - t:.0f}s)", flush=True)
        if it >= a.min_iters and worse >= a.worsen_patience:
            reason = f"validation score worse than best x (1 + {tol:.3g}) for {worse} iterations"
            break
        if it >= a.min_iters and since_improve >= a.plateau_patience:
            reason = f"no validation improvement > {100 * tol:.2g} % in {since_improve} iterations"
            break
    print(f"stopped: {reason}")
    pd.DataFrame(hist).to_csv(stem + "_history.csv", index=False)

    daily, _ = run_continuous(S, best, cfg)
    for ws in sets.values():
        ws.set_states(daily)
    rows = []
    for name, ws in sets.items():
        rows.append(dict(split=name, model="process", **ws.scores(zero, heads=False, grad_weight=gw, depth_weight=dw)))
        rows.append(dict(split=name, model="corrected", **ws.scores(best, grad_weight=gw, depth_weight=dw)))
    wt = pd.DataFrame(rows).set_index(["split", "model"])
    print("\nWindowed scores, same start states (from a run with the best coefficients); RMSE in degC / m, "
          "energy in % of 24-h wind power:")
    print(wt.round(4).to_string())
    tab = coef_tables(best, cfg)
    print("\nCoefficients (kz mult: log mixing factor per band; kz add: K_add = k_ref softplus(10 B.f - 4); "
          "tke eta: entrainment efficiency, eta = 1 at 0, < 1 for negative T.f):")
    print(tab.round(3).to_string())

    save_coef_file(out_path, best, cfg, dict(
        grad_weight=gw, depth_weight=dw, energy_penalty=a.energy_penalty, grad_ri_max=a.grad_ri_max,
        train_years=np.array(tr["year"]), val_years=np.array(va["year"]), test_years=np.array(te["year"]),
        stop_reason=np.array(reason), tol=tol))
    tab.to_csv(stem + "_coef.csv")
    wt.to_csv(stem + "_windowed_rmse.csv")
    print(f"saved {out_path} (+ _coef.csv, _history.csv, _windowed_rmse.csv)", flush=True)

    if not a.no_continuous:
        years = {"train": tr["year"].tolist(), "val": va["year"].tolist(), "test": te["year"].tolist()}
        res_out = a.res_out or os.path.join(lake_dir, "res_lake1_mcl_regime.npz")
        ct, tt, yt, en = continuous_report(S, best, cfg, years, res_out=res_out)
        ct.to_csv(stem + "_continuous_rmse.csv")
        if len(tt):
            tt.to_csv(stem + "_thermocline.csv")
        if len(yt):
            yt.to_csv(stem + "_depth_by_year.csv")
        print("saved _continuous_rmse.csv, _thermocline.csv, _depth_by_year.csv")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
