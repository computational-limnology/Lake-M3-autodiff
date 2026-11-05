"""
Small "regime" corrections for sparse data (MCL option): instead of an hourly
LSTM, a handful of coefficients that depend on season, stratification, wind
and ice. Two heads, applied through full_step_mcl's hooks (the physics code is
not changed):

kz head (kz_override) -- interior mixing / thermocline sharpness
    Kz = clip( Kz_process * exp(c(z, t)) + s(t) * K_add(z, t),  min_kz, max_kz )
    c(z, t)     = M tanh( (sum_b m_b w_b(z) A_b . f(t)) / M )             (smooth cap +-M; m_b = band mask)
    K_add(z, t) = K_ref * sum_b w_b(z) * Mx tanh( softplus(10 B_b . f(t) - 4) / Mx )   (ADD_BANDS only)
    B = 0 -> ~9e-8 m2/s (below k_min); capped at K_ref * Mx (4e-5 m2/s by default).
    The additive part can create mixing where Kz_process sits at k_min (Ravn's
    summer thermocline core), which a multiplicative factor cannot.
    Energy cap on the additive part (buoyancy flux, W m-2):
        P_add = rho_0 * sum K_add N^2 V / A_0
        P_cap = f_max * (1 - ice) * P_wind_24h,      P_wind = rho_air Cd (wind_factor U10)^3
        with --energy-link also P_cap <= gamma * E_withheld / (dt A_0), where
        E_withheld = KE0 * max(1/eta - 1, 0) + KE_dissipated is the wind energy the
        TKE head kept from mixed-layer deepening in this step (so the two heads
        can't both add energy: thermocline mixing is paid for by less entrainment)
        s = P_cap / (P_cap + P_add)

TKE head (mix_override) -- mixed-layer deepening / thermocline depth
    eta(t) = exp( lo + (hi - lo) sigmoid(T . f(t) + c0) ),  eta in [0.3, 1.3],
             c0 chosen so that eta = 1 for T = 0;  KE0 -> eta * KE0 in mixing_step
    stratification gate (default on): eta -> 1 + (eta - 1) * sigmoid((log10(S + 1) - 1.5) / 0.15),
             S = Schmidt stability [J m-2], i.e. the head acts only when the lake is
             stratified (S >~ 30 J m-2). In mixed water near 4 degC (spring) the
             loss responds abruptly to entrainment and its gradients are unusable.
    optional dissipation during deepening (--tke-dissipation):
             1/L = softplus(D . f(t) - 8) / L_ref, L_ref = 1 m (D = 0 -> L ~ 3000 m, ~no-op)

Features f(t) (subset per head):
    const, sin_doy, cos_doy      season
    schmidt                      z-scored log10(Schmidt stability + 1)
    ice                          ice cover (0/1)
    wind7d                       z-scored log10 of the 7-day mean wind power
    lake_number                  z-scored log10 Lake Number (stratification vs 24-h wind)
    qnet                         asinh(net heat input to the water this step / 50 W m-2) / 2
                                 (TKE head only: it needs the state before heating)
Means/sds come from the training years (stored with the coefficients).

Band modes: "mixed_layer" (v1, 4 bands relative to the mixed-layer depth h,
multiplicative only) and "thermocline" (v2, 5 bands: mixed layer, above core,
thermocline core, below core, hypolimnion; core = N^2-weighted centre). All
bands sum to 1 at every depth.

Coefficients: dict {"mult": (n_bands, n_kz_feat), "add": (n_add, n_kz_feat),
"tke": (n_tke_feat,), "tke_L": (n_tke_feat,)}; zeros = the process model.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax_lakeModel_functions import full_step_mcl, calc_dens  # noqa: F401  (re-exported for callers)

FEATURES_ALL = ("const", "sin_doy", "cos_doy", "schmidt", "ice", "wind7d", "lake_number", "qnet")
KZ_FEATURES_V1 = ("const", "sin_doy", "cos_doy", "schmidt", "ice")
BAND_NAMES = {
    "mixed_layer": ("mixed layer", "upper metalimnion", "lower metalimnion", "hypolimnion"),
    "thermocline": ("mixed layer", "above core", "thermocline core", "below core", "hypolimnion"),
}
ADD_BANDS = {"mixed_layer": (), "thermocline": (1, 2, 3)}

BAND_CFG = dict(ml_transition=0.037, upper=(0.05, 0.07), lower=(0.25, 0.10), hypo=(0.45, 0.08))
THERMO_CFG = dict(ml_width=0.5, above=(-1.5, 0.75), core=(0.0, 0.75), below=(1.5, 0.75),
                  hypo=(3.0, 0.5), d_min=0.5, d_max=4.0)

RHO_AIR, RHO_0, G, CP = 1.225, 1000.0, 9.81, 4186.0
ADD_SCALE, ADD_OFFSET = 10.0, -4.0
ETA_LO, ETA_HI = np.log(0.3), np.log(1.3)
ETA_C0 = float(np.log((-ETA_LO / (ETA_HI - ETA_LO)) / (1 - (-ETA_LO / (ETA_HI - ETA_LO)))))   # eta = 1 at T = 0
DISS_OFFSET, L_REF = -8.0, 1.0
GATE_CENTER, GATE_WIDTH = 1.5, 0.15     # log10(Schmidt + 1): gate half-open at S ~ 30 J m-2
USTAR_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def mixed_layer_depth(dens, dx, sbl_drho, sbl_width):
    w = jax.nn.sigmoid((dens[0] + sbl_drho - dens) / sbl_width)
    return jnp.maximum(jnp.sum(w) * dx, dx)


def n2_profile(dens, depth):
    n2 = jnp.abs(dens[1:] - dens[:-1]) / (depth[1:] - depth[:-1]) * G / RHO_0
    return n2, 0.5 * (depth[1:] + depth[:-1])


def thermocline_core(dens, depth, cfg=THERMO_CFG):
    n2, zm = n2_profile(dens, depth)
    q = n2 ** 2 + 1e-30
    q = q / jnp.sum(q)
    zc = jnp.sum(q * zm)
    d = jnp.sqrt(jnp.sum(q * (zm - zc) ** 2) + 1e-12)
    return zc, jnp.clip(d, cfg["d_min"], cfg["d_max"])


def band_weights(depth, h, cfg=BAND_CFG):
    z_max = jnp.max(depth)
    S = jnp.maximum(z_max - h, 0.2 * z_max)
    s = depth - h
    b = [jax.nn.sigmoid(-s / (cfg["ml_transition"] * S))]
    for c, w in (cfg["upper"], cfg["lower"]):
        b.append(jnp.exp(-0.5 * ((s - c * S) / (w * S)) ** 2))
    c, w = cfg["hypo"]
    b.append(jax.nn.sigmoid((s - c * S) / (w * S)))
    phi = jnp.stack(b, axis=0)
    return phi / (jnp.sum(phi, axis=0, keepdims=True) + 1e-12)


def band_weights_thermocline(depth, h, zc, d, cfg=THERMO_CFG):
    s = (depth - zc) / d
    below_ml = jax.nn.sigmoid((depth - h) / cfg["ml_width"])
    b = [jax.nn.sigmoid(-(depth - h) / cfg["ml_width"])]
    for c, w in (cfg["above"], cfg["core"], cfg["below"]):
        b.append(jnp.exp(-0.5 * ((s - c) / w) ** 2) * below_ml)
    c, w = cfg["hypo"]
    b.append(jax.nn.sigmoid((s - c) / w) * below_ml)
    phi = jnp.stack(b, axis=0)
    return phi / (jnp.sum(phi, axis=0, keepdims=True) + 1e-12)


def schmidt_stability(dens, depth, area, volume, g=G):
    z_v = jnp.sum(depth * volume) / jnp.sum(volume)
    return jnp.maximum(g / area[0] * jnp.sum((depth - z_v) * dens * volume), 0.0)


def lake_number(dens, depth, area, volume, pw24, Cd):
    """Lake Number with u* from the 24-h mean wind power (Imberger & Patterson 1990)."""
    S = schmidt_stability(dens, depth, area, volume)
    z_D = jnp.max(depth)
    z_v = jnp.sum(depth * volume) / jnp.sum(volume)
    n2, zm = n2_profile(dens, depth)
    z_T = jnp.sum(zm * n2) / (jnp.sum(n2) + 1e-12)
    U = (jnp.maximum(pw24, 0.0) / (RHO_AIR * Cd) + 1e-12) ** (1.0 / 3.0)
    ustar2 = jnp.maximum(RHO_AIR * Cd * U ** 2 / RHO_0, USTAR_FLOOR ** 2)
    return S * (1.0 - z_T / z_D) / (RHO_0 * ustar2 * jnp.sqrt(area[0]) * (1.0 - z_v / z_D))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def raw_features(doy, dens, ice, forcing, geom, params, u_before=None, u_after=None):
    """Unnormalised feature values (dict)."""
    depth, area, volume = geom["depth"], geom["area"], geom["volume"]
    ang = 2.0 * jnp.pi * doy / 365.25
    out = dict(const=jnp.asarray(1.0), sin_doy=jnp.sin(ang), cos_doy=jnp.cos(ang),
               schmidt=jnp.log10(schmidt_stability(dens, depth, area, volume) + 1.0),
               ice=jnp.where(ice, 1.0, 0.0))
    if "Pw7d" in forcing:
        out["wind7d"] = jnp.log10(forcing["Pw7d"] + 1e-6)
    if "Pw24" in forcing:
        ln = lake_number(dens, depth, area, volume, forcing["Pw24"], params["Cd"])
        out["lake_number"] = jnp.log10(jnp.clip(ln, 1e-3, 1e5))
    if u_before is not None and u_after is not None:
        q = jnp.sum((u_after - u_before) * volume) * RHO_0 * CP / (area[0] * geom["dt"])
        out["qnet"] = jnp.arcsinh(q / 50.0) / 2.0
    return out


NORMED = ("schmidt", "wind7d", "lake_number")


def features(raw, norm, use):
    vals = []
    for k in use:
        v = raw[k]
        if k in NORMED:
            v = (v - norm[f"{k}_mu"]) / norm[f"{k}_sd"]
        vals.append(jnp.asarray(v, dtype=jnp.float64))
    return jnp.stack(vals)


# ---------------------------------------------------------------------------
# configuration and coefficients
# ---------------------------------------------------------------------------

def default_cfg(**kw):
    cfg = dict(band_mode="thermocline", additive=True, kz=True, tke=False, tke_dissipation=False,
               energy_link=False, gamma=1.0, kz_bands=None, tke_gate=True,
               max_log_correction=1.0, min_kz=1e-9, max_kz=10.0, k_ref=5e-6, add_max_factor=8.0, f_max=0.002,
               kz_features=KZ_FEATURES_V1, tke_features=("const", "sin_doy", "cos_doy", "schmidt", "wind7d",
                                                         "lake_number", "qnet"),
               norm=dict(schmidt_mu=0.0, schmidt_sd=1.0, wind7d_mu=0.0, wind7d_sd=1.0,
                         lake_number_mu=0.0, lake_number_sd=1.0))
    if "features" in kw:                      # v1/v2 name
        kw["kz_features"] = kw.pop("features")
    cfg.update(kw)
    return cfg


def zero_coef(cfg):
    nb = len(BAND_NAMES[cfg["band_mode"]])
    na = len(ADD_BANDS[cfg["band_mode"]])
    nk, nt = len(cfg["kz_features"]), len(cfg["tke_features"])
    return {"mult": jnp.zeros((nb, nk)), "add": jnp.zeros((na, nk)), "tke": jnp.zeros((nt,)), "tke_L": jnp.zeros((nt,))}


def band_mask(cfg):
    nb = len(BAND_NAMES[cfg["band_mode"]])
    act = range(nb) if cfg.get("kz_bands") is None else cfg["kz_bands"]
    return jnp.asarray([1.0 if b in act else 0.0 for b in range(nb)])


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------

def tke_terms(coef, state, forcing, u, ice, geom, params, cfg):
    """dict(ke_scale, [dissip_len]) for mixing_step, plus 'eta' for reporting."""
    dens = calc_dens(u)
    raw = raw_features(forcing["doy"], dens, ice, forcing, geom, params, u_before=state["u"], u_after=u)
    f = features(raw, cfg["norm"], cfg["tke_features"])
    eta = jnp.exp(ETA_LO + (ETA_HI - ETA_LO) * jax.nn.sigmoid(coef["tke"] @ f + ETA_C0))
    if cfg.get("tke_gate", True):
        gate = jax.nn.sigmoid((raw["schmidt"] - GATE_CENTER) / GATE_WIDTH)
        eta = 1.0 + (eta - 1.0) * gate
    out = dict(ke_scale=eta)
    if cfg.get("tke_dissipation"):
        out["dissip_len"] = L_REF / (jax.nn.softplus(coef["tke_L"] @ f + DISS_OFFSET) + 1e-12)
    return out


def correction_terms(coef, kz_process, dens, ice, forcing, geom, params, cfg, additive=None, mix_diag=None):
    """(kz_final, dict(P_add, P_cap, P_wind, s, K_add, zc, d, h)) for one step."""
    depth, dx = geom["depth"], geom["dx"]
    mode = cfg["band_mode"]
    additive = cfg.get("additive", False) if additive is None else additive
    h = mixed_layer_depth(dens, dx, params["sbl_drho"], params["sbl_width"])
    raw = raw_features(forcing["doy"], dens, ice, forcing, geom, params)
    f = features(raw, cfg["norm"], cfg["kz_features"])
    if mode == "thermocline":
        zc, d = thermocline_core(dens, depth)
        w = band_weights_thermocline(depth, h, zc, d)
    else:
        zc, d = jnp.asarray(0.0), jnp.asarray(0.0)
        w = band_weights(depth, h)
    M = cfg["max_log_correction"]
    A = coef["mult"] * band_mask(cfg)[:, None]
    c = M * jnp.tanh((w.T @ (A @ f)) / M)
    kz = kz_process * jnp.exp(c)
    aux = dict(h=h, zc=zc, d=d)
    if additive and len(ADD_BANDS[mode]):
        Mx = cfg["add_max_factor"]
        a = jax.nn.softplus(ADD_SCALE * (coef["add"] @ f) + ADD_OFFSET)
        a = Mx * jnp.tanh(a / Mx)
        K_add = cfg["k_ref"] * (w[jnp.array(ADD_BANDS[mode])].T @ a)
        n2, _ = n2_profile(dens, depth)
        n2c = jnp.concatenate([n2, n2[-1:]])
        P_add = RHO_0 * jnp.sum(K_add * n2c * geom["volume"]) / geom["area"][0]
        U = forcing["Uw"] * params["wind_factor"]
        P_wind = forcing["Pw24"] if "Pw24" in forcing else RHO_AIR * params["Cd"] * U ** 3
        P_cap = cfg["f_max"] * jnp.where(ice, 0.0, 1.0) * P_wind
        if cfg.get("energy_link") and mix_diag is not None:
            eta = mix_diag["ke_scale"]
            withheld = mix_diag["KE0"] * jnp.maximum(1.0 / eta - 1.0, 0.0) + mix_diag["KE_diss"]
            P_link = cfg["gamma"] * withheld / (geom["dt"] * geom["area"][0])
            P_cap = jnp.minimum(P_cap, P_link)
            aux["P_link"] = P_link
        s = P_cap / (P_cap + P_add + 1e-30)
        kz = kz + s * K_add
        aux.update(P_add=s * P_add, P_cap=P_cap, P_wind=P_wind, s=s, K_add=s * K_add)
    return jnp.clip(kz, cfg["min_kz"], cfg["max_kz"]), aux


def make_kz_override(coef, geom, params, cfg, additive=None):
    """Callable for full_step_mcl(..., kz_override=...); asks for this step's
    mixing diagnostics when the energy link is on."""
    def override(kz_process, u, ice, dens_u, forcing, ri, mix_diag=None):
        return correction_terms(coef, kz_process, dens_u, ice, forcing, geom, params, cfg, additive, mix_diag)[0]
    override.wants_mix_diag = bool(cfg.get("energy_link"))
    return override


def make_mix_override(coef, geom, params, cfg):
    """Callable for full_step_mcl(..., mix_override=...) (TKE head), or None."""
    if not cfg.get("tke"):
        return None

    def override(state, forcing, u, ice):
        t = tke_terms(coef, state, forcing, u, ice, geom, params, cfg)
        return {k: v for k, v in t.items() if k in ("ke_scale", "dissip_len")}
    return override


def make_overrides(coef, geom, params, cfg, heads=True):
    """(kz_override, mix_override) for the configured heads; heads=False -> process model."""
    if not heads:
        return None, None
    kz = make_kz_override(coef, geom, params, cfg) if cfg.get("kz", True) else None
    return kz, make_mix_override(coef, geom, params, cfg)


# ---------------------------------------------------------------------------
# forcing helpers and file I/O
# ---------------------------------------------------------------------------

def wind_power_mean(Uw, wind_factor, Cd, hours):
    """Trailing `hours` mean wind power rho_air Cd U^3 [W m-2] (numpy, hourly)."""
    p = RHO_AIR * Cd * (np.asarray(Uw) * wind_factor) ** 3
    c = np.cumsum(np.concatenate([[0.0], p]))
    n = np.arange(1, len(p) + 1)
    lo = np.maximum(n - hours, 0)
    return (c[n] - c[lo]) / (n - lo)


def wind_power_24h(Uw, wind_factor, Cd):
    return wind_power_mean(Uw, wind_factor, Cd, 24)


def load_coef_file(path):
    """(coef dict, cfg, years) from a saved mcl_regime_<lake>.npz (v1, v2 or v3)."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        kzf = tuple(str(x) for x in (z["kz_features"] if "kz_features" in keys else z["features"]))
        cfg = default_cfg(
            band_mode=str(z["band_mode"]) if "band_mode" in keys else "mixed_layer",
            additive=bool(z["additive"]) if "additive" in keys else False,
            kz=bool(z["kz"]) if "kz" in keys else True, tke=bool(z["tke"]) if "tke" in keys else False,
            tke_dissipation=bool(z["tke_dissipation"]) if "tke_dissipation" in keys else False,
            energy_link=bool(z["energy_link"]) if "energy_link" in keys else False,
            tke_gate=bool(z["tke_gate"]) if "tke_gate" in keys else True,
            kz_features=kzf,
        )
        if "tke_features" in keys:
            cfg["tke_features"] = tuple(str(x) for x in z["tke_features"])
        if "kz_bands" in keys and len(z["kz_bands"]):
            cfg["kz_bands"] = tuple(int(b) for b in z["kz_bands"])
        norm = dict(cfg["norm"])
        for k in list(norm):
            if k in keys:
                norm[k] = float(z[k])
        cfg["norm"] = norm
        for k in ("max_log_correction", "min_kz", "max_kz", "k_ref", "add_max_factor", "f_max", "gamma"):
            if k in keys:
                cfg[k] = float(z[k])
        mult = np.asarray(z["coef_mult"] if "coef_mult" in keys else z["coef"])
        add = (np.asarray(z["coef_add"]) if "coef_add" in keys
               else np.zeros((len(ADD_BANDS[cfg["band_mode"]]), mult.shape[1])))
        nt = len(cfg["tke_features"])
        tke = np.asarray(z["coef_tke"]) if "coef_tke" in keys else np.zeros(nt)
        tke_L = np.asarray(z["coef_tke_L"]) if "coef_tke_L" in keys else np.zeros(nt)
        years = {k: [int(y) for y in z[f"{k}_years"]] for k in ("train", "val", "test") if f"{k}_years" in keys}
    coef = {"mult": jnp.asarray(mult), "add": jnp.asarray(add), "tke": jnp.asarray(tke), "tke_L": jnp.asarray(tke_L)}
    return coef, cfg, years
