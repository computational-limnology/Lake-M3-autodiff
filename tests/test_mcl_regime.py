"""Tests for the regime Kz correction (mcl/mcl_regime.py).
Run from the project root: python tests/test_mcl_regime.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "mcl")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import windows as W  # noqa: E402
from jax_lakeModel_functions import full_step, full_step_mcl, calc_dens  # noqa: E402
import mcl_regime as R  # noqa: E402

NORM = dict(schmidt_mu=1.0, schmidt_sd=1.0)


def _cfg(**kw):
    return R.default_cfg(norm=NORM, **kw)


def _setup(days=10):
    S = W.build_setup(os.path.join(ROOT, "Ravn"), record_end="1989-08-31", load_obs=False)
    t = S["start"] + pd.to_timedelta(np.arange(S["n_steps"]), unit="h")
    S["forcing"]["doy"] = np.asarray(t.dayofyear, float)
    S["forcing"]["Pw24"] = R.wind_power_24h(S["forcing"]["Uw"], float(S["params"]["wind_factor"]), float(S["params"]["Cd"]))
    i0 = 24 * 180   # end of June (stratified)
    F = {k: jnp.asarray(v[i0:i0 + 24 * days]) for k, v in S["forcing"].items()}
    Fs = {k: jnp.asarray(v[:i0]) for k, v in S["forcing"].items()}
    s0, _ = jax.jit(lambda s: lax.scan(lambda s, f: (full_step(s, f, S["geom"], S["params"])[0], None), s, Fs))(S["s0"])
    return S, s0, F


def _run(S, s0, F, coef, cfg, additive):
    if coef is None:
        step = lambda s, f: (full_step(s, f, S["geom"], S["params"])[0], None)
    else:
        ov = R.make_kz_override(coef, S["geom"], S["params"], cfg, additive)
        step = lambda s, f: (full_step_mcl(s, f, S["geom"], S["params"], kz_override=ov)[0], None)
    return jax.jit(lambda s: lax.scan(step, s, F)[0])(s0)


def test_band_weights():
    depth = jnp.linspace(0.25, 31.75, 64)
    for h in (1.0, 5.0, 12.0, 30.0):
        w = np.asarray(R.band_weights(depth, h))
        assert w.shape == (4, 64) and np.allclose(w.sum(0), 1.0) and (w >= 0).all()
    # v2 on a synthetic profile: 20 degC to 5 m, linear drop to 8 degC at 9 m (core ~7 m), then 8 degC
    T = jnp.where(depth < 5, 20.0, jnp.where(depth < 9, 20.0 - 3.0 * (depth - 5), 8.0))
    dens = calc_dens(T)
    zc, d = R.thermocline_core(dens, depth)
    h = R.mixed_layer_depth(dens, 0.5, 0.05, 0.01)
    w = np.asarray(R.band_weights_thermocline(depth, h, zc, d))
    assert w.shape == (5, 64) and np.allclose(w.sum(0), 1.0) and (w >= 0).all()
    k = int(np.argmax(w[2]))
    assert 5.0 < float(zc) < 9.0 and abs(float(depth[k]) - float(zc)) <= 0.5, (float(zc), float(depth[k]))
    assert w[0, 0] > 0.99 and w[4, -1] > 0.99
    print(f"OK  band weights (v1, v2); synthetic thermocline: core {float(zc):.2f} m, thickness {float(d):.2f} m, "
          f"mixed layer {float(h):.2f} m")


def test_zero_start_and_cap():
    S, s0, F = _setup()
    ref = _run(S, s0, F, None, None, False)
    cfg = _cfg()
    z = R.zero_coef(cfg)
    off = _run(S, s0, F, z, cfg, False)
    for k in ("u", "kz"):
        assert np.array_equal(np.asarray(ref[k]), np.asarray(off[k])), f"{k}: zero coef must be bit-identical"
    for k in ref.keys():
        np.testing.assert_allclose(np.asarray(off[k], float), np.asarray(ref[k], float), rtol=1e-12, atol=0)
    # additive part at zero coefficients: K_add = k_ref * softplus(-4) ~ 9e-8 at most
    f0 = {k: v[0] for k, v in F.items()}
    dens = calc_dens(s0["u"])
    kz0, aux = R.correction_terms(z, s0["kz"], dens, s0["ice"], f0, S["geom"], S["params"], cfg, additive=True)
    assert float(jnp.max(aux["K_add"])) <= cfg["k_ref"] * float(jax.nn.softplus(R.ADD_SCALE * 0 + R.ADD_OFFSET)) * 1.0001
    # large additive coefficients: K_add <= k_ref * add_max_factor and power <= cap
    big = {"mult": z["mult"], "add": z["add"].at[:, 0].set(5.0)}
    for k_step in range(0, 24 * 10, 37):
        fk = {kk: v[k_step] for kk, v in F.items()}
        _, a = R.correction_terms(big, s0["kz"], dens, False, fk, S["geom"], S["params"], cfg, additive=True)
        assert float(jnp.max(a["K_add"])) <= cfg["k_ref"] * cfg["add_max_factor"] * (1 + 1e-9)
        assert float(a["P_add"]) <= float(a["P_cap"]) * (1 + 1e-9) + 1e-30
    _, a_ice = R.correction_terms(big, s0["kz"], dens, True, f0, S["geom"], S["params"], cfg, additive=True)
    assert float(a_ice["s"]) == 0.0 and float(jnp.max(a_ice["K_add"])) == 0.0
    print("OK  zero coefficients = process model (u, kz bit-identical); additive start <= k_ref*softplus(-4); "
          "K_add <= k_ref*max factor; added power <= f_max * wind power; none under ice")


def test_v1_file_loads():
    path = os.path.join(ROOT, "mcl", "mcl_regime_Ravn.npz")
    if not os.path.exists(path):
        print("skip v1 file test (mcl/mcl_regime_Ravn.npz not present)")
        return
    coef, cfg, years = R.load_coef_file(path)
    with np.load(path) as zz:
        if "band_mode" in zz.files:
            print("skip v1 file test (file is already v2)")
            return
        assert np.allclose(np.asarray(coef["mult"]), zz["coef"])
    assert cfg["band_mode"] == "mixed_layer" and not cfg["additive"] and coef["add"].shape[0] == 0
    print(f"OK  v1 file loads (mixed_layer, multiplicative; years {sorted(years)})")


def test_gradient_exact():
    """AD vs central FD for one multiplicative and one additive coefficient over
    5 days with the model's bounded custom derivatives lifted (they shrink some
    derivatives on purpose; see CLAUDE.md finding 17)."""
    import jax_lakeModel_functions as J
    S, s0, F = _setup(days=5)
    cfg = _cfg()
    target = _run(S, s0, F, None, None, False)["u"] + 0.2

    def loss(c):
        ov = R.make_kz_override(c, S["geom"], S["params"], cfg, True)
        s, _ = lax.scan(lambda s, f: (full_step_mcl(s, f, S["geom"], S["params"], kz_override=ov)[0], None), s0, F)
        return jnp.mean((s["u"] - target) ** 2)

    saved = {k: getattr(J, k) for k in ("_EDDY_DIFFUSIVITY_GRAD_MAX", "_KP_GRAD_DD", "_OSBORN_GRAD_N2", "_KCAP_GRAD_WIDTH")}
    try:
        J._EDDY_DIFFUSIVITY_GRAD_MAX, J._KP_GRAD_DD = 1e300, 1e-30
        J._OSBORN_GRAD_N2, J._KCAP_GRAD_WIDTH = J._OSBORN_N2_FLOOR, 1e-9
        c0 = R.zero_coef(cfg)
        c0 = {"mult": c0["mult"], "add": c0["add"].at[:, 0].set(0.3)}   # additive term active (~6e-6 m2/s)
        g = jax.jit(jax.grad(loss))(c0)
        lf = jax.jit(loss)
        res = []
        for part, (i, j) in (("mult", (2, 0)), ("add", (1, 0))):
            e = 1e-4
            cp = dict(c0); cm = dict(c0)
            cp[part] = c0[part].at[i, j].add(e); cm[part] = c0[part].at[i, j].add(-e)
            fd = (float(lf(cp)) - float(lf(cm))) / (2 * e)
            ad = float(g[part][i, j])
            res.append((part, ad, fd))
            assert np.isfinite(ad) and abs(ad - fd) <= 2e-3 * abs(fd) + 1e-14, (part, ad, fd)
    finally:
        for k, v in saved.items():
            setattr(J, k, v)
    print("OK  gradient exact (unbounded derivatives): " + ", ".join(f"{p} AD {a:+.4e} FD {f:+.4e}" for p, a, f in res))


if __name__ == "__main__":
    test_band_weights()
    test_v1_file_loads()
    test_zero_start_and_cap()
    test_gradient_exact()
