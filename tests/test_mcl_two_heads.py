"""Tests for the two-head regime correction (kz head + TKE head, energy link).
Run from the project root: python tests/test_mcl_two_heads.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "mcl")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import windows as W  # noqa: E402
import jax_lakeModel_functions as J  # noqa: E402
import mcl_regime as R  # noqa: E402
import train_mcl_regime as T  # noqa: E402

_S = {}


def _setup(days=10, start_day=180):
    if "S" not in _S:
        S = T.add_forcing(W.build_setup(os.path.join(ROOT, "Ravn"), record_end="1989-08-31", load_obs=False))
        i0 = 24 * start_day
        Fs = {k: jnp.asarray(v[:i0]) for k, v in S["forcing"].items()}
        s0, _ = jax.jit(lambda s: lax.scan(lambda s, f: (J.full_step(s, f, S["geom"], S["params"])[0], None), s, Fs))(S["s0"])
        _S.update(S=S, s0=s0, i0=i0)
    S, s0, i0 = _S["S"], _S["s0"], _S["i0"]
    F = {k: jnp.asarray(v[i0:i0 + 24 * days]) for k, v in S["forcing"].items()}
    return S, s0, F


def _cfg(**kw):
    cfg = R.default_cfg(tke=True, energy_link=True, kz_bands=(1, 2, 3, 4),
                        kz_features=("const", "sin_doy", "cos_doy", "schmidt", "ice", "wind7d", "lake_number"), **kw)
    return cfg


def _run(S, s0, F, coef, cfg, heads=True, keys=("mix_mld", "mix_ke_scale")):
    kz_ov, mix_ov = R.make_overrides(coef, S["geom"], S["params"], cfg, heads)

    def step(s, f):
        ns, d = J.full_step_mcl(s, f, S["geom"], S["params"], kz_override=kz_ov, mix_override=mix_ov)
        return ns, {k: d[k] for k in keys}
    return jax.jit(lambda s: lax.scan(step, s, F))(s0)


def test_features_finite():
    S, s0, F = _setup(days=1)
    f0 = {k: v[12] for k, v in F.items()}
    raw = R.raw_features(f0["doy"], J.calc_dens(s0["u"]), False, f0, S["geom"], S["params"],
                         u_before=s0["u"], u_after=s0["u"] + 0.01)
    assert all(np.isfinite(float(v)) for v in raw.values()), raw
    assert float(raw["qnet"]) > 0          # warming -> positive heat input
    print("OK  features: " + ", ".join(f"{k} {float(v):.3g}" for k, v in raw.items()))


def test_zero_tke_and_direction():
    S, s0, F = _setup()
    cfg = _cfg()
    z = R.zero_coef(cfg)
    ref, dref = _run(S, s0, F, z, cfg, heads=False)
    out, dout = _run(S, s0, F, z, cfg)
    assert np.allclose(np.asarray(dout["mix_ke_scale"]), 1.0, atol=1e-12)
    dmax = float(np.max(np.abs(np.asarray(ref["u"]) - np.asarray(out["u"]))))
    assert dmax < 1e-8, dmax
    mld = {}
    for lab, c in (("eta<1", -3.0), ("eta=1", 0.0), ("eta>1", 3.0)):
        coef = dict(z); coef["tke"] = z["tke"].at[0].set(c)
        _, d = _run(S, s0, F, coef, cfg)
        mld[lab] = (float(np.asarray(d["mix_mld"]).mean()), float(np.asarray(d["mix_ke_scale"]).mean()))
    assert mld["eta<1"][0] <= mld["eta=1"][0] <= mld["eta>1"][0], mld
    assert 0.3 - 1e-9 <= mld["eta<1"][1] < 1 < mld["eta>1"][1] <= 1.3 + 1e-9, mld
    print(f"OK  zero TKE coefficients = process model (max |dT| {dmax:.1e} degC after 10 days); mean MLD / eta: "
          + ", ".join(f"{k} {v[0]:.2f} m / {v[1]:.2f}" for k, v in mld.items()))


def test_energy_link():
    S, s0, F = _setup()
    cfg = _cfg()
    z = R.zero_coef(cfg)
    big_add = dict(z); big_add["add"] = z["add"].at[:, 0].set(5.0)
    # eta = 1 -> nothing withheld -> the additive term must be exactly zero
    a, _ = _run(S, s0, F, dict(big_add), cfg)
    b, _ = _run(S, s0, F, z, cfg)
    assert np.array_equal(np.asarray(a["kz"]), np.asarray(b["kz"])), "eta = 1: energy link must block K_add"
    # formula: P_add <= gamma * withheld / (dt A0) and <= f_max P_wind
    f0 = {k: v[12] for k, v in F.items()}
    dens = J.calc_dens(s0["u"])
    for eta, ke0, diss in ((0.5, 50.0, 0.0), (0.8, 5.0, 2.0), (0.3, 1e3, 0.0), (1.0, 50.0, 0.0)):
        md = dict(KE0=jnp.asarray(ke0), ke_scale=jnp.asarray(eta), KE_diss=jnp.asarray(diss))
        _, aux = R.correction_terms(big_add, s0["kz"], dens, False, f0, S["geom"], S["params"], cfg, True, md)
        link = cfg["gamma"] * (ke0 * max(1 / eta - 1, 0) + diss) / (S["geom"]["dt"] * float(S["geom"]["area"][0]))
        assert float(aux["P_add"]) <= min(link, float(aux["P_wind"]) * cfg["f_max"]) * (1 + 1e-9) + 1e-30
    # with eta < 1 the additive mixing is allowed again
    both = dict(big_add); both["tke"] = z["tke"].at[0].set(-3.0)
    c, _ = _run(S, s0, F, both, cfg)
    d, _ = _run(S, s0, F, dict(z, tke=both["tke"]), cfg)
    assert float(np.max(np.asarray(c["kz"]) - np.asarray(d["kz"]))) > 0
    print("OK  energy link: no withheld energy -> no added mixing (bit-identical kz); P_add <= gamma x withheld "
          "and <= f_max x wind power; with eta < 1 added mixing appears")


def test_gradient_tke():
    S, s0, F = _setup(days=5)
    cfg = _cfg()
    z = R.zero_coef(cfg)
    target = _run(S, s0, F, z, cfg, heads=False)[0]["u"] - 0.3
    sel = jnp.asarray((np.asarray(S["geom"]["depth"]) >= 3) & (np.asarray(S["geom"]["depth"]) < 15))

    def loss(c):
        kz_ov, mix_ov = R.make_overrides(c, S["geom"], S["params"], cfg)
        s, _ = lax.scan(lambda s, f: (J.full_step_mcl(s, f, S["geom"], S["params"], kz_override=kz_ov,
                                                      mix_override=mix_ov)[0], None), s0, F)
        return jnp.sum(jnp.where(sel, (s["u"] - target) ** 2, 0.0)) / jnp.sum(sel)

    def check(c0):
        g = float(jax.jit(jax.grad(loss))(c0)["tke"][0])
        lf = jax.jit(loss)
        e = 1e-6   # small step: the energy link has a kink at eta = 1 (max(1/eta - 1, 0))
        cp = dict(c0, tke=c0["tke"].at[0].add(e)); cm = dict(c0, tke=c0["tke"].at[0].add(-e))
        return g, (float(lf(cp)) - float(lf(cm))) / (2 * e)

    g_b, fd = check(z)
    saved = {k: getattr(J, k) for k in ("_EDDY_DIFFUSIVITY_GRAD_MAX", "_KP_GRAD_DD", "_OSBORN_GRAD_N2", "_KCAP_GRAD_WIDTH")}
    try:
        J._EDDY_DIFFUSIVITY_GRAD_MAX, J._KP_GRAD_DD = 1e300, 1e-30
        J._OSBORN_GRAD_N2, J._KCAP_GRAD_WIDTH = J._OSBORN_N2_FLOOR, 1e-9
        g_x, fd_x = check(z)
    finally:
        for k, v in saved.items():
            setattr(J, k, v)
    assert np.isfinite(g_b) and abs(g_x - fd_x) <= 1e-3 * abs(fd_x) + 1e-12, (g_x, fd_x)
    print(f"OK  TKE gradient: FD {fd:+.4e}; AD bounded {g_b:+.4e} (ratio {g_b / fd:.2f}); "
          f"unbounded ratio {g_x / fd_x:.3f}")


if __name__ == "__main__":
    test_features_finite()
    test_zero_tke_and_direction()
    test_energy_link()
    test_gradient_tke()
