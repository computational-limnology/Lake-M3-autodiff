"""Tests for the wind-mixing MCL hook (mixing_step ke_scale / dissip_len /
return_diag; full_step_mcl mix_override). Run: python tests/test_mixing_hook.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import windows as W  # noqa: E402
import jax_lakeModel_functions as J  # noqa: E402

S = None


def _setup(days=10, start_day=180):
    global S
    if S is None:
        S = W.build_setup(os.path.join(ROOT, "Ravn"), record_end="1989-08-31", load_obs=False)
    i0 = 24 * start_day
    Fs = {k: jnp.asarray(v[:i0]) for k, v in S["forcing"].items()}
    s0, _ = jax.jit(lambda s: lax.scan(lambda s, f: (J.full_step(s, f, S["geom"], S["params"])[0], None), s, Fs))(S["s0"])
    F = {k: jnp.asarray(v[i0:i0 + 24 * days]) for k, v in S["forcing"].items()}
    return s0, F


def _run(s0, F, hook=None, keys=("mix_mld", "mix_KE0", "mix_KE_used", "mix_n_full", "mix_kp_partial")):
    def step(s, f):
        if hook is None:
            ns, d = J.full_step(s, f, S["geom"], S["params"])
        else:
            ns, d = J.full_step_mcl(s, f, S["geom"], S["params"], mix_override=lambda st, ff, u, ice: hook)
        return ns, {k: d[k] for k in keys}
    return jax.jit(lambda s: lax.scan(step, s, F))(s0)


def test_neutral_hook_identical():
    s0, F = _setup()
    ref, dref = _run(s0, F)
    for hook in (dict(ke_scale=1.0), dict(dissip_len=1e30), dict(ke_scale=1.0, dissip_len=1e30)):
        out, d = _run(s0, F, hook)
        assert np.array_equal(np.asarray(ref["u"]), np.asarray(out["u"])), hook
    print("OK  neutral hook (ke_scale 1, dissip_len 1e30) bit-identical to full_step")


def test_hook_acts_and_diagnostics():
    s0, F = _setup()
    res = {}
    for lab, hook in (("ke0", dict(ke_scale=0.0)), ("ke05", dict(ke_scale=0.5)), ("ref", None),
                      ("ke15", dict(ke_scale=1.5)), ("L2", dict(dissip_len=2.0))):
        _, d = _run(s0, F, hook)
        res[lab] = {k: np.asarray(v) for k, v in d.items()}
    assert np.all(res["ke0"]["mix_KE0"] == 0) and np.all(res["ke0"]["mix_n_full"] == 0)
    m = {k: float(v["mix_mld"].mean()) for k, v in res.items()}
    assert m["ke0"] <= m["ke05"] <= m["ref"] <= m["ke15"], m
    assert m["L2"] <= m["ref"], m
    for v in res.values():
        assert np.all(v["mix_KE_used"] <= v["mix_KE0"] * (1 + 1e-12) + 1e-9)
        assert np.all(v["mix_n_full"] >= 0) and np.allclose(v["mix_n_full"], np.round(v["mix_n_full"]))
        assert np.all((v["mix_kp_partial"] >= 0) & (v["mix_kp_partial"] <= 1 + 1e-12))
    print("OK  hook acts: mean mixed-layer depth (m) " + ", ".join(f"{k} {x:.2f}" for k, x in m.items())
          + f"; energy used / input (ref) {res['ref']['mix_KE_used'].sum() / max(res['ref']['mix_KE0'].sum(), 1e-30):.3f}")


def test_gradient_ke_scale():
    """d(mean temperature 5-15 m after 5 days)/d(ke_scale) vs central FD, with
    the model's bounded derivatives and with them lifted."""
    s0, F = _setup(days=5)
    depth = np.asarray(S["geom"]["depth"])
    sel = jnp.asarray((depth >= 5) & (depth < 15))

    def f_of(k):
        def step(s, f):
            ns, _ = J.full_step_mcl(s, f, S["geom"], S["params"], mix_override=lambda st, ff, u, ice: dict(ke_scale=k))
            return ns, None
        s, _ = lax.scan(step, s0, F)
        return jnp.sum(jnp.where(sel, s["u"], 0.0)) / jnp.sum(sel)

    def check():
        g = float(jax.jit(jax.grad(f_of))(1.0))
        lf = jax.jit(f_of)
        fd = (float(lf(1.0 + 1e-3)) - float(lf(1.0 - 1e-3))) / 2e-3
        return g, fd

    g_b, fd = check()
    saved = {k: getattr(J, k) for k in ("_EDDY_DIFFUSIVITY_GRAD_MAX", "_KP_GRAD_DD", "_OSBORN_GRAD_N2", "_KCAP_GRAD_WIDTH")}
    try:
        J._EDDY_DIFFUSIVITY_GRAD_MAX, J._KP_GRAD_DD = 1e300, 1e-30
        J._OSBORN_GRAD_N2, J._KCAP_GRAD_WIDTH = J._OSBORN_N2_FLOOR, 1e-9
        g_x, fd_x = check()
    finally:
        for k, v in saved.items():
            setattr(J, k, v)
    assert np.isfinite(g_b) and np.isfinite(g_x)
    print(f"OK  d(T 5-15 m)/d(ke_scale): FD {fd:+.4e}; AD bounded {g_b:+.4e} (ratio {g_b / fd:.2f}), "
          f"AD unbounded {g_x:+.4e} (ratio {g_x / fd_x:.2f})")


if __name__ == "__main__":
    test_neutral_hook_identical()
    test_hook_acts_and_diagnostics()
    test_gradient_ke_scale()
