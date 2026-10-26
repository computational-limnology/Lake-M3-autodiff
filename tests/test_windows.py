"""Window infrastructure (src/windows.py): a window started from the reference
run's saved state reproduces the continuous run exactly, and the batched
loss/gradient equals the sum over single windows (to round-off). Short record,
runs in about a minute:
    python -m pytest tests/test_windows.py   (or: python tests/test_windows.py)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import windows as W  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

_S = None


def _setup():
    global _S
    if _S is None:
        S = W.build_setup(ROOT / "Ravn", record_end=None, load_obs=False)
        # first ~70 days of the record are enough
        S["n_steps"] = 24 * 70
        S["forcing"] = {k: v[:S["n_steps"]] for k, v in S["forcing"].items()}
        _S = S
    return _S


def _windows(S, steps):
    start = S["start"]
    md = (start + np.timedelta64(31, "D")).strftime("%m-%d")
    return W.select_windows(S, md, steps, 0, require_obs=None)


def test_single_window_matches_continuous_run():
    S = _setup()
    win = _windows(S, 240)
    daily = W.reference_states(S, None, days_per_call=35)
    F = W.window_forcing(S["forcing"], win, 240)
    st = W.window_start_states(daily, win)
    sim = W.make_window_simulator(S["geom"], S["params"], daily_states=True)
    sb = W.as_state({k: jnp.asarray(v[0]) for k, v in st.items()})
    fb = {k: jnp.asarray(v[0]) for k, v in F.items()}
    res = jax.jit(lambda s, f: sim(None, s, f)["state"])(sb, fb)
    d0 = int(win["i0"].iloc[0] // 24)
    ref = daily["u"][d0 + 1:d0 + 11]
    assert np.max(np.abs(np.asarray(res["u"]) - ref)) == 0.0


def test_batched_loss_and_gradient_equal_sum_of_windows():
    S = _setup()
    win = _windows(S, 48)
    win = W.pd.concat([win, win.assign(i0=win["i0"] + 96, i1=win["i1"] + 96)], ignore_index=True)
    daily = W.reference_states(S, None, days_per_call=35)
    F = W.window_forcing(S["forcing"], win, 48)
    st = W.window_start_states(daily, win)
    sim = W.make_window_simulator(S["geom"], S["params"], chunk_steps=24)
    theta = {k: jnp.log(jnp.asarray(float(S["params"][k]))) for k in ("Cd", "wind_factor")}
    loss1 = lambda th, s, f: jnp.mean(sim(th, s, f)["u"])
    v, g = jax.value_and_grad(lambda th: W.run_batched(loss1, th, (st, F), reduce_sum=True))(theta)
    vs, gs = 0.0, {k: 0.0 for k in theta}
    for b in range(len(win)):
        sb = W.as_state({k: jnp.asarray(x[b]) for k, x in st.items()})
        fb = {k: jnp.asarray(x[b]) for k, x in F.items()}
        vb, gb = jax.value_and_grad(lambda th: loss1(th, sb, fb))(theta)
        vs += float(vb)
        gs = {k: gs[k] + float(gb[k]) for k in theta}
    assert abs(float(v) - vs) <= 1e-6 * abs(vs)
    for k in theta:
        assert abs(float(g[k]) - gs[k]) <= 1e-3 * abs(gs[k]) + 1e-12


if __name__ == "__main__":
    test_single_window_matches_continuous_run()
    test_batched_loss_and_gradient_equal_sum_of_windows()
    print("OK: windows match the continuous run; batched loss/gradient = sum over windows")
