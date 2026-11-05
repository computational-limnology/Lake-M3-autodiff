"""Tests for src/lm_optimizer.py on synthetic least-squares problems.
Run: python tests/test_lm.py   (or pytest tests/test_lm.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lm_optimizer import lm_minimize, covariance_report  # noqa: E402


def _quiet(*a, **k):
    pass


def test_rosenbrock():
    """Classic hard valley: r = (10 (x1 - x0^2), 1 - x0), minimum at (1, 1)."""
    def res(x):
        return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

    def jac(x):
        return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])

    def ne(x):
        r, J = res(x), jac(x)
        return J.T @ J, J.T @ r, r @ r

    out = lm_minimize(ne, lambda x: res(x) @ res(x), np.array([-1.2, 1.0]), [-10, -10], [10, 10],
                      max_iters=200, max_step=10, ftol=1e-14, xtol=1e-10, log=_quiet)
    assert np.allclose(out["x"], [1, 1], atol=1e-5), out


def test_exponential_fit_and_covariance():
    """y = a exp(-b t) + noise, fitted in log-parameters; compare with
    scipy.optimize.curve_fit (parameter values and relative standard errors)."""
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(1)
    t = np.linspace(0, 5, 200)
    a, b, sig = 3.0, 0.7, 0.05
    y = a * np.exp(-b * t) + sig * rng.standard_normal(t.size)

    def model(x):   # x = log(a), log(b)
        return np.exp(x[0]) * np.exp(-np.exp(x[1]) * t)

    def ne(x):
        f = model(x)
        r = f - y
        J = np.column_stack([f, -np.exp(x[1]) * t * f])
        return J.T @ J, J.T @ r, r @ r

    x0 = np.log([1.0, 2.0])
    out = lm_minimize(ne, lambda x: np.sum((model(x) - y) ** 2), x0, x0 - 5, x0 + 5,
                      names=["a", "b"], max_iters=100, ftol=1e-15, xtol=1e-12, log=_quiet)
    p, pcov = curve_fit(lambda tt, aa, bb: aa * np.exp(-bb * tt), t, y, p0=[1, 2])
    assert np.allclose(np.exp(out["x"]), p, rtol=1e-6), (np.exp(out["x"]), p)
    tab, corr, info = covariance_report(out["x"], out["A"], out["loss"], n=t.size, n_eff=t.size, names=["a", "b"])
    rel_se_scipy = np.sqrt(np.diag(pcov)) / p
    assert np.allclose(tab["rel_se"], rel_se_scipy, rtol=1e-3), (tab["rel_se"].values, rel_se_scipy)
    # conservative version with n_eff = n/10 inflates SE by sqrt(10)
    tab2, _, _ = covariance_report(out["x"], out["A"], out["loss"], n=t.size, n_eff=t.size // 10, names=["a", "b"])
    assert np.allclose(tab2["rel_se_conservative"] / tab["rel_se"], np.sqrt(10), rtol=1e-6)
    assert abs(corr.loc["a", "b"]) < 1


def test_bounds_and_max_step():
    """Optimum outside the box: LM must stop at the bound; steps never exceed max_step."""
    def ne(x):
        r = x - 3.0
        return np.eye(1), r, r @ r

    steps = []
    out = lm_minimize(ne, lambda x: float(np.sum((x - 3.0) ** 2)), np.array([0.0]), [-1.0], [1.0],
                      max_iters=50, max_step=0.25, log=lambda s: steps.append(s))
    assert np.isclose(out["x"][0], 1.0)
    h = out["history"]
    assert (h["max_step"] <= 0.25 + 1e-12).all()


def test_active_bound_is_held():
    """x0 at its upper bound with the gradient pushing outward: lm_step must keep
    it there and still optimise the free parameter (minimum of x1 at 2)."""
    from lm_optimizer import lm_step
    x = np.array([1.0, 0.0])
    A = np.eye(2); g = np.array([-1.0, -2.0])          # r = (x0 - 2, x1 - 2) at x = (1, 0)
    delta, pred = lm_step(x, A, g, 1e-6, lo=[-1, -5], hi=[1, 5], max_step=10)
    assert delta[0] == 0.0 and np.isclose(delta[1], 2.0, rtol=1e-5) and pred > 0


def test_stall_stop_with_refresh():
    """Accepted but tiny steps (~1 %) with a refresh after every step (so the
    loss test is skipped): LM must stop after 2 such steps, not run all iters."""
    calls = {"n": 0}

    def ne(x):
        return np.eye(1) * 1e6, np.array([1e4]), 1.0     # GN step ~ -0.01

    def loss(x):
        calls["n"] += 1
        return 1.0 - 1e-3 * calls["n"]                    # always a little lower -> accepted

    out = lm_minimize(ne, loss, np.zeros(1), [-5], [5], max_iters=20, refresh=lambda x: None,
                      refresh_every=1, ftol=1e-4, xtol=1e-4, log=_quiet)
    assert "consecutive steps" in out["reason"], out["reason"]
    assert len(out["history"]) == 3   # iteration 0 + 2 stalled steps


def test_noisy_loss_stops():
    """A loss that never decreases (noise floor) ends with 'no decrease' instead of looping."""
    def ne(x):
        return np.eye(2), np.array([1.0, -1.0]), 1.0

    out = lm_minimize(ne, lambda x: 1.0 + 1e-3, np.zeros(2), [-5, -5], [5, 5], max_trials=3, log=_quiet)
    assert "no decrease" in out["reason"]
    assert np.allclose(out["x"], 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
