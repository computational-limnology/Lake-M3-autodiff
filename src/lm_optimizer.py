"""
Levenberg-Marquardt (LM) for small least-squares calibrations (a handful of
parameters), used by calibrate_M3_jax.py --optimizer lm. Plain numpy: the
model-specific parts (Jacobian, loss, start-state refresh) come in as
callbacks, so the algorithm can be tested on synthetic problems
(tests/test_lm.py).

Problem: minimise L(x) = sum r(x)^2 over x (log-parameters) within box bounds.
Each iteration needs the normal equations at x,

    A = J^T J,   g = J^T r,   L = r^T r          (J = dr/dx)

and proposes

    delta = -(A + lam * D) ^-1 g,   D = diag(A) (Marquardt scaling, floored)

clipped to the bounds and to a maximum step (`max_step`, in log units; a
trust-region cap that keeps steps along flat, poorly identified directions
from running away). The step is accepted if the true loss decreases; lam is
then updated with Nielsen's rule from rho = actual / predicted reduction
(predicted from the local quadratic model ||r + J delta||^2). A rejected step
raises lam (x nu, nu doubles) and is retried with the same A, g.

Uncertainty at the solution: Cov(x) = s^2 (J^T J)^-1, s^2 = L / (n - P). The
residuals here are profiles (correlated over depth), so a conservative version
treats every profile date as one independent value: Cov_cons = Cov * n / n_eff.
x is log(parameter), so sqrt(Cov_ii) is a relative standard error.
"""
import numpy as np
import pandas as pd


def lm_step(x, A, g, lam, lo, hi, max_step=0.5, d_floor=1e-12):
    """Damped Gauss-Newton step from x. Returns (delta, predicted_reduction).
    `lo`/`hi` are bounds on x, `max_step` caps max|delta| (log units)."""
    A = np.asarray(A, float); g = np.asarray(g, float); x = np.asarray(x, float)
    d = np.diag(A).copy()
    d = np.maximum(d, d_floor * max(d.max(), 1e-300))
    M = A + lam * np.diag(d)
    try:
        delta = -np.linalg.solve(M, g)
    except np.linalg.LinAlgError:
        delta = -np.linalg.lstsq(M, g, rcond=None)[0]
    m = np.max(np.abs(delta))
    if max_step and m > max_step:
        delta = delta * (max_step / m)
    delta = np.clip(x + delta, lo, hi) - x
    pred = -(2.0 * g @ delta + delta @ A @ delta)   # L(x) - model(x + delta)
    return delta, float(pred)


def nielsen_update(lam, nu, rho, accepted, lam_min=1e-12):
    if accepted:
        return max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), lam_min), 2.0
    return lam * nu, nu * 2.0


def lm_minimize(normal_eq, loss, x0, lo, hi, names=None, max_iters=20, lam0=1e-2, max_trials=4,
                max_step=0.5, ftol=1e-4, xtol=1e-3, lam_max=1e8, refresh=None, refresh_every=0,
                validate=None, log=print):
    """Run LM from x0 (1-D array).

    normal_eq(x) -> (A, g, L)   L = sum r^2 at x (same start states as `loss`)
    loss(x)      -> L           forward-only loss
    refresh(x)   optional; called after every `refresh_every` accepted steps
                 (windowed runs: new start states). The loss is re-scored after.
    validate(x)  optional; returns a held-out loss to print (not used for decisions)

    Returns dict(x, loss, A, g, lam, history (DataFrame), reason)."""
    names = list(names) if names is not None else [f"x{i}" for i in range(len(x0))]
    x = np.asarray(x0, float).copy()
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    lam, nu = float(lam0), 2.0
    rows, reason = [], f"reached --iters {max_iters}"
    A, g, L = normal_eq(x)
    A, g, L = np.asarray(A, float), np.asarray(g, float), float(L)
    v = validate(x) if validate else None
    rows.append(dict(iter=0, loss=L, test_loss=v, lam=lam, rho=np.nan, trials=0, max_step=0.0,
                     **{n: float(np.exp(xi)) for n, xi in zip(names, x)}))
    log(f"  LM iter   0: loss={L:.6g}" + (f"  test={v:.6g}" if v is not None else "")
        + "  grad=" + " ".join(f"{n}:{2 * gi:+.3g}" for n, gi in zip(names, g)))
    n_acc = 0
    for it in range(1, max_iters + 1):
        accepted = False
        for trial in range(1, max_trials + 1):
            delta, pred = lm_step(x, A, g, lam, lo, hi, max_step)
            if not np.any(delta):
                reason = "step is zero (all parameters at their bounds?)"
                break
            L_new = float(loss(x + delta))
            rho = (L - L_new) / pred if pred > 0 else -np.inf
            accepted = np.isfinite(L_new) and L_new < L
            log(f"     trial {trial}: lam={lam:.3g}  max|step|={np.max(np.abs(delta)):.3g}  "
                f"loss={L_new:.6g}  rho={rho:.3g}  {'accepted' if accepted else 'rejected'}")
            lam, nu = nielsen_update(lam, nu, rho, accepted)
            if accepted:
                break
            if lam > lam_max:
                break
        if not accepted:
            if reason.startswith("reached"):
                reason = (f"no decrease after {max_trials} trial step(s) (lam={lam:.3g}) -- "
                          "converged to the loss noise level")
            break
        x_old, L_old = x, L
        x = x + delta
        n_acc += 1
        if refresh is not None and refresh_every and n_acc % refresh_every == 0:
            refresh(x)
        A, g, L = normal_eq(x)
        A, g, L = np.asarray(A, float), np.asarray(g, float), float(L)
        v = validate(x) if validate else None
        step = float(np.max(np.abs(x - x_old)))
        rows.append(dict(iter=it, loss=L, test_loss=v, lam=lam, rho=rho, trials=trial, max_step=step,
                         **{n: float(np.exp(xi)) for n, xi in zip(names, x)}))
        log(f"  LM iter {it:3d}: loss={L:.6g}" + (f"  test={v:.6g}" if v is not None else "")
            + f"  max|dlog p|={step:.3g}  " + " ".join(f"{n}={np.exp(xi):.4g}" for n, xi in zip(names, x)))
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(g)):
            reason = "non-finite Jacobian -- stopped (kept the last finite point)"
            x, L = x_old, L_old
            break
        if abs(L_old - L) <= ftol * max(abs(L_old), 1e-300) and not (refresh_every and n_acc % refresh_every == 0):
            reason = f"relative loss change < {ftol:g}"
            break
        if step < xtol:
            reason = f"max parameter step < {xtol:g} (log units, ~{100 * xtol:.2g}%)"
            break
    return dict(x=x, loss=L, A=A, g=g, lam=lam, history=pd.DataFrame(rows), reason=reason)


def covariance_report(x, A, L, n, n_eff, names):
    """Parameter uncertainty from J^T J at the solution (x = log parameters).
    Returns (table DataFrame, correlation DataFrame, info dict)."""
    A = np.asarray(A, float); x = np.asarray(x, float)
    P = len(x)
    s2 = L / max(n - P, 1)
    Ainv = np.linalg.pinv(A)
    cov = s2 * Ainv
    cov_c = cov * n / max(n_eff, 1)
    se = np.sqrt(np.maximum(np.diag(cov), 0)); se_c = np.sqrt(np.maximum(np.diag(cov_c), 0))
    sd = np.where(se > 0, se, 1.0)
    corr = cov / np.outer(sd, sd)
    ev = np.linalg.eigvalsh(A)
    rows = []
    for i, nme in enumerate(names):
        p = float(np.exp(x[i]))
        rows.append(dict(parameter=nme, value=p, rel_se=se[i], rel_se_conservative=se_c[i],
                         lo95_conservative=p * np.exp(-1.96 * se_c[i]),
                         hi95_conservative=p * np.exp(1.96 * se_c[i]),
                         identifiable=bool(se_c[i] < 1.0)))
    info = dict(n=int(n), n_eff=int(n_eff), s2=s2, cond=float(ev[-1] / max(ev[0], 1e-300)),
                eig_min=float(ev[0]), eig_max=float(ev[-1]))
    return pd.DataFrame(rows), pd.DataFrame(corr, index=names, columns=names), info
