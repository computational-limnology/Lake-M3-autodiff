"""
Windowed simulation: run the lake model as many short, independent windows
(e.g. one per profile year, 1 Feb - 31 Dec) side by side instead of one long
continuous run. Used by windowed calibration / MCL training (Step 2 of the
calibration plan); this module is the infrastructure (Phase 1).

Pieces
  build_setup(lake_dir)            forcing, geometry, params, initial state, observations
  select_windows(...)              which windows: start MM-DD, length, spin-up, years
  reference_states(...)            start state of every window, taken from ONE continuous
                                   forward run with given parameters (daily states, cached)
  window_forcing / window_obs      forcing slices and observations per window (local step
                                   index, only scored after the spin-up, padded + masked)
  make_window_simulator(...)       one window: state0 + forcing -> per-step outputs, with the
                                   same truncated BPTT as calibrate_M3_jax.simulate_truncated
  run_batched(...)                 runs a per-window function over all windows: spread over
                                   CPU devices (pmap) and batched within each (vmap); padding
                                   windows are added so the count divides the device count

Start states: a window starting at hour i0 of the record takes the model state
after i0 continuous steps. With unchanged parameters the windows therefore
reproduce the continuous run; after parameters change the saved states go
stale and are refreshed by another reference run (Phase 2: --refresh-every).

Devices: JAX must be told the CPU device count before it starts. Call
`configure_devices(n, single_thread)` before importing jax (the CLI below does
this from --devices / --single-thread).

Consistency test (Phase 1 check):
    python src/windows.py Ravn --test [--record-end 2006-12-31] [--n-windows 3]
        [--window-start 02-01 --window-steps 8016] [--devices 4 --single-thread]
Runs the reference forward run, then each window on its own (must match bit for
bit), all windows batched, and a control window with a 1e-15 perturbation;
see _consistency_test for how to read the result.
"""
import os
import sys
import time
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def configure_devices(n_devices=1, single_thread=False):
    """Set XLA flags for `n_devices` CPU devices (and optionally one compute
    thread per device). Must run before `import jax` anywhere in the process."""
    flags = os.environ.get("XLA_FLAGS", "")
    if n_devices and n_devices > 1 and "xla_force_host_platform_device_count" not in flags:
        flags += f" --xla_force_host_platform_device_count={n_devices}"
    if single_thread and "xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = flags.strip()


if __name__ == "__main__":   # device flags must be set before JAX is imported
    _argv = sys.argv
    _dev = int(_argv[_argv.index("--devices") + 1]) if "--devices" in _argv else 1
    configure_devices(_dev, "--single-thread" in _argv)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from processBased_lakeModel_functions import (  # noqa: E402
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile, provide_phosphorus,
    provide_carbon, get_lake_config, get_model_params, get_run_config, get_ice_and_snow,
)
from jax_lakeModel_functions import (  # noqa: E402
    make_initial_state_full, full_step, default_params, default_geometry_wq, LakeState,
)
from run_M3_jax import build_forcing_series, add_wq_forcing_series  # noqa: E402

STATE_KEYS = None  # LakeState key order, set by build_setup (pytree structure must match in scans)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def build_setup(lake_dir, record_end=None, temp_obs="profiles", load_obs=True):
    """Everything a windowed run needs, read from a lake folder the same way
    calibrate_M3_jax.py does. `record_end` (date) shortens the record."""
    global STATE_KEYS
    lake_dir = os.path.abspath(lake_dir)
    cwd = os.getcwd()
    os.chdir(lake_dir)
    try:
        lc = get_lake_config("./lake_config.csv", 1); mp = get_model_params("./model_params.csv", 1)
        rc = get_run_config("./run_config.csv", 1); ias = get_ice_and_snow("./ice_and_snow.csv", 1)
        wf = float(lc["WindSpeed"]); nx = int(rc["nx"]); dt = float(rc["dt"]); dx = float(rc["dx"])
        area, depth, volume, hw = get_hypsography(hypsofile=rc["hypso_ini_file"], dx=dx, nx=nx,
                                                  outflow_depth=float(lc["outflow_depth"]))
        start = pd.Timestamp(rc["start_time"])
        end = pd.Timestamp(record_end) if record_end else pd.Timestamp(rc["end_time"])
        n = len(pd.date_range(start, end, freq="h"))
        step_times = np.arange(1, (n + 1) * dt, dt)[:n]
        met = provide_meteorology(meteofile=rc["meteo_ini_file"], windfactor=wf, lat=lc["Latitude"],
                                  lon=lc["Longitude"], elev=lc["Elevation"], startDate=start)
        u0 = initial_profile(initfile=rc["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=start)
        wq = wq_initial_profile(initfile=rc["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=start)
        tp = provide_phosphorus(tpfile=rc["tp_ini_file"], startingDate=start, startTime=1)
        cd = provide_carbon(ocloadfile=rc["oc_load_file"], startingDate=start, startTime=1).dropna(subset=["oc"])
        forcing = add_wq_forcing_series(build_forcing_series(met, step_times, wf), tp, cd, step_times)
        geom = default_geometry_wq(area=area, depth=depth, volume=volume, dx=dx, dt=dt,
                                   latitude=float(lc["Latitude"]), altitude=float(lc["Elevation"]),
                                   hypso_weight=hw, mean_depth=float(np.sum(volume) / np.max(area)),
                                   hydro_res_time_hr=float(mp["hydro_res_time"]) * 8760)
        params = default_params(mp, ias, diffusion_method=str(rc.get("diffusion_method", "hendersonSellers")))
        tb = lambda x: x.strip().lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)
        s0 = make_initial_state_full(
            jnp.asarray(deepcopy(u0)), jnp.asarray(wq[0]), jnp.asarray(wq[1] * 0.75), jnp.asarray(wq[1] * 0.25),
            jnp.asarray(0.5 * volume), jnp.asarray(0.5 * volume), nx,
            ice=tb(ias["ice"]), Hi=ias["Hi"], Hs=ias["Hs"], Hsi=ias["Hsi"], iceT=ias["iceT"],
            rho_snow=ias["rho_snow"])
        obs = None
        if load_obs:
            from calibrate_M3_jax import load_observations
            obs = load_observations("./", depth, volume, start, step_times, dt, run_config=rc, temp_obs=temp_obs)
    finally:
        os.chdir(cwd)
    STATE_KEYS = list(s0.keys())
    return dict(lake_dir=lake_dir, s0=s0, forcing={k: np.asarray(v) for k, v in forcing.items()},
                geom=geom, params=params, start=start, dt=dt, n_steps=n, step_times=step_times,
                depth=np.asarray(depth), volume=np.asarray(volume), obs=obs, run_config=rc)


def as_state(d):
    """dict of arrays -> LakeState in the model's key order."""
    keys = STATE_KEYS or list(d.keys())
    return LakeState({k: d[k] for k in keys})


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def select_windows(setup, window_start="02-01", window_steps=8016, spinup_days=28, years=None,
                   require_obs="temp"):
    """One window per year starting at 00:00 on `window_start` (MM-DD), all of
    the same length `window_steps` (8016 h = 1 Feb - 31 Dec in non-leap years;
    in leap years the window ends at 30 Dec 23:00). Observations in the first
    `spinup_days` are not scored. Years are those whose window fits in the
    record and, if `require_obs` is set, that have at least one scored
    observation of that variable. Returns a DataFrame (year, i0, i_score, i1)
    with global step indices (i1 exclusive)."""
    start, n = setup["start"], setup["n_steps"]
    rows = []
    for y in range(start.year, (start + pd.Timedelta(hours=n)).year + 1):
        if years is not None and y not in years:
            continue
        t0 = pd.Timestamp(f"{y}-{window_start}")
        i0 = int((t0 - start) / pd.Timedelta(hours=1))
        if i0 < 0 or i0 % 24 or i0 + window_steps > n:
            continue
        i_score = i0 + 24 * spinup_days
        rows.append(dict(year=y, start=t0, i0=i0, i_score=i_score, i1=i0 + window_steps))
    win = pd.DataFrame(rows)
    if require_obs and setup.get("obs") and setup["obs"].get(require_obs) is not None and len(win):
        idx = np.asarray(setup["obs"][require_obs]["step_idx"])
        has = [bool(np.any((idx >= r.i_score) & (idx < r.i1))) for r in win.itertuples()]
        win = win[has].reset_index(drop=True)
    return win


def window_forcing(forcing, windows, window_steps):
    """Batched forcing: dict of arrays (n_windows, window_steps, ...)."""
    return {k: np.stack([v[i0:i0 + window_steps] for i0 in windows["i0"]]) for k, v in forcing.items()}


def window_obs(obs_var, windows):
    """Observations of one variable (calibrate_M3_jax.load_observations format:
    step_idx, values, mask) split by window: local step index relative to the
    window start, only those at or after the spin-up, padded to a common count
    with mask False. Returns dict(step_idx (B,K), values (B,K,...), mask (B,K,...),
    valid (B,K)) or None."""
    if obs_var is None:
        return None
    idx = np.asarray(obs_var["step_idx"]); vals = np.asarray(obs_var["values"]); mask = np.asarray(obs_var["mask"])
    per = []
    for r in windows.itertuples():
        sel = np.where((idx >= r.i_score) & (idx < r.i1))[0]
        per.append(sel)
    K = max(1, max(len(s) for s in per))
    B = len(per)
    out_idx = np.zeros((B, K), int)
    out_val = np.zeros((B, K) + vals.shape[1:])
    out_mask = np.zeros((B, K) + mask.shape[1:], bool)
    valid = np.zeros((B, K), bool)
    for b, (sel, r) in enumerate(zip(per, windows.itertuples())):
        k = len(sel)
        out_idx[b, :k] = idx[sel] - r.i0
        out_val[b, :k] = vals[sel]
        out_mask[b, :k] = mask[sel]
        valid[b, :k] = True
    return dict(step_idx=out_idx, values=out_val, mask=out_mask, valid=valid)


# ---------------------------------------------------------------------------
# reference run -> start states
# ---------------------------------------------------------------------------

def _apply_theta(params, theta):
    p = dict(params)
    if theta:
        p.update({k: jnp.exp(v) for k, v in theta.items()})
    return p


def reference_states(setup, theta=None, cache=None, days_per_call=365):
    """Daily states (model state at every midnight) of one continuous forward
    run with `theta` (log-parameters applied over setup["params"]). Returns a
    dict of arrays with a leading day axis: day d = state after 24*d steps
    (day 0 = initial state). Run in yearly jitted pieces so memory stays small.
    If `cache` (path .npz) is given and holds states for the same theta and
    record length, they are loaded instead."""
    key = _theta_key(theta, setup["n_steps"])
    if cache and os.path.exists(cache):
        with np.load(cache) as d:
            if str(d["_key"]) == key:
                return {k: np.asarray(d[k]) for k in d.files if not k.startswith("_")}
    geom, params = setup["geom"], setup["params"]
    p = _apply_theta(params, theta)
    n_days = setup["n_steps"] // 24
    F = setup["forcing"]

    @jax.jit
    def run_days(s, f_days):
        def day(s, fd):
            def step(s, ft):
                ns, _ = full_step(s, ft, geom, p)
                return ns, None
            s, _ = lax.scan(step, s, fd)
            return s, s
        return lax.scan(day, s, f_days)

    s = setup["s0"]
    out = {k: [np.asarray(v)[None]] for k, v in s.items()}
    d = 0
    while d < n_days:
        nd = min(days_per_call, n_days - d)
        f = {k: jnp.asarray(v[24 * d:24 * (d + nd)].reshape((nd, 24) + v.shape[1:])) for k, v in F.items()}
        s, ss = run_days(s, f)   # recompiles only for the last, shorter piece
        for k in out:
            out[k].append(np.asarray(ss[k]))
        d += nd
    states = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    if cache:
        np.savez(cache, _key=np.array(key), **states)
    return states


def _theta_key(theta, n_steps):
    if not theta:
        return f"base|{n_steps}"
    return "|".join(f"{k}={float(v):.12g}" for k, v in sorted(theta.items())) + f"|{n_steps}"


def window_start_states(daily_states, windows):
    """Batched start state (LakeState of arrays with leading window axis)."""
    days = (windows["i0"].to_numpy() // 24).astype(int)
    return as_state({k: v[days] for k, v in daily_states.items()})


# ---------------------------------------------------------------------------
# window simulator + batched runner
# ---------------------------------------------------------------------------

def make_window_simulator(geom, params, chunk_steps=2000, outputs=("u",), daily_states=False):
    """sim(theta, state0, forcing) for ONE window -> dict of per-step outputs
    (n_steps, ...). Truncated BPTT exactly like calibrate_M3_jax: stop_gradient
    on the state every `chunk_steps`, jax.checkpoint per step and per chunk.
    If the window length isn't a multiple of chunk_steps, the largest divisor
    below it is used (`fit_chunk`; 8016 h -> 1336 h chunks). With daily_states=True the state at
    every midnight is returned instead (key 'state', for consistency tests)."""
    def sim(theta, state0, forcing):
        p = _apply_theta(params, theta)

        @jax.checkpoint
        def step(s, f):
            ns, diag = full_step(s, f, geom, p)
            if daily_states:
                return ns, None
            out = {k: (diag[k] if k in diag else ns[k]) for k in outputs}
            return ns, out

        n = forcing["Uw"].shape[0]
        if daily_states:
            fd = jax.tree_util.tree_map(lambda v: v.reshape((n // 24, 24) + v.shape[1:]), forcing)

            def day(s, f):
                s, _ = lax.scan(step, s, f)
                return s, s
            _, ss = lax.scan(day, state0, fd)
            return {"state": ss}

        @jax.checkpoint
        def chunk(s, fc):
            s, o = lax.scan(step, s, fc)
            return jax.tree_util.tree_map(lax.stop_gradient, s), o

        c = fit_chunk(n, chunk_steps)   # e.g. 8016 h -> 6 chunks of 1336 h
        fch = jax.tree_util.tree_map(lambda v: v.reshape((n // c, c) + v.shape[1:]), forcing)
        _, o = lax.scan(chunk, state0, fch)
        return jax.tree_util.tree_map(lambda v: v.reshape((n,) + v.shape[2:]), o)
    return sim


def fit_chunk(window_steps, target=2000):
    """Largest divisor of window_steps that is <= target (truncation length)."""
    for c in range(min(target, window_steps), 0, -1):
        if window_steps % c == 0:
            return c
    return window_steps


def run_batched(fn, theta, batched_args, n_devices=None, reduce_sum=False):
    """Apply `fn(theta, *args_for_one_window)` to every window. `batched_args`
    is a tuple of pytrees with a leading window axis B. With D > 1 devices the
    windows are padded (by repeating window 0) to a multiple of D and run as
    pmap over devices x vmap within a device; otherwise plain vmap.
    reduce_sum=True sums a scalar result over the real windows (for losses;
    padding windows are excluded by a weight of 0) -- use with jax.grad.
    Returns the per-window results (B, ...) or the sum."""
    D = n_devices or jax.device_count()
    B = jax.tree_util.tree_leaves(batched_args)[0].shape[0]
    pad = (-B) % D if D > 1 else 0
    w = jnp.concatenate([jnp.ones(B), jnp.zeros(pad)])

    def padb(x):
        x = jnp.asarray(x)
        return jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])]) if pad else x

    args = jax.tree_util.tree_map(padb, batched_args)
    vf = jax.vmap(lambda *a: fn(theta, *a))
    if D > 1:
        Bp = B + pad
        split = lambda x: x.reshape((D, Bp // D) + x.shape[1:])
        args_d = jax.tree_util.tree_map(split, args)
        w_d = split(w)
        if reduce_sum:
            per_dev = jax.pmap(lambda wd, *a: jnp.sum(wd * jax.vmap(lambda *aa: fn(theta, *aa))(*a)))(w_d, *args_d)
            return jnp.sum(per_dev)
        res = jax.pmap(lambda *a: jax.vmap(lambda *aa: fn(theta, *aa))(*a))(*args_d)
        return jax.tree_util.tree_map(lambda x: x.reshape((Bp,) + x.shape[2:])[:B], res)
    if reduce_sum:
        return jnp.sum(w * vf(*args))
    res = vf(*args)
    return jax.tree_util.tree_map(lambda x: x[:B], res)


# ---------------------------------------------------------------------------
# consistency test (CLI)
# ---------------------------------------------------------------------------

def _consistency_test(a):
    """Three comparisons against the continuous reference run, at every midnight:
    1. each window run on its own (no batching): must match bit for bit
       (this checks start states, forcing slices and step alignment);
    2. all windows batched (vmap, and pmap if --devices > 1): batching changes
       the order of floating-point operations, so results differ at round-off
       level and that difference then grows with the model's own sensitivity;
    3. control: a single window with a round-off-sized perturbation that acts
       every step, like batching does (start temperature and the drag
       coefficient Cd both changed by one part in 1e15). A perturbation of the
       start state alone is not enough: in a well-mixed or ice-covered winter
       it is often wiped out within hours, while batching keeps adding
       round-off differences all window long.
    The model amplifies round-off in threshold events (mixed-layer deepening,
    convection, ice), so (2) and (3) are of similar size but differ window by
    window; batched runs are fine if, over all windows, their differences are
    no larger than the control's (median and maximum within a factor 2)."""
    t0 = time.time()
    S = build_setup(a.lake_dir, record_end=a.record_end, temp_obs="profiles")
    print(f"setup {time.time() - t0:.0f}s: record {S['start'].date()} + {S['n_steps']} h, "
          f"{jax.device_count()} device(s), single-thread={a.single_thread}")
    win = select_windows(S, a.window_start, a.window_steps, a.spinup_days)
    if a.n_windows:
        win = win.head(a.n_windows)
    if not len(win):
        raise SystemExit("no windows fit the record (try a later --record-end or shorter --window-steps)")
    wo = window_obs(S["obs"].get("temp"), win) if S["obs"] else None
    print("windows:", ", ".join(f"{r.year} ({int(wo['valid'][i].sum()) if wo else 0} profiles after spin-up)"
                               for i, r in enumerate(win.itertuples())))
    t = time.time()
    daily = reference_states(S, None)
    print(f"reference run: {len(daily['u'])} daily states in {time.time() - t:.0f}s")
    W = a.window_steps - a.window_steps % 24
    F = window_forcing(S["forcing"], win, W)
    st = window_start_states(daily, win)
    sim = make_window_simulator(S["geom"], S["params"], daily_states=True)
    one = jax.jit(lambda s, f: sim(None, s, f)["state"])
    p_ctl = dict(S["params"]); p_ctl["Cd"] = float(p_ctl["Cd"]) * (1 + 1e-15)
    sim_ctl = make_window_simulator(S["geom"], p_ctl, daily_states=True)
    one_ctl = jax.jit(lambda s, f: sim_ctl(None, s, f)["state"])
    t = time.time()
    res_b = run_batched(lambda th, s, f: sim(th, s, f)["state"], None, (st, F))
    jax.block_until_ready(res_b)
    print(f"batched run ({len(win)} windows x {W} h): {time.time() - t:.0f}s\n")

    def udiff(got, ref):
        d = np.max(np.abs(np.asarray(got) - ref).reshape(len(ref), -1), axis=1)
        first = int(np.argmax(d > 1e-10)) + 1 if np.any(d > 1e-10) else None
        return float(d.max()), first, d

    days0 = (win["i0"].to_numpy() // 24).astype(int)
    ok_single = True
    mbs, mcs = [], []
    print(f"{'window':>6} | {'single: max dT':>14} | {'batched: max dT  first day>1e-10':>34} | "
          f"{'control (1e-15 on u0 and Cd): max dT  first day':>48}")
    for b, r in enumerate(win.itertuples()):
        dd = days0[b] + 1 + np.arange(W // 24)
        ref = daily["u"][dd]
        sb = as_state({k: jnp.asarray(v[b]) for k, v in st.items()})
        fb = {k: jnp.asarray(v[b]) for k, v in F.items()}
        r1 = one(sb, fb)
        m1, f1, _ = udiff(r1["u"], ref)
        sp = dict(sb); sp["u"] = sb["u"] * (1 + 1e-15)
        rc = one_ctl(as_state(sp), fb)
        mc, fc, dc = udiff(rc["u"], ref)
        mb, fbd, db = udiff(res_b["u"][b], ref)
        rel = {k: float(np.max(np.abs(np.asarray(r1[k], float) - daily[k][dd])
                                / (np.abs(daily[k][dd]) + 1e-12))) for k in r1 if k != "ice"}
        worst_k = max(rel, key=rel.get)
        others = rel[worst_k]
        ok_single &= (m1 <= 1e-12 and others <= 1e-12)
        mbs.append(mb); mcs.append(mc)
        print(f"{r.year:>6} | {m1:14.1e} | {mb:16.2e}  {str(fbd):>16} | {mc:34.2e}  {str(fc):>12}"
              + ("" if others == 0.0 else f"   (other fields: max rel. diff {others:.1e} in {worst_k})"))
    print(f"\nsingle windows {'match the continuous run to round-off (<= 1e-12) (PASS)' if ok_single else 'DO NOT match (FAIL)'}")
    mbs, mcs = np.array(mbs), np.array(mcs)
    ok_batched = (np.median(mbs) <= 2 * max(np.median(mcs), 1e-10)) and (mbs.max() <= 2 * max(mcs.max(), 1e-10))
    print(f"batched vs control over {len(mbs)} windows: median max-dT {np.median(mbs):.3g} vs {np.median(mcs):.3g} degC, "
          f"largest {mbs.max():.3g} vs {mcs.max():.3g} degC")
    print(f"batched windows {'are within the round-off sensitivity of the model (PASS)' if ok_batched else 'differ more than round-off explains (CHECK)'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--test", action="store_true", help="run the window-vs-continuous consistency test")
    ap.add_argument("--record-end", default=None, help="shorten the record (date), e.g. for a quick test")
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--window-start", default="02-01")
    ap.add_argument("--window-steps", type=int, default=8016)
    ap.add_argument("--spinup-days", type=int, default=28)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--single-thread", action="store_true")
    a = ap.parse_args()
    if a.test:
        _consistency_test(a)
    else:
        ap.print_help()
