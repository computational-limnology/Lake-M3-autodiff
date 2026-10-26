"""
Stage 0 benchmark for windowed calibration: how does the cost of running B
independent simulation windows (forward, and forward + gradient) scale with
B, and which way of running them in parallel is fastest on this machine?

Modes
  seq   -- one jitted single-window function, called B times in a loop
  vmap  -- one jitted function over a batch of B windows (jax.vmap)
  pmap  -- B windows spread over D CPU "devices" (jax.pmap), vmapped within
           each device; needs --devices D (sets XLA's host device count
           before JAX starts, so run it as its own process)

Each window starts from the run's initial state with the forcing of a
different year (start date --window-start of successive years). The window
simulator is the one windowed calibration will use: truncated BPTT with a
stop_gradient every --chunk-steps, jax.checkpoint per step and per chunk
(same as calibrate_M3_jax.simulate_truncated). The gradient is taken w.r.t.
the 8 parameters of the latest profiles-only calibration.

Usage (from the project root):
    python src/benchmark_windows.py Ravn --batches 1,4,8,16,31 --modes seq,vmap
    python src/benchmark_windows.py Ravn --batches 4,8,16,32 --modes pmap --devices 4
    [--window-steps 8016 --chunk-steps 2000 --repeats 2 --what fwd,grad --out bench.csv]

Prints, per mode and batch size: compile time, run time (best of --repeats),
and ms per window-step; appends the rows to --out (CSV).
"""
import argparse
import os
import sys
import time


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lake_dir")
    ap.add_argument("--batches", default="1,4,8,16,31")
    ap.add_argument("--modes", default="seq,vmap")
    ap.add_argument("--what", default="fwd,grad")
    ap.add_argument("--window-steps", type=int, default=8016, help="hourly steps per window (1 Feb - 31 Dec ~ 8016)")
    ap.add_argument("--chunk-steps", type=int, default=2000)
    ap.add_argument("--window-start", default="02-01", help="MM-DD of each window's start")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--devices", type=int, default=1, help="CPU devices for --modes pmap")
    ap.add_argument("--single-thread", action="store_true",
                    help="one compute thread per device (XLA: no multi-threaded Eigen), so --devices "
                         "equal to the core count doesn't oversubscribe the CPU")
    ap.add_argument("--out", default=None, help="CSV to append results to")
    return ap.parse_args()


args = parse()
if args.devices > 1:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={args.devices}")
if args.single_thread:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from copy import deepcopy  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processBased_lakeModel_functions import (  # noqa: E402
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile, provide_phosphorus,
    provide_carbon, get_lake_config, get_model_params, get_run_config, get_ice_and_snow,
)
from jax_lakeModel_functions import (  # noqa: E402
    make_initial_state_full, full_step, default_params, default_geometry_wq, LakeState,
)
from run_M3_jax import build_forcing_series, add_wq_forcing_series  # noqa: E402

PARAMS = ["Cd", "sw_factor", "at_factor", "wind_factor", "k_max", "alpha", "seiche_alpha", "sbl_drho"]


def setup(lake_dir):
    cwd = os.getcwd()
    os.chdir(lake_dir)
    try:
        lc = get_lake_config("./lake_config.csv", 1); mp = get_model_params("./model_params.csv", 1)
        rc = get_run_config("./run_config.csv", 1); ias = get_ice_and_snow("./ice_and_snow.csv", 1)
        wf = float(lc["WindSpeed"]); nx = int(rc["nx"]); dt = float(rc["dt"]); dx = float(rc["dx"])
        area, depth, volume, hw = get_hypsography(hypsofile=rc["hypso_ini_file"], dx=dx, nx=nx,
                                                  outflow_depth=float(lc["outflow_depth"]))
        start, end = pd.Timestamp(rc["start_time"]), pd.Timestamp(rc["end_time"])
        n = len(pd.date_range(start, end, freq="h"))
        st = np.arange(1, (n + 1) * dt, dt)[:n]
        met = provide_meteorology(meteofile=rc["meteo_ini_file"], windfactor=wf, lat=lc["Latitude"],
                                  lon=lc["Longitude"], elev=lc["Elevation"], startDate=start)
        u0 = initial_profile(initfile=rc["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=start)
        wq = wq_initial_profile(initfile=rc["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=start)
        tp = provide_phosphorus(tpfile=rc["tp_ini_file"], startingDate=start, startTime=1)
        cd = provide_carbon(ocloadfile=rc["oc_load_file"], startingDate=start, startTime=1).dropna(subset=["oc"])
        forcing = add_wq_forcing_series(build_forcing_series(met, st, wf), tp, cd, st)
        geom = default_geometry_wq(area=area, depth=depth, volume=volume, dx=dx, dt=dt,
                                   latitude=float(lc["Latitude"]), altitude=float(lc["Elevation"]),
                                   hypso_weight=hw, mean_depth=float(np.sum(volume) / np.max(area)),
                                   hydro_res_time_hr=float(mp["hydro_res_time"]) * 8760)
        params = default_params(mp, ias, diffusion_method=str(rc.get("diffusion_method", "hendersonSellers")))
        tb = lambda x: x.strip().lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)
        s0 = make_initial_state_full(
            jnp.asarray(deepcopy(u0)), jnp.asarray(wq[0]), jnp.asarray(wq[1] * 0.75), jnp.asarray(wq[1] * 0.25),
            jnp.asarray(0.5 * volume), jnp.asarray(0.5 * volume), nx,
            ice=tb(ias["ice"]), Hi=ias["Hi"], Hs=ias["Hs"], Hsi=ias["Hsi"], iceT=ias["iceT"], rho_snow=ias["rho_snow"])
    finally:
        os.chdir(cwd)
    return s0, {k: np.asarray(v) for k, v in forcing.items()}, geom, params, start


def make_window_sim(geom, params, chunk_steps):
    """Single window: state0 + forcing (W steps) -> per-step surface..bottom temperature.
    Truncated BPTT exactly as calibrate_M3_jax.simulate_truncated (W must be a
    multiple of chunk_steps here)."""
    def sim(theta, state0, forcing):
        p = dict(params)
        p.update({k: jnp.exp(v) for k, v in theta.items()})

        @jax.checkpoint
        def step(s, f):
            ns, _ = full_step(s, f, geom, p)
            return ns, ns.u

        @jax.checkpoint
        def chunk(s, fc):
            s, u = lax.scan(step, s, fc)
            return jax.tree_util.tree_map(lax.stop_gradient, s), u

        n = forcing["Uw"].shape[0]
        fch = jax.tree_util.tree_map(lambda v: v.reshape((n // chunk_steps, chunk_steps) + v.shape[1:]), forcing)
        _, u = lax.scan(chunk, state0, fch)
        return u.reshape((n,) + u.shape[2:])
    return sim


def main():
    t0 = time.time()
    s0, forcing, geom, params, start = setup(os.path.abspath(args.lake_dir))
    W = (args.window_steps // args.chunk_steps) * args.chunk_steps
    if W != args.window_steps:
        print(f"window steps rounded down to a multiple of --chunk-steps: {W}")
    times = start + pd.to_timedelta(np.arange(forcing["Uw"].shape[0]), unit="h")
    years = sorted(set(times.year))
    starts = []
    for y in years:
        i = int(np.searchsorted(times, pd.Timestamp(f"{y}-{args.window_start}")))
        if i + W <= len(times):
            starts.append(i)
    batches = [int(b) for b in args.batches.split(",")]
    print(f"setup {time.time() - t0:.0f}s; {len(starts)} possible windows of {W} steps "
          f"(chunk {args.chunk_steps}); devices: {jax.device_count()} ({jax.devices()[0].platform})")

    theta = {k: jnp.log(jnp.asarray(float(params[k]))) for k in PARAMS}
    sim = make_window_sim(geom, params, args.chunk_steps)
    loss1 = lambda th, s, f: jnp.mean(sim(th, s, f))
    fwd1 = jax.jit(loss1)
    grad1 = jax.jit(jax.value_and_grad(loss1))

    def batch_inputs(B):
        idx = [starts[i % len(starts)] for i in range(B)]
        f = {k: jnp.stack([jnp.asarray(v[i:i + W]) for i in idx]) for k, v in forcing.items()}
        s = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)), s0)
        return LakeState(s), f

    def lossB(th, S, F):
        return jnp.sum(jax.vmap(loss1, in_axes=(None, 0, 0))(th, S, F))
    fwdB, gradB = jax.jit(lossB), jax.jit(jax.value_and_grad(lossB))

    D = jax.device_count()

    def pm_loss(th, S, F):  # S, F: (D, B/D, ...)
        return jax.pmap(lambda s, f: jnp.sum(jax.vmap(loss1, in_axes=(None, 0, 0))(th, s, f)))(S, F)
    pm_grad = jax.pmap(jax.value_and_grad(lambda th, s, f: jnp.sum(jax.vmap(loss1, in_axes=(None, 0, 0))(th, s, f))),
                       in_axes=(None, 0, 0))

    rows = []
    for mode in args.modes.split(","):
        for what in args.what.split(","):
            for B in batches:
                S, F = batch_inputs(B)
                if mode == "seq":
                    fn = fwd1 if what == "fwd" else grad1
                    def run():
                        out = None
                        for b in range(B):
                            sb = jax.tree_util.tree_map(lambda x: x[b], S)
                            fb = {k: v[b] for k, v in F.items()}
                            out = fn(theta, LakeState(sb), fb)
                        return jax.block_until_ready(out)
                elif mode == "vmap":
                    fn = fwdB if what == "fwd" else gradB
                    run = lambda: jax.block_until_ready(fn(theta, S, F))
                elif mode == "pmap":
                    if B % D:
                        print(f"  skip pmap B={B}: not divisible by {D} devices"); continue
                    rs = lambda x: x.reshape((D, B // D) + x.shape[1:])
                    Sd = LakeState({k: rs(S[k]) for k in S})  # keep LakeState key order
                    Fd = {k: rs(v) for k, v in F.items()}
                    run = (lambda: jax.block_until_ready(pm_loss(theta, Sd, Fd))) if what == "fwd" else \
                          (lambda: jax.block_until_ready(pm_grad(theta, Sd, Fd)))
                else:
                    raise SystemExit(f"unknown mode {mode}")
                t = time.time(); run(); first = time.time() - t
                best = float("inf")
                for _ in range(args.repeats):
                    t = time.time(); run(); best = min(best, time.time() - t)
                row = dict(mode=mode, what=what, batch=B, window_steps=W, chunk_steps=args.chunk_steps,
                           devices=D, first_call_s=first, run_s=best, compile_s=max(first - best, 0.0),
                           ms_per_window_step=1000 * best / (B * W))
                rows.append(row)
                print(f"  {mode:4s} {what:4s} B={B:3d}: compile ~{row['compile_s']:6.1f}s  run {best:7.2f}s  "
                      f"-> {row['ms_per_window_step']:.3f} ms per window-step", flush=True)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"appended {len(df)} rows to {args.out}")
    # extrapolation to a full calibration iteration
    if len(df):
        print("\nExtrapolated to one iteration over 31 windows of 8016 steps (gradient):")
        for (mode, what), g in df[df["what"] == "grad"].groupby(["mode", "what"]):
            r = g.sort_values("batch").iloc[-1]
            per = r["ms_per_window_step"] / 1000
            print(f"  {mode:4s}: {per * 31 * 8016 / 60:.1f} min (using B={int(r['batch'])} cost per window-step)")


if __name__ == "__main__":
    main()
