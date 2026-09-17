"""
Learned (LSTM) correction to the process-based eddy-diffusivity closure.

`jax_lakeModel_functions.eddy_diffusivity_hendersonSellers` (`kz` in the
model's own notation) is the empirical Henderson-Sellers vertical-mixing
closure used everywhere temperature/O2/DOC/POC diffuse or settle. This
script asks a narrower question than a full model calibration: can a small
recurrent network, trained against the buoy/HD temperature record, learn a
*correction* to that closure that the fixed empirical formula is missing,
and does it actually improve the temperature fit on data it wasn't trained
on?

Design (see the feasibility discussion this script followed from, and the
diagnostic follow-up that motivated the depth-basis/stratification-feature
revision below -- an initial uniform-plus-linear-in-depth version fixed the
deep-water bias nicely (test RMSE at >=20m dropped ~44%) but barely moved
the surface (~9%), and the per-snapshot kz profiles showed mismatches with
a kink near the thermocline that a 2-coefficient linear-in-depth form
cannot represent, being monotonic in depth by construction):

  * Hybrid/residual, not a full replacement. Every step, the process-based
    `kz_process` is still computed exactly as before (via `full_step`'s
    `kz_override` hook -- see its docstring), and the network only supplies
    a *multiplicative* correction, now a degree-`--depth-basis-degree`
    (default 3) polynomial in depth rather than a straight line:

        kz_final(depth) = kz_process(depth) * exp(c_0 + c_1*z + c_2*z^2 + ... )

    with `z = depth_norm = depth / max(depth)` and `(c_0, c_1, ...)` the
    `--depth-basis-degree + 1` numbers a small LSTM emits each step. This
    is still deliberately low-dimensional (4 numbers/step at the default
    degree 3, not a raw `nx`-length output) -- kz is never directly
    observed, only temperature is, so an unconstrained high-capacity
    network would be free to absorb *any* other model error into kz
    without anything in the loss telling it not to. A cubic-in-depth log
    correction can represent a non-monotonic (e.g. thermocline-centered)
    adjustment that the original linear form could not, while `--kz-reg`
    (a small L2 penalty on the network's weights, default
    `DEFAULT_KZ_REG`) keeps the added degrees of freedom from chasing
    noise where observations are sparse. Set `--depth-basis-degree 1` to
    recover the original straight-line-in-depth behavior.

  * The network's final Dense layer is zero-initialized (both kernel and
    bias), so at initialization every `c_k = 0` and the hybrid model is
    *exactly* the process-based model (`correction == 1` everywhere,
    regardless of the basis degree). Any deviation is then attributable to
    training, not initialization noise.

  * Per-step network inputs: the 9 meteorological forcing series
    (Tair/CC/ea/Jsw/Jlw/Uw/Pa/RH/PP), z-scored using TRAIN-window mean/std
    (computed once, not data-snooped from the test window), plus 7 dynamic
    features taken from the *incoming* state at the start of the step (not
    this step's post-heating/ice values, to avoid a circular dependency on
    this step's own `kz_process`): the ice flag, surface and bottom
    temperature (`state.u[0]`, `state.u[-1]`, scaled by 1/10),
    `log10(mean(state.kz))` (the *previous* step's diffusivity -- mirrors
    the process-based closure's own `kzn_prev` memory term), and 3
    stratification features computed from `calc_dens(state.u)`: the bulk
    and peak magnitude of `|d(density)/d(depth)|` (log-scaled) and the
    normalized depth at which that gradient peaks (the current thermocline
    location). These three mirror the exact `buoy`/`diff_rho` quantity
    `eddy_diffusivity_hendersonSellers` itself computes internally (see
    that function) -- handing the network the same physical signal the
    process-based closure's own mixing decision is based on, rather than
    making it infer stratification indirectly from raw temperatures, and
    giving the (now depth-resolved) correction a direct cue for *where*
    the thermocline currently sits.

  * Only the network's weights are trained; every physical parameter stays
    fixed at whatever is already in `model_params.csv`. This keeps the
    comparison clean -- any difference between the baseline and hybrid
    runs is attributable to the kz correction alone, not to also
    re-fitting other physics at the same time.

  * Chronological train/test split (`--train-start/--train-end` default
    2023, `--test-start/--test-end` default 2024) rather than a random
    split: avoids leaking test-period information into the LSTM's
    recurrent state during training, and both years now have full hourly
    buoy coverage (see `calibrate_M3_jax.load_buoy_temperature`). The
    physical simulation itself still starts from `run_config.csv`'s own
    `start_time` (2022 here, providing a needed thermal spin-up); only the
    *loss* is restricted to the training window's observations.

  * Training reuses the same truncated-BPTT (chunked, nested `lax.scan`,
    `jax.checkpoint`) machinery as `calibrate_M3_jax.py`'s
    `simulate_truncated`, extended to carry the LSTM's hidden/cell state
    through the chunked scan alongside the physical state -- so it is
    subject to the same chunk-boundary `stop_gradient` truncation as
    everything else (a safety net, not a new limitation; see that
    script's module docstring).

Known limitations worth keeping in mind when reading the results: kz has
no direct observational target, so "improvement" here is only ever an
indirect, temperature-mediated signal -- a lower test-period RMSE means
the correction generalizes, but it does not by itself prove the learned
kz is physically correct at depths/times far from where temperature
gradients happen to be informative. Training a network end-to-end through
a multi-year hourly rollout is also a substantially bigger optimization
problem than fitting a handful of scalar parameters (see
`calibrate_M3_jax.py`); expect a full run to take much longer than that
script's calibration runs.

Usage:
    python src/run_M3_mcl_jax.py Ravn [--steps N] [--chunk-steps N]
        [--hidden-size H] [--depth-basis-degree D] [--kz-reg LAMBDA]
        [--iters K] [--lr LR]
        [--train-start DATE --train-end DATE --test-start DATE --test-end DATE]
        [--out result.npz] [--nn-params-out nn_params.pkl]
"""
import argparse
import os
import pickle
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import optax
import flax.linen as nn

from processBased_lakeModel_functions import (
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile,
    provide_phosphorus, provide_carbon, get_lake_config, get_model_params, get_run_config,
    get_ice_and_snow,
)
from jax_lakeModel_functions import (
    default_params, default_geometry_wq, make_initial_state_full, full_step, calc_dens,
)
from run_M3_jax import build_forcing_series, add_wq_forcing_series
from calibrate_M3_jax import (
    load_observations, _rmse, _nse, _kge, _r2,
    DEFAULT_CHUNK_STEPS, DEFAULT_EARLY_STOP_PATIENCE, DEFAULT_EARLY_STOP_TOL,
)


DEFAULT_HIDDEN_SIZE = 16
DEFAULT_ITERS = 40
DEFAULT_LR = 1e-2
DEFAULT_DEPTH_BASIS_DEGREE = 3   # polynomial-in-depth degree for the log-correction (1 = old linear form)
DEFAULT_KZ_REG = 1e-4            # L2 penalty on the NN's weights (0 disables)
DEFAULT_TRAIN_START = "2023-01-01"
DEFAULT_TRAIN_END = "2023-12-31"
DEFAULT_TEST_START = "2024-01-01"
DEFAULT_TEST_END = "2024-12-31"

# Meteorological forcing series used as (z-scored) NN inputs every step.
FORCING_FEATURE_KEYS = ["Tair", "CC", "ea", "Jsw", "Jlw", "Uw", "Pa", "RH", "PP"]
# ice flag, surface temp/10, bottom temp/10, log10(mean(prev kz))/10,
# bulk/peak log10(density gradient), thermocline depth_norm location
N_DYNAMIC_FEATURES = 7
N_FEATURES = len(FORCING_FEATURE_KEYS) + N_DYNAMIC_FEATURES


# ---------------------------------------------------------------------------
# The correction network: a single LSTM cell + a zero-initialized linear head
# producing the `depth_basis_degree + 1` coefficients of a polynomial-in-
# depth log correction (degree 1 = the original uniform-plus-linear form).
# ---------------------------------------------------------------------------

def make_modules(hidden_size, depth_basis_degree):
    lstm = nn.OptimizedLSTMCell(features=hidden_size, param_dtype=jnp.float64, dtype=jnp.float64)
    head = nn.Dense(
        depth_basis_degree + 1, param_dtype=jnp.float64, dtype=jnp.float64,
        kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
    )
    return lstm, head


def init_nn_params(key, hidden_size, depth_basis_degree):
    lstm, head = make_modules(hidden_size, depth_basis_degree)
    k_lstm, k_head = jax.random.split(key)
    carry0 = lstm.initialize_carry(k_lstm, (N_FEATURES,))
    x0 = jnp.zeros((N_FEATURES,), dtype=jnp.float64)
    lstm_params = lstm.init(k_lstm, carry0, x0)
    (_, _), y0 = lstm.apply(lstm_params, carry0, x0)
    head_params = head.init(k_head, y0)
    return dict(lstm=lstm_params, head=head_params)


def build_depth_basis(depth, depth_basis_degree):
    """`(degree+1, nx)` array of `depth_norm**k` for `k in 0..degree`, so the
    correction coefficients combine via a single `jnp.dot`."""
    depth_norm = depth / jnp.max(depth)
    return jnp.stack([depth_norm ** k for k in range(depth_basis_degree + 1)], axis=0)


def compute_density_gradient_features(u, depth, g):
    """Stratification cue for the NN: bulk and peak magnitude of
    `|d(density)/d(depth)|` (log10-scaled) and the normalized depth at
    which it peaks (current thermocline location). Deliberately computed
    the same way `eddy_diffusivity_hendersonSellers`'s own `buoy`/
    `diff_rho` term is (see that function) -- this hands the network the
    same physical signal the process-based closure's mixing decision is
    based on, rather than making it infer stratification indirectly from
    raw surface/bottom temperatures."""
    dens = calc_dens(u)
    rho_0 = jnp.mean(dens)
    diff_rho = jnp.abs(dens[1:] - dens[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    diff_rho = jnp.concatenate([diff_rho, diff_rho[-1:]])  # length nx, same padding as the closure itself
    bulk = jnp.log10(jnp.mean(diff_rho) + 1e-12) / 10.0
    peak = jnp.log10(jnp.max(diff_rho) + 1e-12) / 10.0
    depth_norm = depth / jnp.max(depth)
    thermocline_loc = depth_norm[jnp.argmax(diff_rho)]
    return jnp.array([bulk, peak, thermocline_loc])


def build_forcing_features(forcing, train_lo, train_hi):
    """Z-score each of `FORCING_FEATURE_KEYS` using TRAIN-window mean/std
    (never the test window), stacked into a (n_steps, 9) array. Returns the
    array plus the (mean, std) stats used, for reproducibility/inspection."""
    cols = []
    stats = {}
    for key in FORCING_FEATURE_KEYS:
        vals = forcing[key]
        train_vals = vals[train_lo:train_hi]
        mean = jnp.mean(train_vals)
        std = jnp.std(train_vals)
        std = jnp.where(std < 1e-8, 1.0, std)
        stats[key] = (float(mean), float(std))
        cols.append((vals - mean) / std)
    return jnp.stack(cols, axis=1), stats


# ---------------------------------------------------------------------------
# Baseline (pure process-based) simulation: a single flat `lax.scan`, no
# chunking/checkpointing needed since nothing is differentiated through it.
# ---------------------------------------------------------------------------

def simulate_baseline(phys_params, geometry, forcing, ice_state, init_state):
    u0, o2_0, docr_0, docl_0, pocr_0, pocl_0 = init_state
    nx = u0.shape[0]
    state0 = make_initial_state_full(u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, **ice_state)

    def step_body(state, forcing_t):
        new_state, diag = full_step(state, forcing_t, geometry, phys_params)
        outputs = dict(u=new_state.u, kz=new_state.kz)
        return new_state, outputs

    _, per_step = lax.scan(step_body, state0, forcing)
    return per_step


# ---------------------------------------------------------------------------
# Hybrid (process + learned correction) simulation: nested chunked scan,
# same truncated-BPTT structure as calibrate_M3_jax.simulate_truncated, but
# carrying the LSTM's (h, c) alongside the physical state.
# ---------------------------------------------------------------------------

def simulate_hybrid(nn_params, phys_params, geometry, forcing, ice_state, init_state,
                     chunk_steps, hidden_size, depth_basis_degree, forcing_features):
    lstm, head = make_modules(hidden_size, depth_basis_degree)

    u0, o2_0, docr_0, docl_0, pocr_0, pocl_0 = init_state
    nx = u0.shape[0]
    state0 = make_initial_state_full(u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, **ice_state)
    depth = geometry["depth"]
    depth_powers = build_depth_basis(depth, depth_basis_degree)

    forcing_with_features = dict(forcing)
    forcing_with_features["_nn_static"] = forcing_features

    n_steps = forcing["Uw"].shape[0]
    n_chunks = -(-n_steps // chunk_steps)  # ceil division
    n_padded = n_chunks * chunk_steps
    pad = n_padded - n_steps

    if pad > 0:
        forcing_padded = {
            k: jnp.concatenate([v, jnp.broadcast_to(v[-1], (pad,) + v.shape[1:])], axis=0)
            for k, v in forcing_with_features.items()
        }
    else:
        forcing_padded = forcing_with_features

    forcing_chunked = jax.tree_util.tree_map(
        lambda v: v.reshape((n_chunks, chunk_steps) + v.shape[1:]), forcing_padded,
    )

    h0 = jnp.zeros((hidden_size,), dtype=jnp.float64)
    c0 = jnp.zeros((hidden_size,), dtype=jnp.float64)

    @jax.checkpoint
    def step_body(carry, forcing_t):
        state, h, c = carry

        ice_flag = jnp.where(state.ice, 1.0, 0.0)
        strat_feat = compute_density_gradient_features(state.u, depth, phys_params["g"])
        dyn_feat = jnp.concatenate([
            jnp.array([
                ice_flag,
                state.u[0] / 10.0,
                state.u[-1] / 10.0,
                jnp.log10(jnp.mean(state.kz) + 1e-12) / 10.0,
            ]),
            strat_feat,
        ])
        features = jnp.concatenate([forcing_t["_nn_static"], dyn_feat])

        (h, c), y = lstm.apply(nn_params["lstm"], (h, c), features)
        coefs = head.apply(nn_params["head"], y)
        correction = jnp.exp(jnp.dot(coefs, depth_powers))

        def kz_override(kz_process, u, ice, dens_u, forcing_step):
            return kz_process * correction

        new_state, diag = full_step(state, forcing_t, geometry, phys_params, kz_override=kz_override)
        outputs = dict(
            u=new_state.u, o2=new_state.o2, docr=new_state.docr, docl=new_state.docl,
            pocr=new_state.pocr, pocl=new_state.pocl,
            kz=new_state.kz, kz_process=diag["kz_process"],
        )
        return (new_state, h, c), outputs

    @jax.checkpoint
    def chunk_body(carry, forcing_chunk):
        carry, per_step_c = lax.scan(step_body, carry, forcing_chunk)
        carry = jax.tree_util.tree_map(lax.stop_gradient, carry)
        return carry, per_step_c

    init_carry = (state0, h0, c0)
    _, per_step_chunked = lax.scan(chunk_body, init_carry, forcing_chunked)
    return {
        k: v.reshape((n_padded,) + v.shape[2:])[:n_steps]
        for k, v in per_step_chunked.items()
    }


# ---------------------------------------------------------------------------
# Observation-window splitting, loss, metrics.
# ---------------------------------------------------------------------------

def split_obs_by_step_range(obs_var, lo, hi):
    """Restrict a `load_observations()`-style temp obs dict to profile
    dates whose `step_idx` falls in `[lo, hi)`. Returns `None` if empty."""
    if obs_var is None:
        return None
    step_idx = np.asarray(obs_var["step_idx"])
    keep = (step_idx >= lo) & (step_idx < hi)
    if not keep.any():
        return None
    return dict(
        step_idx=jnp.asarray(step_idx[keep]),
        values=jnp.asarray(np.asarray(obs_var["values"])[keep]),
        mask=jnp.asarray(np.asarray(obs_var["mask"])[keep]),
    )


def make_loss_fn(phys_params, geometry, forcing, ice_state, init_state, obs_train,
                  chunk_steps, hidden_size, depth_basis_degree, forcing_features, reg_weight=0.0):
    def loss_fn(nn_params):
        per_step = simulate_hybrid(
            nn_params, phys_params, geometry, forcing, ice_state, init_state,
            chunk_steps, hidden_size, depth_basis_degree, forcing_features,
        )
        sim = per_step["u"][obs_train["step_idx"]]
        err2 = (sim - obs_train["values"]) ** 2 * obs_train["mask"]
        data_loss = jnp.sum(err2) / jnp.maximum(jnp.sum(obs_train["mask"]), 1.0)
        if reg_weight > 0:
            # Small L2 penalty on the network's own weights (not the runtime
            # kz correction itself) -- standard weight decay, added because
            # the expanded depth basis below has more freedom to overfit
            # where observations are sparse than the original 2-coefficient
            # linear form did.
            reg = sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(nn_params))
            data_loss = data_loss + reg_weight * reg
        return data_loss

    return loss_fn


def extract_temp_pairs(u_all, obs_var):
    if obs_var is None:
        return None
    sim = np.asarray(u_all[np.asarray(obs_var["step_idx"])])
    mask = np.asarray(obs_var["mask"])
    return sim[mask], np.asarray(obs_var["values"])[mask]


def compute_temp_metrics(u_all, obs_var):
    pair = extract_temp_pairs(u_all, obs_var)
    if pair is None:
        return None
    sim, obsv = pair
    return dict(n=len(obsv), rmse=_rmse(sim, obsv), nse=_nse(sim, obsv),
                kge=_kge(sim, obsv), r2=_r2(sim, obsv))


def print_comparison_row(label, m_base, m_hybrid):
    if m_base is None or m_hybrid is None:
        print(f"  {label:22s}  no observations in this window")
        return
    print(f"  {label:22s} n={m_hybrid['n']:5d}  "
          f"RMSE {m_base['rmse']:8.4g} -> {m_hybrid['rmse']:<8.4g}  "
          f"NSE {m_base['nse']:7.3f} -> {m_hybrid['nse']:<7.3f}  "
          f"KGE {m_base['kge']:7.3f} -> {m_hybrid['kge']:<7.3f}  "
          f"R2 {m_base['r2']:7.3f} -> {m_hybrid['r2']:<7.3f}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=".")
    parser.add_argument("--steps", type=int, default=None,
                         help="total simulation window in hourly steps (default: the full "
                              "record per run_config.csv's start_time/end_time)")
    parser.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS,
                         help=f"gradient-truncation chunk size (default {DEFAULT_CHUNK_STEPS}; "
                              "see calibrate_M3_jax.py's module docstring on truncated BPTT)")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE,
                         help=f"LSTM hidden size (default {DEFAULT_HIDDEN_SIZE})")
    parser.add_argument("--depth-basis-degree", type=int, default=DEFAULT_DEPTH_BASIS_DEGREE,
                         help=f"degree of the polynomial-in-depth log kz correction (default "
                              f"{DEFAULT_DEPTH_BASIS_DEGREE}; 1 = the original uniform-plus-linear "
                              "form, higher values allow non-monotonic-in-depth corrections at the "
                              "cost of more free parameters -- see --kz-reg)")
    parser.add_argument("--kz-reg", type=float, default=DEFAULT_KZ_REG,
                         help=f"L2 weight-decay penalty on the NN's own weights (default "
                              f"{DEFAULT_KZ_REG}; 0 disables). Keeps a higher --depth-basis-degree "
                              "from overfitting where observations are sparse.")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                         help=f"maximum Adam iterations (default {DEFAULT_ITERS}; may stop earlier)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Adam learning rate (default {DEFAULT_LR})")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-tol", type=float, default=DEFAULT_EARLY_STOP_TOL)
    parser.add_argument("--train-start", type=str, default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--test-start", type=str, default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", type=str, default=DEFAULT_TEST_END)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="mcl_result.npz",
                         help="output .npz path (relative to data_dir) with the kz/temperature "
                              "fields (default mcl_result.npz)")
    parser.add_argument("--nn-params-out", type=str, default="mcl_nn_params.pkl",
                         help="pickle path (relative to data_dir) to save the trained NN weights")
    args = parser.parse_args()

    os.chdir(args.data_dir)
    lake_num = 1
    lake_config = get_lake_config("./lake_config.csv", lake_num)
    model_params = get_model_params("./model_params.csv", lake_num)
    run_config = get_run_config("./run_config.csv", lake_num)
    ice_and_snow = get_ice_and_snow("./ice_and_snow.csv", lake_num)

    windfactor = float(lake_config["WindSpeed"])
    nx = int(run_config["nx"]); dt = float(run_config["dt"]); dx = float(run_config["dx"])
    area, depth, volume, hypso_weight = get_hypsography(
        "./lake_bathymetry.csv", dx=dx, nx=nx, outflow_depth=float(lake_config["outflow_depth"]),
    )

    desired_start = pd.Timestamp(run_config["start_time"])
    meteo_all = provide_meteorology(
        meteofile=run_config["meteo_ini_file"], windfactor=windfactor, lat=lake_config["Latitude"],
        lon=lake_config["Longitude"], elev=lake_config["Elevation"], startDate=desired_start,
    )
    u_ini = initial_profile(initfile=run_config["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=desired_start)
    wq_ini = wq_initial_profile(
        initfile=run_config["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=desired_start,
    )
    tp_boundary = provide_phosphorus(tpfile=run_config["tp_ini_file"], startingDate=desired_start, startTime=1)
    carbon_data = provide_carbon(ocloadfile=run_config["oc_load_file"], startingDate=desired_start, startTime=1)
    carbon_data = carbon_data.dropna(subset=["oc"])

    desired_end = pd.Timestamp(run_config["end_time"])
    n_steps_full = len(pd.date_range(desired_start, desired_end, freq="h"))
    n_steps = args.steps if args.steps is not None else n_steps_full
    step_times = np.arange(1, (n_steps + 1) * dt, dt)[:n_steps]
    forcing = build_forcing_series(meteo_all, step_times, windfactor)
    forcing = add_wq_forcing_series(forcing, tp_boundary, carbon_data, step_times)

    mean_depth = float(np.sum(volume) / np.max(area))
    hydro_res_time_hr = float(model_params["hydro_res_time"]) * 8760
    geometry = default_geometry_wq(
        area=area, depth=depth, volume=volume, dx=dx, dt=dt, latitude=float(lake_config["Latitude"]),
        altitude=float(lake_config["Elevation"]), hypso_weight=hypso_weight, mean_depth=mean_depth,
        hydro_res_time_hr=hydro_res_time_hr,
    )
    phys_params = default_params(model_params, ice_and_snow)

    def _to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        return bool(x)

    ice_state = dict(
        ice=_to_bool(ice_and_snow["ice"]), Hi=ice_and_snow["Hi"], Hs=ice_and_snow["Hs"],
        Hsi=ice_and_snow["Hsi"], iceT=ice_and_snow["iceT"], rho_snow=ice_and_snow["rho_snow"],
    )

    u0 = jnp.asarray(deepcopy(u_ini), dtype=jnp.float64)
    o2_0 = jnp.asarray(deepcopy(wq_ini[0]), dtype=jnp.float64)
    docr_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.75, dtype=jnp.float64)
    docl_0 = jnp.asarray(deepcopy(wq_ini[1]) * 0.25, dtype=jnp.float64)
    pocr_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)
    pocl_0 = jnp.asarray(0.5 * volume, dtype=jnp.float64)
    init_state = (u0, o2_0, docr_0, docl_0, pocr_0, pocl_0)

    # --- train/test windows, as step indices into this simulation ---
    def to_step_idx(date_str):
        step_seconds = (pd.Timestamp(date_str) - desired_start).total_seconds()
        return int(round(step_seconds / dt))

    train_lo, train_hi = to_step_idx(args.train_start), to_step_idx(args.train_end) + 1
    test_lo, test_hi = to_step_idx(args.test_start), to_step_idx(args.test_end) + 1
    train_lo, train_hi = max(0, train_lo), min(n_steps, train_hi)
    test_lo, test_hi = max(0, test_lo), min(n_steps, test_hi)
    print(f"Simulating {n_steps} steps from {desired_start.date()}; "
          f"train window steps [{train_lo}, {train_hi}) ({args.train_start} - {args.train_end}), "
          f"test window steps [{test_lo}, {test_hi}) ({args.test_start} - {args.test_end})")
    if train_hi <= train_lo:
        raise SystemExit("Train window falls outside the simulated record -- adjust --train-start/--train-end "
                          "or --steps.")

    print("Loading temperature observations (L0001-HD.csv + buoy ravn_2023/2024.json)...")
    obs = load_observations("./", depth, volume, desired_start, step_times, dt)
    obs_train = split_obs_by_step_range(obs["temp"], train_lo, train_hi)
    obs_test = split_obs_by_step_range(obs["temp"], test_lo, test_hi)
    if obs_train is None:
        raise SystemExit("No temperature observations fall inside the training window -- nothing to train against.")
    print(f"  train: {int(obs_train['step_idx'].shape[0])} profile(s), "
          f"test: {int(obs_test['step_idx'].shape[0]) if obs_test is not None else 0} profile(s)")

    forcing_features, feature_stats = build_forcing_features(forcing, train_lo, train_hi)

    # --- baseline (process-based) run ---
    print("\nRunning baseline (process-based kz) simulation...")
    t0 = time.time()
    baseline_fn = jax.jit(lambda p: simulate_baseline(p, geometry, forcing, ice_state, init_state))
    baseline = baseline_fn(phys_params)
    jax.block_until_ready(baseline)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- train the NN correction ---
    key = jax.random.PRNGKey(args.seed)
    nn_params = init_nn_params(key, args.hidden_size, args.depth_basis_degree)

    loss_fn = make_loss_fn(
        phys_params, geometry, forcing, ice_state, init_state, obs_train,
        args.chunk_steps, args.hidden_size, args.depth_basis_degree, forcing_features,
        reg_weight=args.kz_reg,
    )
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    print(f"\nTraining LSTM kz-correction ({args.hidden_size} hidden units, depth-basis degree "
          f"{args.depth_basis_degree}, kz-reg {args.kz_reg:g}) for up to {args.iters} "
          f"Adam iterations against the {args.train_start}-{args.train_end} training window...")
    opt = optax.adam(args.lr)
    opt_state = opt.init(nn_params)

    val0, _ = grad_fn(nn_params)
    jax.block_until_ready(val0)
    reg_note = (" -- includes L2 penalty on the randomly-initialized LSTM weights, so this "
                "isn't quite the baseline's own loss, but the temperature fit itself still is"
                if args.kz_reg > 0 else " (untrained -- identical to baseline)")
    print(f"  iter  -1: loss={float(val0):.6g}{reg_note}")

    best_loss, best_params = float(val0), deepcopy(nn_params)
    stall_count = 0
    for it in range(args.iters):
        t0 = time.time()
        loss_val, grads = grad_fn(nn_params)
        jax.block_until_ready((loss_val, grads))
        non_finite = not bool(jnp.isfinite(loss_val)) or not all(
            bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads)
        )
        if non_finite:
            print(f"  iter {it}: non-finite loss/gradient -- stopping early, keeping best result so far.")
            break
        updates, opt_state = opt.update(grads, opt_state, nn_params)
        nn_params = optax.apply_updates(nn_params, updates)
        t1 = time.time()
        loss_f = float(loss_val)
        print(f"  iter {it:3d}: loss={loss_f:.6g}  ({t1 - t0:.1f}s)")

        improved_enough = (best_loss - loss_f) > args.early_stop_tol * max(abs(best_loss), 1e-12)
        if loss_f < best_loss:
            best_loss, best_params = loss_f, deepcopy(nn_params)
        if args.early_stop_patience > 0:
            if improved_enough:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= args.early_stop_patience:
                    print(f"  iter {it}: loss hasn't improved by more than {args.early_stop_tol:.4g} "
                          f"(relative) for {args.early_stop_patience} consecutive iteration(s) -- "
                          "stopping early. Keeping the best result found so far.")
                    break

    nn_params = best_params
    print(f"  best training loss: {best_loss:.6g}")

    # --- final hybrid simulation at the trained weights ---
    print("\nRunning final hybrid (process + learned correction) simulation...")
    t0 = time.time()
    hybrid_fn = jax.jit(lambda p: simulate_hybrid(
        p, phys_params, geometry, forcing, ice_state, init_state,
        args.chunk_steps, args.hidden_size, args.depth_basis_degree, forcing_features,
    ))
    hybrid = hybrid_fn(nn_params)
    jax.block_until_ready(hybrid)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- metrics: baseline vs hybrid, train vs test window ---
    print("\nTemperature fit: baseline (process-based kz) -> hybrid (process + LSTM correction)")
    m_base_train = compute_temp_metrics(baseline["u"], obs_train)
    m_hybrid_train = compute_temp_metrics(hybrid["u"], obs_train)
    print_comparison_row(f"train {args.train_start[:4]}", m_base_train, m_hybrid_train)
    m_base_test = compute_temp_metrics(baseline["u"], obs_test)
    m_hybrid_test = compute_temp_metrics(hybrid["u"], obs_test)
    print_comparison_row(f"test {args.test_start[:4]}", m_base_test, m_hybrid_test)

    # --- save outputs ---
    out_path = os.path.join(os.getcwd(), args.out)
    np.savez(
        out_path,
        times=step_times, depth=np.asarray(depth),
        temp_baseline=np.asarray(baseline["u"]), temp_hybrid=np.asarray(hybrid["u"]),
        kz_baseline=np.asarray(baseline["kz"]),
        kz_hybrid=np.asarray(hybrid["kz"]), kz_process_hybrid=np.asarray(hybrid["kz_process"]),
        train_lo=train_lo, train_hi=train_hi, test_lo=test_lo, test_hi=test_hi,
    )
    print(f"\nSaved kz/temperature fields to {out_path}")

    nn_params_path = os.path.join(os.getcwd(), args.nn_params_out)
    with open(nn_params_path, "wb") as f:
        pickle.dump(dict(
            nn_params=jax.tree_util.tree_map(np.asarray, nn_params),
            hidden_size=args.hidden_size, depth_basis_degree=args.depth_basis_degree,
            kz_reg=args.kz_reg, feature_stats=feature_stats,
            forcing_feature_keys=FORCING_FEATURE_KEYS,
        ), f)
    print(f"Saved trained NN weights to {nn_params_path}")


if __name__ == "__main__":
    main()
