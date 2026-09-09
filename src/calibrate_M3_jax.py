"""
Gradient-based calibration for the JAX lake model.

Fits selected `model_params.csv` entries so the model's simulated
temperature/oxygen/DOC depth profiles better match the observed profiles
in `Ravn/L0001-HD.csv` (temperature) and `Ravn/L0001-WQ.csv` (dissolved
oxygen "do" and dissolved organic carbon "doc", both in mg/L), using
`jax.grad` through the full differentiable simulation
(`simulate_truncated()` below, built on `full_step` in
`jax_lakeModel_functions.py`).

Usage:
    python src/calibrate_M3_jax.py Ravn [--steps N] [--chunk-steps N] [--iters K]
        [--lr LR] [--topk K] [--params p1,p2,...] [--sensitivity-only]

Key design choices
-------------------
Unit conversion (mass vs. concentration): the model's internal WQ state
(`o2`, `docr`, `docl`, ...) is a per-layer *mass*, not a concentration --
exactly like `wq_initial_profile()` in `processBased_lakeModel_functions.py`
builds its initial condition as `observation * volume`. Observations here
are converted the same way: `obs_mass = obs_concentration_mgL * volume`
(volume in m^3; mg/L is numerically g/m^3, so this is already in the same
mass units the model itself uses -- no extra factor needed). DOC is
compared against `docr + docl` combined, since the model only initializes
that 75/25 refractory/labile split heuristically and there is no separate
refractory/labile observation to calibrate each piece against. Temperature
needs no conversion.

Observation matching: `L0001-HD.csv`/`L0001-WQ.csv` are long-format,
irregular in both time (roughly monthly profile dates) and depth. For
each profile date, the observed depths are linearly interpolated onto the
model's depth grid (`processBased_lakeModel_functions.get_hypsography`'s
cell-centered `depth` array); grid points beyond the observed depth range
are masked out rather than extrapolated (matching the spirit of
`initial_profile`/`wq_initial_profile`'s own "extend to lake max depth by
repeating the deepest observation" rule would be an unjustified
assumption for calibration targets, so we simply don't score those
points). Each profile date is mapped to the nearest simulation step via
`round((obs_datetime - start_date).total_seconds() / dt)`.

Gradient horizon and truncated backpropagation through time (TBPTT):
`jax_lakeModel_functions.py`'s module docstring documents two genuine,
now-fixed gradient-safety bugs found while building this feature: (1)
`eddy_diffusivity_hendersonSellers` dividing by an exact-float64-zero
under deep, weakly-mixed conditions, and (2) `mixing_step`'s wind-mixing
`KE0` term taking `sqrt(0)` on exactly-calm-wind hours -- both are
`0 * inf = NaN` gradient artifacts at a single, identifiable expression,
not accumulated chaos. (2) is specifically what caused an earlier version
of this script to see NaN gradients somewhere between ~7500 and ~8000
hourly steps (roughly 10-11 months) into a continuous rollout over the
real Ravn forcing -- that was originally (incorrectly) attributed to an
inherent property of `mixing_step`'s chaotic, argmax-based branch
selection accumulating unbounded sensitivity. With both bugs fixed,
isolated ~2000-step windows spot-checked across the entire multi-year
Ravn record all give finite gradients, so there is no longer a known
fixed horizon to stay under.

Chunked/truncated backpropagation is still used here regardless, as a
safety net rather than a workaround for a known limit: this script runs
the *forward* simulation as one continuous multi-year trajectory (via
`simulate_truncated()`), but backpropagates through it in fixed-size
chunks (`--chunk-steps`, default `DEFAULT_CHUNK_STEPS` below), applying
`jax.lax.stop_gradient` to the carried state at each chunk boundary. This
is the standard "truncated BPTT" technique used for long RNN-like
rollouts: the simulated trajectory is identical to one un-chunked run
(chunking is invisible to the forward physics), but the backward pass
only ever has to differentiate through a single chunk's worth of steps
at a time. Two independent benefits fall out of this: (a) memory --
reverse-mode AD through `lax.scan` needs per-step residuals kept live for
the whole differentiated span, and chunking (together with `jax.checkpoint`
on the per-step body) keeps that bounded regardless of the total
simulation length; and (b) robustness -- if some future, still-undiscovered
singularity like (1)/(2) above turns up in one chunk, only that chunk's
gradient is lost, not the whole run's. Each parameter still receives a
gradient contribution from every chunk (summed into the total loss's
backward pass as usual), so the parameters are still fit against the
*entire* selected window, just without any single chunk's backward pass
spanning more than `--chunk-steps` steps. Setting `--chunk-steps` >=
`--steps` recovers the old, un-chunked behavior exactly (a single chunk).
This script still checks every gradient for finiteness each iteration and
stops early (keeping the best result so far) if a chunk ever does produce
a non-finite gradient.

Parameter selection ("sensitive" parameters): rather than guessing a fixed
list, this script screens a documented pool of `model_params.csv` entries
that `default_params()` actually uses (CANDIDATE_PARAMS below -- excluding
fixed physical constants like `g`/`sigma`, and the OC-partitioning
fractions `prop_oc_*`, which must stay on a simplex and so aren't safe to
tweak individually) by computing the elasticity `|d(loss)/d(theta) *
theta|` for each one. Two rankings are computed from the same forward
pass: a *combined* one (`|d(combined weighted loss)/d(log theta)|`, one
`jax.grad` call -- this is what "sensitivity screen" printed before this
feature existed), and a *per-variable* one (`|d(loss_temp)/d(log theta)|`,
`|d(loss_o2)/d(log theta)|`, `|d(loss_doc)/d(log theta)|` separately, each
an *unweighted* per-variable MSE so one variable's arbitrary
inverse-variance weight can't hide a parameter that matters a lot to a
different variable -- one `jax.jacrev` call over the 3 per-variable losses,
which is one extra backward pass per present variable, not a separate
model run each). The per-variable ranking exists because a parameter can
be highly influential for one variable (e.g. a light-attenuation
parameter for DOC) while contributing little to the combined loss simply
because that loss is dominated by a differently-scaled variable.

By default (`--select-mode per-variable`), the parameters actually
calibrated are the *union* of each present variable's top
`--topk-per-variable` (default 5) most sensitive parameters -- so
whichever of temp/O2/DOC have observations in the window, their own most
sensitive parameters are guaranteed to be included, not just whichever
parameters happen to dominate the combined loss. `--select-mode combined`
restores the old behavior (top `--topk`, default 6, from the combined
ranking only). `--params` bypasses both and calibrates exactly the named
parameters. Either way, selected parameters are optimized together with
Adam (`optax`), in log-space (so a single learning rate means "N%
relative step" for every parameter regardless of its raw scale, e.g.
km ~ 1e-6 vs. theta_r ~ 1.2), under a box constraint keeping each
parameter within [0.1x, 10x] of its starting value.

Evaluation: after optimization, the model is run once more at the initial
and at the calibrated parameters, and RMSE, NSE (Nash-Sutcliffe
efficiency), KGE (Kling-Gupta efficiency), and R^2 (squared Pearson
correlation) are reported for each variable that has observations
(temperature/O2/DOC), over exactly the (step, depth) pairs the loss
scores. These are plain numpy diagnostics computed after the fact, not
part of the differentiable loss itself. Printed to the console and saved
to `calibration_metrics.csv` alongside `calibration_result.csv` (the
parameter table).
"""
import argparse
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import optax

from processBased_lakeModel_functions import (
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile,
    provide_phosphorus, provide_carbon, get_lake_config, get_model_params, get_run_config,
    get_ice_and_snow,
)
from jax_lakeModel_functions import (
    default_params, default_geometry_wq, make_initial_state_full, full_step,
)
from run_M3_jax import build_forcing_series, add_wq_forcing_series


# `--steps` defaults to None (the full available record -- safe now thanks
# to chunked/truncated backprop). `--chunk-steps` is what needs to stay
# safely inside the empirically-found ~7500-8000 hourly-step gradient
# horizon documented in jax_lakeModel_functions.py (see module docstring);
# 2000 leaves a healthy margin.
DEFAULT_CHUNK_STEPS = 2000   # ~83 days
DEFAULT_ITERS = 20
DEFAULT_LR = 0.05       # log-space Adam step ~= 5% relative parameter change
DEFAULT_TOPK = 6
DEFAULT_TOPK_PER_VARIABLE = 5

# model_params.csv entries that default_params() actually threads into the
# simulation, restricted to ones that are safe/sensible to tweak in
# isolation. Excluded: physical constants (g, sigma), the OC-partitioning
# fractions prop_oc_docr/docl/pocr/pocl (must sum sensibly across pools,
# not safe to move independently), and beta (not a model_params.csv column
# at all -- run_wq_model never lets the caller override it either).
CANDIDATE_PARAMS = [
    "km", "weight_kz", "Cd", "denThresh",
    "kd_light", "light_water", "light_doc", "light_poc",
    "sw_factor", "at_factor", "turb_factor", "Hgeo",
    "theta_r", "theta_npp", "k_half",
    "resp_docr", "resp_docl", "resp_pocr", "resp_pocl",
    "settling_rate_labile", "settling_rate_refractory",
    "f_sod", "d_thick", "meltP", "p2", "eps", "emissivity",
]


def load_observations(data_dir, depth, volume, start_date, step_times, dt):
    """Load L0001-HD.csv/L0001-WQ.csv, restrict to the simulation window,
    and build per-variable dicts of (step_idx, values, mask) arrays on the
    model's depth grid. `values` are in the model's own units: degC for
    temperature, mass (obs_mgL * volume) for o2/doc. Returns a dict with
    keys "temp", "o2", "doc" (each None if no observations fall in the
    simulation window)."""
    depth = np.asarray(depth)
    volume = np.asarray(volume)
    n_steps = len(step_times)
    sim_end = start_date + pd.Timedelta(seconds=float(step_times[-1]))

    def profile_to_grid(depth_obs, val_obs):
        order = np.argsort(depth_obs)
        d_sorted, v_sorted = depth_obs[order], val_obs[order]
        # de-duplicate identical depths (irregular source data occasionally
        # repeats a depth for the same date) by averaging
        uniq_d, inv = np.unique(d_sorted, return_inverse=True)
        if len(uniq_d) < len(d_sorted):
            v_avg = np.zeros_like(uniq_d, dtype=float)
            counts = np.zeros_like(uniq_d, dtype=float)
            np.add.at(v_avg, inv, v_sorted)
            np.add.at(counts, inv, 1.0)
            v_sorted = v_avg / counts
            d_sorted = uniq_d
        if len(d_sorted) < 2:
            return None, None
        f = interp1d(d_sorted, v_sorted, kind="linear", bounds_error=False, fill_value=np.nan)
        vals = f(depth)
        mask = ~np.isnan(vals)
        return np.where(mask, vals, 0.0), mask

    def build(df, depth_col, value_col, mass_convert):
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= sim_end)]
        idxs, vals_list, mask_list = [], [], []
        for dtval, group in df.groupby("datetime"):
            vals, mask = profile_to_grid(group[depth_col].values.astype(float), group[value_col].values.astype(float))
            if vals is None or not mask.any():
                continue
            if mass_convert:
                vals = vals * volume
            step_idx = int(round((dtval - start_date).total_seconds() / dt))
            if step_idx < 0 or step_idx >= n_steps:
                continue
            idxs.append(step_idx)
            vals_list.append(vals)
            mask_list.append(mask)
        if not idxs:
            return None
        return dict(
            step_idx=jnp.asarray(np.asarray(idxs)),
            values=jnp.asarray(np.stack(vals_list)),
            mask=jnp.asarray(np.stack(mask_list)),
        )

    hd = pd.read_csv(os.path.join(data_dir, "L0001-HD.csv"))
    hd["datetime"] = pd.to_datetime(hd["datetime"])
    temp_obs = build(hd, "Depth_meter", "Water_Temperature_celsius", mass_convert=False)

    wq = pd.read_csv(os.path.join(data_dir, "L0001-WQ.csv"))
    wq["datetime"] = pd.to_datetime(wq["datetime"])
    o2_obs = build(wq[wq["variable"] == "do"], "depth", "observation", mass_convert=True)
    doc_obs = build(wq[wq["variable"] == "doc"], "depth", "observation", mass_convert=True)

    return dict(temp=temp_obs, o2=o2_obs, doc=doc_obs)


def obs_weight(obs):
    """Inverse-variance weight so temperature (~degC) and mass-unit
    O2/DOC (which can be many orders of magnitude larger) contribute
    comparably to a combined loss."""
    if obs is None:
        return 0.0
    vals = np.asarray(obs["values"])[np.asarray(obs["mask"])]
    var = float(np.var(vals)) if vals.size > 0 else 0.0
    return 1.0 / var if var > 1e-12 else 1.0


def _rmse(sim, obs):
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def _r2(sim, obs):
    """Squared Pearson correlation coefficient (the "goodness of fit"
    sense of R^2 -- distinct from NSE below, which also penalizes bias
    and scale errors, not just the correlation)."""
    if len(obs) < 2 or np.std(obs) < 1e-12 or np.std(sim) < 1e-12:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    return float(r ** 2)


def _nse(sim, obs):
    """Nash-Sutcliffe Efficiency: 1 - SS_res/SS_tot. 1 = perfect,
    0 = no better than predicting the observed mean, <0 = worse."""
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1 - np.sum((obs - sim) ** 2) / denom)


def _kge(sim, obs):
    """Kling-Gupta Efficiency: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),
    with alpha = std(sim)/std(obs) (variability ratio) and
    beta = mean(sim)/mean(obs) (bias ratio). 1 = perfect."""
    if np.std(obs) < 1e-12 or abs(np.mean(obs)) < 1e-12:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def extract_pairs(per_step, obs):
    """Flatten the masked (sim, obs) pairs actually scored by the loss,
    per variable, as plain numpy arrays -- for RMSE/NSE/KGE/R2 reporting
    (metrics are computed outside of JAX; they're diagnostics, not part
    of the differentiable loss)."""
    pairs = {}
    if obs["temp"] is not None:
        sim = np.asarray(per_step["u"][obs["temp"]["step_idx"]])
        mask = np.asarray(obs["temp"]["mask"])
        pairs["temp"] = (sim[mask], np.asarray(obs["temp"]["values"])[mask])
    if obs["o2"] is not None:
        sim = np.asarray(per_step["o2"][obs["o2"]["step_idx"]])
        mask = np.asarray(obs["o2"]["mask"])
        pairs["o2"] = (sim[mask], np.asarray(obs["o2"]["values"])[mask])
    if obs["doc"] is not None:
        sim = np.asarray(per_step["docr"][obs["doc"]["step_idx"]] + per_step["docl"][obs["doc"]["step_idx"]])
        mask = np.asarray(obs["doc"]["mask"])
        pairs["doc"] = (sim[mask], np.asarray(obs["doc"]["values"])[mask])
    return pairs


def compute_metrics(params, geometry, forcing, ice_state, init_state, obs, chunk_steps):
    """Run the model once at `params` (via the same chunked
    `simulate_truncated` used for training -- gradients aren't needed
    here, but reusing it keeps the forward trajectory identical) and
    compute RMSE/NSE/KGE/R2 for every variable that has observations,
    over exactly the (step, depth) pairs the loss function scores.

    `jax.jit`-wrapped here (rather than run eagerly): with no wrapping
    jit, every op in `simulate_truncated` dispatches one at a time,
    which is dramatically slower over a multi-thousand-step trajectory
    than one fused/optimized XLA program -- the difference that made an
    early version of this function the slow part of a calibration run.
    """
    sim_fn = jax.jit(
        lambda p: simulate_truncated(p, geometry, forcing, ice_state, init_state, chunk_steps)
    )
    per_step = sim_fn(params)
    pairs = extract_pairs(per_step, obs)
    metrics = {}
    for name, (sim, obsv) in pairs.items():
        metrics[name] = dict(
            n=len(obsv), rmse=_rmse(sim, obsv), nse=_nse(sim, obsv),
            kge=_kge(sim, obsv), r2=_r2(sim, obsv),
        )
    return metrics


def print_metrics_table(title, metrics_initial, metrics_final):
    print(f"\n{title}")
    header = f"  {'variable':10s} {'n':>5s} {'RMSE (init->cal)':>24s} {'NSE (init->cal)':>22s} " \
             f"{'KGE (init->cal)':>22s} {'R2 (init->cal)':>22s}"
    print(header)
    for name in metrics_final:
        mi, mf = metrics_initial[name], metrics_final[name]
        print(f"  {name:10s} {mf['n']:5d} "
              f"{mi['rmse']:10.4g} -> {mf['rmse']:<9.4g} "
              f"{mi['nse']:8.3f} -> {mf['nse']:<9.3f} "
              f"{mi['kge']:8.3f} -> {mf['kge']:<9.3f} "
              f"{mi['r2']:8.3f} -> {mf['r2']:<9.3f}")


def make_chunk_bounds(n_steps, chunk_steps):
    """[(start, end), ...] covering [0, n_steps) in steps of at most
    `chunk_steps` (the last chunk may be shorter)."""
    bounds = []
    start = 0
    while start < n_steps:
        end = min(start + chunk_steps, n_steps)
        bounds.append((start, end))
        start = end
    return bounds


def simulate_truncated(params, geometry, forcing, ice_state, init_state, chunk_steps):
    """Run the same physics as `run_full_model`, but split into fixed-size
    chunks with `lax.stop_gradient` applied to the carried state at each
    chunk boundary -- truncated backpropagation through time (see module
    docstring). The *forward* trajectory is bit-identical to one
    un-chunked `run_full_model` call (chunking only changes which
    state-to-state gradient paths survive backpropagation); only the
    gradient differs from an (unstable, exploding) un-chunked one.

    Returns a `per_step` dict with keys u/o2/docr/docl/pocr/pocl, each
    shape (n_steps, nx) -- same shape/keys `run_full_model` returns,
    concatenated across chunks. This does mean the whole trajectory's
    output is held in memory at once (a few hundred MB at most for this
    model's nx=64 and multi-year record); that's an acceptable trade for
    keeping the loss/metrics code identical regardless of chunking.
    """
    u0, o2_0, docr_0, docl_0, pocr_0, pocl_0 = init_state
    nx = u0.shape[0]
    state = make_initial_state_full(
        u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, **ice_state,
    )
    n_steps = forcing["Uw"].shape[0]
    bounds = make_chunk_bounds(n_steps, chunk_steps)

    # `jax.checkpoint` (gradient checkpointing / rematerialization) on the
    # per-step body: without it, differentiating through `lax.scan` needs
    # every step's internal residuals (Thomas-solve intermediates, the
    # mixing/convection loops' internals, the per-layer 5x5 MPRK solves,
    # ...) kept live for the whole chunk, which is what actually ran this
    # sandbox out of memory (~4.6 GB) at just 6 chunks of 2000 steps. With
    # `jax.checkpoint`, only the small per-step carry (the physical state)
    # is kept; the backward pass recomputes each step's forward residuals
    # on demand instead of storing them, trading ~2x extra compute for
    # memory that no longer scales with chunk length.
    @jax.checkpoint
    def body(s, forcing_t):
        new_s, _ = full_step(s, forcing_t, geometry, params)
        outputs = dict(
            u=new_s.u, o2=new_s.o2, docr=new_s.docr, docl=new_s.docl, pocr=new_s.pocr, pocl=new_s.pocl,
        )
        return new_s, outputs

    per_step_chunks = []
    for start, end in bounds:
        forcing_c = {k: v[start:end] for k, v in forcing.items()}
        state, per_step_c = lax.scan(body, state, forcing_c)
        per_step_chunks.append(per_step_c)
        # Truncate BPTT here: gradients flow freely *within* the chunk
        # just completed, but not from later chunks back into it.
        state = jax.tree_util.tree_map(lax.stop_gradient, state)

    return {k: jnp.concatenate([c[k] for c in per_step_chunks], axis=0) for k in per_step_chunks[0]}


def make_loss_fn(base_params, geometry, forcing, ice_state, init_state, obs, weights, chunk_steps):
    def loss_fn(theta_log):
        params = dict(base_params)
        params.update({name: jnp.exp(v) for name, v in theta_log.items()})
        per_step = simulate_truncated(params, geometry, forcing, ice_state, init_state, chunk_steps)
        total = 0.0
        if obs["temp"] is not None:
            sim = per_step["u"][obs["temp"]["step_idx"]]
            err2 = (sim - obs["temp"]["values"]) ** 2 * obs["temp"]["mask"]
            total = total + weights["temp"] * jnp.sum(err2) / jnp.maximum(jnp.sum(obs["temp"]["mask"]), 1.0)
        if obs["o2"] is not None:
            sim = per_step["o2"][obs["o2"]["step_idx"]]
            err2 = (sim - obs["o2"]["values"]) ** 2 * obs["o2"]["mask"]
            total = total + weights["o2"] * jnp.sum(err2) / jnp.maximum(jnp.sum(obs["o2"]["mask"]), 1.0)
        if obs["doc"] is not None:
            sim = per_step["docr"][obs["doc"]["step_idx"]] + per_step["docl"][obs["doc"]["step_idx"]]
            err2 = (sim - obs["doc"]["values"]) ** 2 * obs["doc"]["mask"]
            total = total + weights["doc"] * jnp.sum(err2) / jnp.maximum(jnp.sum(obs["doc"]["mask"]), 1.0)
        return total

    return loss_fn


def make_per_variable_loss_fn(base_params, geometry, forcing, ice_state, init_state, obs, chunk_steps, var_names):
    """Like `make_loss_fn`, but returns a length-`len(var_names)` vector of
    *unweighted* per-variable mean squared errors instead of one combined
    weighted scalar -- one run of the model feeds `jax.jacrev` (a handful
    of extra backward passes, not a separate simulation per variable), used
    for the per-variable sensitivity ranking in `main()`."""
    def fn(theta_log):
        params = dict(base_params)
        params.update({name: jnp.exp(v) for name, v in theta_log.items()})
        per_step = simulate_truncated(params, geometry, forcing, ice_state, init_state, chunk_steps)
        losses = []
        for name in var_names:
            if name == "temp":
                sim = per_step["u"][obs["temp"]["step_idx"]]
                err2 = (sim - obs["temp"]["values"]) ** 2 * obs["temp"]["mask"]
                losses.append(jnp.sum(err2) / jnp.maximum(jnp.sum(obs["temp"]["mask"]), 1.0))
            elif name == "o2":
                sim = per_step["o2"][obs["o2"]["step_idx"]]
                err2 = (sim - obs["o2"]["values"]) ** 2 * obs["o2"]["mask"]
                losses.append(jnp.sum(err2) / jnp.maximum(jnp.sum(obs["o2"]["mask"]), 1.0))
            elif name == "doc":
                sim = per_step["docr"][obs["doc"]["step_idx"]] + per_step["docl"][obs["doc"]["step_idx"]]
                err2 = (sim - obs["doc"]["values"]) ** 2 * obs["doc"]["mask"]
                losses.append(jnp.sum(err2) / jnp.maximum(jnp.sum(obs["doc"]["mask"]), 1.0))
        return jnp.stack(losses)

    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=".")
    parser.add_argument("--steps", type=int, default=None,
                         help="total simulation window in hourly steps (default: the full "
                              "record per run_config.csv's start_time/end_time)")
    parser.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS,
                         help=f"gradients are truncated every this many steps (default "
                              f"{DEFAULT_CHUNK_STEPS} -- see module docstring on truncated "
                              "BPTT; must stay well inside the ~7500-8000-step gradient horizon "
                              "documented in jax_lakeModel_functions.py)")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Adam iterations")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="log-space Adam learning rate")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                         help="number of parameters to calibrate from the COMBINED ranking "
                              "(only used with --select-mode combined)")
    parser.add_argument("--topk-per-variable", type=int, default=DEFAULT_TOPK_PER_VARIABLE,
                         help="number of parameters to take from EACH variable's own sensitivity "
                              "ranking (temp/o2/doc); the union across present variables is "
                              "calibrated (default select mode -- see --select-mode)")
    parser.add_argument("--select-mode", choices=["per-variable", "combined"], default="per-variable",
                         help="'per-variable' (default): calibrate the union of each present "
                              "variable's top --topk-per-variable most sensitive parameters. "
                              "'combined': calibrate the top --topk parameters from the single "
                              "combined-loss ranking (the original behavior)")
    parser.add_argument("--params", type=str, default=None,
                         help="comma-separated parameter names to calibrate directly, "
                              "bypassing the sensitivity screen")
    parser.add_argument("--sensitivity-only", action="store_true",
                         help="only run and print the sensitivity screen, skip optimization")
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
    base_params = default_params(model_params, ice_and_snow)

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

    n_chunks = len(make_chunk_bounds(n_steps, args.chunk_steps))
    print(f"Loading observations and restricting to the {n_steps}-step "
          f"({n_steps * dt / 86400:.0f}-day) simulation window "
          f"({n_chunks} chunk(s) of up to {args.chunk_steps} steps for gradient truncation)...")
    obs = load_observations("./", depth, volume, desired_start, step_times, dt)
    for name in ("temp", "o2", "doc"):
        n = int(obs[name]["step_idx"].shape[0]) if obs[name] is not None else 0
        print(f"  {name}: {n} profile date(s) in window")
    weights = {name: obs_weight(obs[name]) for name in ("temp", "o2", "doc")}

    if all(obs[name] is None for name in ("temp", "o2", "doc")):
        print("No observations fall inside the simulation window -- nothing to calibrate against. "
              "Try passing --steps explicitly (or a larger value), or check that "
              "run_config.csv's start_time overlaps L0001-HD.csv/L0001-WQ.csv's date range.")
        return

    loss_fn = make_loss_fn(base_params, geometry, forcing, ice_state, init_state, obs, weights, args.chunk_steps)

    # --- sensitivity screen ---
    candidates = [p for p in CANDIDATE_PARAMS if base_params.get(p) is not None]
    theta_log0 = {name: jnp.log(jnp.asarray(float(base_params[name]))) for name in candidates}

    print(f"\nRunning sensitivity screen over {len(candidates)} candidate parameters "
          f"({n_steps} steps across {n_chunks} chunk(s) -- this runs the model once, jitted)...")
    t0 = time.time()
    screen_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    val0, grad0 = screen_grad_fn(theta_log0)
    t1 = time.time()
    print(f"  baseline loss = {float(val0):.6g}  (screen took {t1 - t0:.1f}s)")

    elasticity = {}
    for name in candidates:
        g = float(grad0[name])  # d(loss)/d(log theta) = theta * d(loss)/d(theta) -- already the elasticity
        elasticity[name] = abs(g) if np.isfinite(g) else -1.0  # non-finite -> sort last, flagged below

    ranked = sorted(elasticity.items(), key=lambda kv: kv[1], reverse=True)
    print("\nCombined sensitivity ranking (|d(combined weighted loss)/d(log param)|, i.e. elasticity):")
    for name, e in ranked:
        flag = "  [NaN/Inf gradient -- excluded]" if e < 0 else ""
        print(f"  {name:28s} {e:12.4g}  (value={float(base_params[name]):.4g}){flag}")

    # --- per-variable sensitivity screen ---
    # Same forward pass, but ranks each present variable's *own* unweighted
    # loss separately (jax.jacrev over the stacked per-variable losses --
    # one extra backward pass per present variable, not a separate model
    # run), so a parameter that matters a lot to one variable can't be
    # buried by the combined ranking just because a differently-scaled
    # variable dominates the combined loss. See module docstring.
    present_vars = [name for name in ("temp", "o2", "doc") if obs[name] is not None]
    per_var_loss_fn = make_per_variable_loss_fn(
        base_params, geometry, forcing, ice_state, init_state, obs, args.chunk_steps, present_vars,
    )
    print(f"\nRunning per-variable sensitivity screen over {present_vars} "
          f"({len(present_vars)} extra backward pass(es), same forward pass as above)...")
    t0 = time.time()
    jac_fn = jax.jit(jax.jacrev(per_var_loss_fn))
    jac = jac_fn(theta_log0)
    t1 = time.time()
    print(f"  (per-variable screen took {t1 - t0:.1f}s)")

    per_var_ranked = {}
    for i, var in enumerate(present_vars):
        elast_var = {}
        for name in candidates:
            g = float(jac[name][i])
            elast_var[name] = abs(g) if np.isfinite(g) else -1.0
        per_var_ranked[var] = sorted(elast_var.items(), key=lambda kv: kv[1], reverse=True)

    for var in present_vars:
        print(f"\nTop {args.topk_per_variable} sensitivity ranking for '{var}' "
              f"(|d(loss_{var})/d(log param)|, unweighted, this variable only):")
        for name, e in per_var_ranked[var][: args.topk_per_variable]:
            flag = "  [NaN/Inf gradient -- excluded]" if e < 0 else ""
            print(f"  {name:28s} {e:12.4g}  (value={float(base_params[name]):.4g}){flag}")

    if args.sensitivity_only:
        return

    if args.params:
        selected = [p.strip() for p in args.params.split(",") if p.strip()]
        missing = [p for p in selected if p not in candidates]
        if missing:
            raise SystemExit(f"--params entries not in CANDIDATE_PARAMS/model_params.csv: {missing}")
    elif args.select_mode == "combined":
        selected = [name for name, e in ranked if e >= 0][: args.topk]
    else:
        # Union of each present variable's own top --topk-per-variable,
        # order preserved by first appearance (temp's picks first, then
        # any new names from o2, then doc), duplicates dropped.
        selected = []
        for var in present_vars:
            for name, e in per_var_ranked[var][: args.topk_per_variable]:
                if e >= 0 and name not in selected:
                    selected.append(name)
    print(f"\nCalibrating ({args.select_mode} selection): {selected}")

    theta_log = {name: theta_log0[name] for name in selected}
    x0 = {name: float(base_params[name]) for name in selected}
    log_lo = {name: jnp.log(jnp.asarray(min(0.1 * x0[name], 10 * x0[name]))) for name in selected}
    log_hi = {name: jnp.log(jnp.asarray(max(0.1 * x0[name], 10 * x0[name]))) for name in selected}

    opt = optax.adam(args.lr)
    opt_state = opt.init(theta_log)
    opt_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    print(f"\nRunning {args.iters} Adam iterations ({n_steps} steps/iteration across {n_chunks} "
          "gradient-truncated chunk(s), jitted -- first iteration includes compile time)...")
    best_loss, best_theta_log = float(val0), dict(theta_log0)
    history = [float(val0)]
    for it in range(args.iters):
        t0 = time.time()
        loss_val, grads = opt_grad_fn(theta_log)
        non_finite = any(not bool(jnp.all(jnp.isfinite(v))) for v in grads.values()) or not np.isfinite(float(loss_val))
        if non_finite:
            print(f"  iter {it}: non-finite loss/gradient encountered -- stopping early "
                  "(see jax_lakeModel_functions.py's module docstring on the gradient horizon; "
                  "try a smaller --chunk-steps). Keeping the best result found so far.")
            break
        updates, opt_state = opt.update(grads, opt_state, theta_log)
        theta_log = optax.apply_updates(theta_log, updates)
        theta_log = {name: jnp.clip(v, log_lo[name], log_hi[name]) for name, v in theta_log.items()}
        t1 = time.time()
        loss_f = float(loss_val)
        history.append(loss_f)
        if loss_f < best_loss:
            best_loss, best_theta_log = loss_f, dict(theta_log)
        print(f"  iter {it:3d}: loss={loss_f:.6g}  ({t1 - t0:.1f}s)")

    print(f"\nBest loss: {best_loss:.6g}  (baseline was {float(val0):.6g}, "
          f"{100 * (1 - best_loss / float(val0)):.1f}% reduction)")
    print("\nCalibrated parameters:")
    print(f"  {'parameter':28s} {'initial':>14s} {'calibrated':>14s} {'change':>10s}")
    results = {}
    for name in selected:
        v0 = x0[name]
        v1 = float(jnp.exp(best_theta_log[name]))
        results[name] = dict(initial=v0, calibrated=v1)
        pct = 100 * (v1 / v0 - 1) if v0 != 0 else float("nan")
        print(f"  {name:28s} {v0:14.6g} {v1:14.6g} {pct:9.1f}%")

    # --- per-variable RMSE/NSE/KGE/R2, initial vs. calibrated params ---
    params_final = dict(base_params)
    params_final.update({name: float(jnp.exp(v)) for name, v in best_theta_log.items() if name in selected})
    metrics_initial = compute_metrics(base_params, geometry, forcing, ice_state, init_state, obs, args.chunk_steps)
    metrics_final = compute_metrics(params_final, geometry, forcing, ice_state, init_state, obs, args.chunk_steps)
    print_metrics_table(
        "Evaluation metrics (computed at the same (step, depth) pairs the loss scores; "
        "NSE/KGE/R2: 1.0 = perfect fit):",
        metrics_initial, metrics_final,
    )

    out_path = os.path.join(os.getcwd(), "calibration_result.csv")
    pd.DataFrame(results).T.rename_axis("parameter").to_csv(out_path)
    print(f"\nSaved calibrated parameter table to {out_path}")

    metrics_rows = []
    for name in metrics_final:
        row = dict(variable=name, n=metrics_final[name]["n"])
        for m in ("rmse", "nse", "kge", "r2"):
            row[f"{m}_initial"] = metrics_initial[name][m]
            row[f"{m}_calibrated"] = metrics_final[name][m]
        metrics_rows.append(row)
    metrics_path = os.path.join(os.getcwd(), "calibration_metrics.csv")
    pd.DataFrame(metrics_rows).set_index("variable").to_csv(metrics_path)
    print(f"Saved evaluation metrics (RMSE/NSE/KGE/R2, initial vs. calibrated) to {metrics_path}")


if __name__ == "__main__":
    main()
