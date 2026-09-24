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
        [--lr LR] [--topk K] [--params p1,p2,...] [--variables temp,o2,doc,poc] [--sensitivity-only]

    `--params p1,p2,...` skips the sensitivity screen entirely (both the
    combined and per-variable rankings) and calibrates exactly the named
    parameters directly -- use this when you already know which parameters
    you want to tune and don't want to pay for screening. Omit `--params`
    to run the screen and auto-select as described below. `--params` together
    with `--sensitivity-only` is a no-op (nothing left to screen for) and
    exits immediately without calibrating.

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
needs no conversion. POC ("particulate organic carbon") is handled exactly
the same way as DOC: compared against `pocr + pocl` combined, converted
from mg/L the same `obs_mass = obs_concentration_mgL * volume` way. Unlike
temp/O2/DOC, POC observations are optional at the dataset level, not just
possibly absent from a given window -- Ravn's `L0001-WQ.csv` has no "poc"
rows at all, while Mendota's `ME_obs_depths3_wpoc.csv` does. This falls out
for free from the same "`None` if no matching rows produce a usable
profile" handling every other variable already gets in `load_observations()`
below: a dataset without POC observations simply gets `obs["poc"] is None`,
which drops out of the loss, the sensitivity screens, `--variables`
validation, and the final metrics table exactly like an out-of-window
temp/O2/DOC would, with zero dataset-specific branching anywhere in this
script.

Observation matching: temperature/O2/DOC observations (read from
`run_config.csv`'s `u_ini_file`/`wq_ini_file` -- `L0001-HD.csv`/
`L0001-WQ.csv` for Ravn -- via `observation_data.py`, shared with every
other script in this project) are long-format, irregular in both time
(roughly monthly profile dates) and depth. For each profile date, the
observed depths are linearly interpolated onto the model's depth grid
(`processBased_lakeModel_functions.get_hypsography`'s cell-centered
`depth` array); grid points beyond the observed depth range are masked
out rather than extrapolated (matching the spirit of
`initial_profile`/`wq_initial_profile`'s own "extend to lake max depth by
repeating the deepest observation" rule would be an unjustified
assumption for calibration targets, so we simply don't score those
points). Each profile date is mapped to the nearest simulation step via
`round((obs_datetime - start_date).total_seconds() / dt)`.

The high-frequency buoy thermistor-chain record (`ravn_2023.json`/
`ravn_2024.json`, plus `sensor_level_buoy.sen` for the sensor->depth
mapping), where available, is not read directly by this script (or by
`plot_output.py`): run `integrate_buoy_temperature.py` once first to fold
it into `u_ini_file` itself (see that script's module docstring and
`observation_data.load_buoy_temperature()`) -- it is a fixed-depth
thermistor chain (18 sensors) natively sampled roughly every 15 minutes,
averaged down to hourly bins, reshaped to the same long-format schema as
`u_ini_file`, and merged in as one additional "profile" per hourly
timestamp. This can add hundreds to thousands of extra profile dates
within its coverage window, all scored the same way as the sparser
manual profiles, with no buoy-specific code needed in this script at all.

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

Chunking is implemented as a *nested* `lax.scan` -- an outer scan over
chunks wrapping the existing per-step scan within each chunk (padding the
forcing series up to a whole number of chunks first, then trimming the
padding back off the output) -- rather than a plain Python `for` loop over
chunks. This matters for compile time, not just runtime: a Python loop is
unrolled at trace time, so `jax.jit` would bake one full copy of the
per-chunk computation into the compiled program *for every chunk*, making
compiled-program size (and compile time) grow with the total simulation
length. At a handful of chunks that's unnoticeable; at the default full
multi-year record (~22 chunks of 2000 steps), it made compilation itself
slow to the point of stalling or failing outright -- before ever reaching
the actual simulation. The nested-scan version compiles the chunk-body
computation exactly once and reuses it via the scan's own loop mechanism,
so compile time/size stay roughly constant regardless of how many chunks
the record needs. Numerically and for gradient-truncation purposes it is
identical to the old loop -- only how it gets compiled changed.

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
because that loss is dominated by a differently-scaled variable. Note
that "one extra backward pass per present variable" is still real cost,
not a free byproduct of the one forward pass: with all three of
temp/O2/DOC present, the per-variable screen genuinely takes roughly 3x
as long as the combined screen above, since `jacrev` runs one backward
pass per output component. Both screens print how long they actually
took (blocking on the result first -- `jax.jit` dispatches
asynchronously, so timing the call itself without forcing completion
understates the real cost, sometimes badly).

By default (`--select-mode per-variable`), the parameters actually
calibrated are the *union* of each present variable's top
`--topk-per-variable` (default 5) most sensitive parameters -- so
whichever of temp/O2/DOC have observations in the window, their own most
sensitive parameters are guaranteed to be included, not just whichever
parameters happen to dominate the combined loss. `--select-mode combined`
restores the old behavior (top `--topk`, default 6, from the combined
ranking only). `--params` bypasses both screens *entirely* -- they are not
run at all in that case, not merely ignored -- and calibrates exactly the
named parameters. This is the fast path for "I already know which
parameters I want to tune": it skips the extra forward/backward passes
the screens cost (see above) and goes straight to the Adam loop. Either
way, selected parameters are optimized together with
Adam (`optax`), in log-space (so a single learning rate means "N%
relative step" for every parameter regardless of its raw scale, e.g.
km ~ 1e-6 vs. theta_r ~ 1.2), under a box constraint keeping each
parameter within [0.1x, 10x] of its starting value.

`--iters` is a ceiling, not a target: by default the Adam loop stops early
once the loss has gone `--early-stop-patience` (default 5) consecutive
iterations without improving by more than `--early-stop-tol` (default
1e-4, relative to the best loss seen so far) -- Adam's own step-to-step
noise means "no improvement over the last step" would trigger constantly,
so this tracks stalling against the *best* loss seen, not just the
previous iteration. `--early-stop-patience 0` disables this and always
runs the full `--iters`. Either way the best iterate found (not
necessarily the last one) is what gets reported/saved.

Restricting which variables count towards calibration (`--variables`):
by default every variable with observations in the window (temp/O2/DOC/POC,
whichever are present -- POC only for datasets whose `wq_ini_file` has
"poc" rows, e.g. Mendota but not Ravn) contributes to the loss and to the
sensitivity screens. `--variables temp` or `--variables o2,temp` (etc.)
restricts this to a named subset -- e.g. "only calibrate against
temperature" even though O2/DOC/POC observations also exist in the window. The excluded
variable(s) are dropped entirely from the loss (so their gradient
contributes nothing) and from the per-variable sensitivity screen/
selection, but -- since they cost nothing extra to compute, being read
from the same forward simulation -- they are still reported in the final
evaluation metrics table for reference, marked as "(not targeted)" so
it's clear they didn't influence the fit. Requesting a variable with no
observations in the window is an error (nothing to calibrate against for
it), rather than silently ignored.

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
from observation_data import (
    load_temperature_dataframe, load_water_quality_dataframe,
    load_buoy_temperature,  # noqa: F401 -- re-exported for backward-compat imports of this name
)


# `--steps` defaults to None (the full available record -- safe now thanks
# to chunked/truncated backprop). `--chunk-steps` is what needs to stay
# safely inside the empirically-found ~7500-8000 hourly-step gradient
# horizon documented in jax_lakeModel_functions.py (see module docstring);
# 2000 leaves a healthy margin.
DEFAULT_CHUNK_STEPS = 2000   # ~83 days
DEFAULT_ITERS = 20
DEFAULT_LR = 0.05       # log-space Adam step ~= 5% relative parameter change
DEFAULT_EARLY_STOP_PATIENCE = 5   # 0 disables early stopping (always run --iters)
DEFAULT_EARLY_STOP_TOL = 1e-4     # relative loss improvement below this doesn't reset patience
DEFAULT_TOPK = 6
DEFAULT_TOPK_PER_VARIABLE = 5

# model_params.csv entries that default_params() actually threads into the
# simulation, restricted to ones that are safe/sensible to tweak in
# isolation. Excluded: physical constants (g, sigma), the OC-partitioning
# fractions prop_oc_docr/docl/pocr/pocl (must sum sensibly across pools,
# not safe to move independently), and beta (not a model_params.csv column
# at all -- run_wq_model never lets the caller override it either).
# "oc_load_factor" is a JAX-only addition (see default_params()/full_step()
# in jax_lakeModel_functions.py): a multiplier on oc_load_file.csv's "oc"
# concentration column, treating the overall scale of the carbon-loading
# boundary condition as an unknown to calibrate rather than a fixed input.
CANDIDATE_PARAMS = [
    "km", "weight_kz", "Cd", "denThresh",
    "kd_light", "light_water", "light_doc", "light_poc",
    "sw_factor", "at_factor", "turb_factor", "wind_factor", "Hgeo",
    "theta_r", "theta_npp", "k_half",
    "resp_docr", "resp_docl", "resp_pocr", "resp_pocl",
    "settling_rate_labile", "settling_rate_refractory",
    "f_sod", "d_thick", "meltP", "p2", "eps", "emissivity",
    "oc_load_factor",
]


def load_observations(data_dir, depth, volume, start_date, step_times, dt, run_config=None):
    """Load temperature/O2/DOC observations, restrict to the simulation
    window, and build per-variable dicts of (step_idx, values, mask)
    arrays on the model's depth grid. `values` are in the model's own
    units: degC for temperature, mass (obs_mgL * volume) for o2/doc.
    Returns a dict with keys "temp", "o2", "doc" (each None if no
    observations fall in the simulation window).

    Temperature comes from `run_config`'s `u_ini_file` (falling back to
    `L0001-HD.csv` if `run_config` isn't given) and O2/DOC from
    `wq_ini_file` (falling back to `L0001-WQ.csv`) -- see
    `observation_data.load_temperature_dataframe()`/
    `load_water_quality_dataframe()`. If the high-frequency buoy
    thermistor-chain record should be included, run
    `integrate_buoy_temperature.py` once first to fold `ravn_2023.json`/
    `ravn_2024.json` into `u_ini_file` directly -- each of its hourly
    timestamps then becomes its own "profile" (matching `u_ini_file`'s
    per-datetime grouping) automatically, with no buoy-specific code
    needed here any more. This can add hundreds to thousands of extra
    profile dates within the buoy's coverage window, all scored the same
    way as the sparser manual profiles."""
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

    hd = load_temperature_dataframe(data_dir, run_config)
    temp_obs = build(hd, "Depth_meter", "Water_Temperature_celsius", mass_convert=False) if hd is not None else None

    wq = load_water_quality_dataframe(data_dir, run_config)
    if wq is not None:
        o2_obs = build(wq[wq["variable"] == "do"], "depth", "observation", mass_convert=True)
        doc_obs = build(wq[wq["variable"] == "doc"], "depth", "observation", mass_convert=True)
        # POC ("particulate organic carbon") is only present for some
        # datasets (e.g. Mendota's ME_obs_depths3_wpoc.csv has a "poc" row
        # per sample; Ravn's L0001-WQ.csv has no such rows at all) -- absent
        # entirely is the normal case, not an error, so this falls out to
        # None below exactly like temp/o2/doc do when their own source rows
        # are missing. Compared against pocr+pocl combined (see extract_pairs/
        # make_loss_fn), mirroring how doc is compared against docr+docl.
        poc_obs = build(wq[wq["variable"] == "poc"], "depth", "observation", mass_convert=True)
    else:
        o2_obs = doc_obs = poc_obs = None

    return dict(temp=temp_obs, o2=o2_obs, doc=doc_obs, poc=poc_obs)


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
    if obs["poc"] is not None:
        sim = np.asarray(per_step["pocr"][obs["poc"]["step_idx"]] + per_step["pocl"][obs["poc"]["step_idx"]])
        mask = np.asarray(obs["poc"]["mask"])
        pairs["poc"] = (sim[mask], np.asarray(obs["poc"]["values"])[mask])
    return pairs


def compute_metrics(sim_fn, params, obs):
    """Run the model once at `params` via the already-`jax.jit`-compiled
    `sim_fn` (built once in `main()` from `simulate_truncated` and reused
    for both the initial and calibrated parameters -- same compiled
    program either way, since only the parameter *values* differ, not
    their shapes/dtypes/keys, so the second call is a cache hit rather
    than a second compile) and compute RMSE/NSE/KGE/R2 for every variable
    that has observations, over exactly the (step, depth) pairs the loss
    function scores.
    """
    per_step = sim_fn(params)
    pairs = extract_pairs(per_step, obs)
    metrics = {}
    for name, (sim, obsv) in pairs.items():
        metrics[name] = dict(
            n=len(obsv), rmse=_rmse(sim, obsv), nse=_nse(sim, obsv),
            kge=_kge(sim, obsv), r2=_r2(sim, obsv),
        )
    return metrics


def print_metrics_table(title, metrics_initial, metrics_final, active_vars=None):
    """`active_vars`, if given, marks variables that did NOT contribute to
    the loss (e.g. excluded via --variables) as "(not targeted)" -- they
    are still reported here since the forward simulation already produces
    them at no extra cost, but they had no gradient influence on the fit."""
    print(f"\n{title}")
    header = f"  {'variable':10s} {'n':>5s} {'RMSE (init->cal)':>24s} {'NSE (init->cal)':>22s} " \
             f"{'KGE (init->cal)':>22s} {'R2 (init->cal)':>22s}"
    print(header)
    for name in metrics_final:
        mi, mf = metrics_initial[name], metrics_final[name]
        flag = "  (not targeted)" if active_vars is not None and name not in active_vars else ""
        print(f"  {name:10s} {mf['n']:5d} "
              f"{mi['rmse']:10.4g} -> {mf['rmse']:<9.4g} "
              f"{mi['nse']:8.3f} -> {mf['nse']:<9.3f} "
              f"{mi['kge']:8.3f} -> {mf['kge']:<9.3f} "
              f"{mi['r2']:8.3f} -> {mf['r2']:<9.3f}{flag}")


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

    Implemented as a *nested* `lax.scan` (outer scan over chunks, wrapping
    the inner per-step scan within a chunk) rather than a Python loop over
    chunks, so the compiled program is the same size regardless of how
    many chunks the record needs -- see module docstring. `chunk_steps`
    need not evenly divide the record: the forcing series is padded up to
    a whole number of chunks first (by repeating its last real row, so the
    padded tail is physically boring rather than an artificial spike) and
    the padding is trimmed back off the output before returning, invisibly
    to every caller.

    Returns a `per_step` dict with keys u/o2/docr/docl/pocr/pocl, each
    shape (n_steps, nx) -- same shape/keys `run_full_model` returns. This
    does mean the whole trajectory's output is held in memory at once (a
    few hundred MB at most for this model's nx=64 and multi-year record);
    that's an acceptable trade for keeping the loss/metrics code identical
    regardless of chunking.
    """
    u0, o2_0, docr_0, docl_0, pocr_0, pocl_0 = init_state
    nx = u0.shape[0]
    state = make_initial_state_full(
        u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, **ice_state,
    )
    n_steps = forcing["Uw"].shape[0]
    n_chunks = -(-n_steps // chunk_steps)  # ceil division
    n_padded = n_chunks * chunk_steps
    pad = n_padded - n_steps

    if pad > 0:
        forcing_padded = {
            k: jnp.concatenate([v, jnp.broadcast_to(v[-1], (pad,) + v.shape[1:])], axis=0)
            for k, v in forcing.items()
        }
    else:
        forcing_padded = forcing

    forcing_chunked = jax.tree_util.tree_map(
        lambda v: v.reshape((n_chunks, chunk_steps) + v.shape[1:]), forcing_padded,
    )

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
    def step_body(s, forcing_t):
        new_s, _ = full_step(s, forcing_t, geometry, params)
        outputs = dict(
            u=new_s.u, o2=new_s.o2, docr=new_s.docr, docl=new_s.docl, pocr=new_s.pocr, pocl=new_s.pocl,
        )
        return new_s, outputs

    # Checkpointed again at the chunk level, for the same reason: without
    # it, backpropagating through the *outer* scan would keep every
    # chunk's forward residuals live at once (scaling with chunk COUNT,
    # i.e. with total simulation length again, just one level up); with
    # it, only the small state carry is kept between chunks, and each
    # chunk's own (already-checkpointed) inner scan is recomputed on
    # demand during the backward pass.
    @jax.checkpoint
    def chunk_body(s, forcing_chunk):
        s, per_step_c = lax.scan(step_body, s, forcing_chunk)
        # Truncate BPTT here: gradients flow freely *within* the chunk
        # just completed, but not from later chunks back into it.
        s = jax.tree_util.tree_map(lax.stop_gradient, s)
        return s, per_step_c

    _, per_step_chunked = lax.scan(chunk_body, state, forcing_chunked)
    return {
        k: v.reshape((n_padded,) + v.shape[2:])[:n_steps]
        for k, v in per_step_chunked.items()
    }


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
        if obs["poc"] is not None:
            sim = per_step["pocr"][obs["poc"]["step_idx"]] + per_step["pocl"][obs["poc"]["step_idx"]]
            err2 = (sim - obs["poc"]["values"]) ** 2 * obs["poc"]["mask"]
            total = total + weights["poc"] * jnp.sum(err2) / jnp.maximum(jnp.sum(obs["poc"]["mask"]), 1.0)
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
            elif name == "poc":
                sim = per_step["pocr"][obs["poc"]["step_idx"]] + per_step["pocl"][obs["poc"]["step_idx"]]
                err2 = (sim - obs["poc"]["values"]) ** 2 * obs["poc"]["mask"]
                losses.append(jnp.sum(err2) / jnp.maximum(jnp.sum(obs["poc"]["mask"]), 1.0))
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
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                         help="maximum Adam iterations (may stop earlier -- see --early-stop-patience)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="log-space Adam learning rate")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE,
                         help=f"stop the Adam loop once this many consecutive iterations fail to "
                              f"improve the loss by more than --early-stop-tol (default "
                              f"{DEFAULT_EARLY_STOP_PATIENCE}); 0 disables early stopping and always "
                              f"runs the full --iters")
    parser.add_argument("--early-stop-tol", type=float, default=DEFAULT_EARLY_STOP_TOL,
                         help=f"relative loss improvement (over the best loss so far) below which an "
                              f"iteration counts as non-improving for --early-stop-patience (default "
                              f"{DEFAULT_EARLY_STOP_TOL})")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                         help="number of parameters to calibrate from the COMBINED ranking "
                              "(only used with --select-mode combined)")
    parser.add_argument("--topk-per-variable", type=int, default=DEFAULT_TOPK_PER_VARIABLE,
                         help="number of parameters to take from EACH variable's own sensitivity "
                              "ranking (temp/o2/doc/poc); the union across present variables is "
                              "calibrated (default select mode -- see --select-mode)")
    parser.add_argument("--select-mode", choices=["per-variable", "combined"], default="per-variable",
                         help="'per-variable' (default): calibrate the union of each present "
                              "variable's top --topk-per-variable most sensitive parameters. "
                              "'combined': calibrate the top --topk parameters from the single "
                              "combined-loss ranking (the original behavior)")
    parser.add_argument("--params", type=str, default=None,
                         help="comma-separated parameter names to calibrate directly, "
                              "bypassing the sensitivity screen")
    parser.add_argument("--variables", type=str, default=None,
                         help="comma-separated subset of temp,o2,doc,poc to calibrate against "
                              "(default: all of temp/o2/doc/poc that have observations in the "
                              "window -- poc only if the configured wq_ini_file has 'poc' rows). "
                              "Excluded variables are dropped from the loss and "
                              "sensitivity screens entirely, but still reported in the final "
                              "evaluation metrics for reference.")
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
        hypsofile=run_config["hypso_ini_file"], dx=dx, nx=nx, outflow_depth=float(lake_config["outflow_depth"]),
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
    obs = load_observations("./", depth, volume, desired_start, step_times, dt, run_config=run_config)
    for name in ("temp", "o2", "doc", "poc"):
        n = int(obs[name]["step_idx"].shape[0]) if obs[name] is not None else 0
        print(f"  {name}: {n} profile date(s) in window")
    weights = {name: obs_weight(obs[name]) for name in ("temp", "o2", "doc", "poc")}

    present_vars = [name for name in ("temp", "o2", "doc", "poc") if obs[name] is not None]

    if args.variables:
        requested = [v.strip() for v in args.variables.split(",") if v.strip()]
        unknown = [v for v in requested if v not in ("temp", "o2", "doc", "poc")]
        if unknown:
            raise SystemExit(f"--variables entries must be from temp,o2,doc,poc: unknown {unknown}")
        missing = [v for v in requested if v not in present_vars]
        if missing:
            raise SystemExit(
                f"--variables requested {missing}, but there are no observations for "
                f"{missing} in this {n_steps}-step window -- nothing to calibrate against "
                "for them. Try a different --steps window, or drop them from --variables."
            )
        # canonical temp/o2/doc/poc order regardless of how the user typed --variables
        active_vars = [name for name in ("temp", "o2", "doc", "poc") if name in requested]
    else:
        active_vars = present_vars

    if not active_vars:
        print("No observations fall inside the simulation window -- nothing to calibrate against. "
              "Try passing --steps explicitly (or a larger value), or check that "
              "run_config.csv's start_time overlaps L0001-HD.csv/L0001-WQ.csv's date range.")
        return

    excluded_vars = [v for v in present_vars if v not in active_vars]
    print(f"  calibrating against: {active_vars}"
          + (f"  (present but excluded by --variables: {excluded_vars})" if excluded_vars else ""))

    # Variables left out of --variables are dropped from the loss entirely
    # (rather than merely down-weighted) by hiding their observations from
    # make_loss_fn/make_per_variable_loss_fn -- the full, unfiltered `obs`
    # is still used for the final evaluation metrics, so an excluded
    # variable is still reported (for reference) even though it never
    # contributed a gradient.
    obs_for_loss = {name: (obs[name] if name in active_vars else None) for name in ("temp", "o2", "doc", "poc")}

    loss_fn = make_loss_fn(base_params, geometry, forcing, ice_state, init_state, obs_for_loss, weights, args.chunk_steps)

    candidates = [p for p in CANDIDATE_PARAMS if base_params.get(p) is not None]
    theta_log0 = {name: jnp.log(jnp.asarray(float(base_params[name]))) for name in candidates}
    # Shared across the screen (if run) and the Adam loop below (rather than
    # two separately-constructed jax.jit wrappers around the same loss_fn) --
    # they're called with different-sized parameter dicts (all candidates
    # vs. the selected subset) so each still compiles once regardless, but
    # there's no reason to build two wrapper objects for one function.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    if args.params:
        # Bypass the sensitivity screen entirely: its only purpose is to
        # *choose* which parameters to calibrate, which is moot when the
        # caller already knows -- skipping it saves the (up to 1 + len(
        # active_vars)) extra backward passes the screen below runs, at the
        # cost of not seeing the ranking. A single forward+backward pass is
        # still run, just to report a baseline loss for the "before/after"
        # comparison at the end.
        selected = [p.strip() for p in args.params.split(",") if p.strip()]
        missing = [p for p in selected if p not in candidates]
        if missing:
            raise SystemExit(f"--params entries not in CANDIDATE_PARAMS/model_params.csv: {missing}")
        if args.sensitivity_only:
            print("--sensitivity-only has no effect together with --params (there is nothing "
                  "left to screen for once the parameters to calibrate are given explicitly) "
                  "-- exiting without calibrating. Drop --params to run the sensitivity screen.")
            return
        print(f"\n--params given: skipping the sensitivity screen, calibrating {selected} directly.")
        t0 = time.time()
        val0, _ = grad_fn(theta_log0)
        jax.block_until_ready(val0)  # see note below on async dispatch
        print(f"  baseline loss = {float(val0):.6g}  ({time.time() - t0:.1f}s)")
    else:
        # --- sensitivity screen ---
        print(f"\nRunning sensitivity screen over {len(candidates)} candidate parameters "
              f"({n_steps} steps across {n_chunks} chunk(s) -- this runs the model once, jitted)...")
        t0 = time.time()
        val0, grad0 = grad_fn(theta_log0)
        jax.block_until_ready((val0, grad0))  # jax.jit dispatches asynchronously --
        # without forcing completion here, this timing (and every other one in this
        # script) would only measure dispatch, not the actual compute, making a
        # slow call look deceptively fast right up until something later actually
        # reads a value and blocks for the real duration.
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
        per_var_loss_fn = make_per_variable_loss_fn(
            base_params, geometry, forcing, ice_state, init_state, obs_for_loss, args.chunk_steps, active_vars,
        )
        print(f"\nRunning per-variable sensitivity screen over {active_vars} "
              f"({len(active_vars)} extra backward pass(es), same forward pass as above)...")
        t0 = time.time()
        jac_fn = jax.jit(jax.jacrev(per_var_loss_fn))
        jac = jac_fn(theta_log0)
        jax.block_until_ready(jac)  # see note above -- force real completion before timing
        t1 = time.time()
        print(f"  (per-variable screen took {t1 - t0:.1f}s)")

        per_var_ranked = {}
        for i, var in enumerate(active_vars):
            elast_var = {}
            for name in candidates:
                g = float(jac[name][i])
                elast_var[name] = abs(g) if np.isfinite(g) else -1.0
            per_var_ranked[var] = sorted(elast_var.items(), key=lambda kv: kv[1], reverse=True)

        for var in active_vars:
            print(f"\nTop {args.topk_per_variable} sensitivity ranking for '{var}' "
                  f"(|d(loss_{var})/d(log param)|, unweighted, this variable only):")
            for name, e in per_var_ranked[var][: args.topk_per_variable]:
                flag = "  [NaN/Inf gradient -- excluded]" if e < 0 else ""
                print(f"  {name:28s} {e:12.4g}  (value={float(base_params[name]):.4g}){flag}")

        if args.sensitivity_only:
            return

        if args.select_mode == "combined":
            selected = [name for name, e in ranked if e >= 0][: args.topk]
        else:
            # Union of each active variable's own top --topk-per-variable,
            # order preserved by first appearance (temp's picks first, then
            # any new names from o2, then doc), duplicates dropped.
            selected = []
            for var in active_vars:
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

    early_stop_msg = (
        f"early-stopping after {args.early_stop_patience} stalled iteration(s) "
        f"(tol={args.early_stop_tol:.4g})" if args.early_stop_patience > 0 else "early-stopping disabled"
    )
    print(f"\nRunning up to {args.iters} Adam iterations ({n_steps} steps/iteration across {n_chunks} "
          f"gradient-truncated chunk(s), jitted -- first iteration includes compile time; {early_stop_msg})...")
    best_loss, best_theta_log = float(val0), dict(theta_log0)
    history = [float(val0)]
    stall_count = 0  # consecutive iterations with no "significant" improvement -- see --early-stop-patience
    for it in range(args.iters):
        t0 = time.time()
        loss_val, grads = grad_fn(theta_log)
        jax.block_until_ready((loss_val, grads))  # see note above on async dispatch
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
        # Relative to the best loss seen *before* this iteration -- matches
        # the sense of "has progress stalled", not just "did this exact
        # step improve on the previous one" (which would be noisy with Adam).
        improved_enough = (best_loss - loss_f) > args.early_stop_tol * max(abs(best_loss), 1e-12)
        if loss_f < best_loss:
            best_loss, best_theta_log = loss_f, dict(theta_log)
        print(f"  iter {it:3d}: loss={loss_f:.6g}  ({t1 - t0:.1f}s)")
        if args.early_stop_patience > 0:
            if improved_enough:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= args.early_stop_patience:
                    print(f"  iter {it}: loss hasn't improved by more than "
                          f"{args.early_stop_tol:.4g} (relative) for "
                          f"{args.early_stop_patience} consecutive iteration(s) -- stopping "
                          "early. Keeping the best result found so far.")
                    break

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
    # Built once and reused for both calls below: `params`'s keys/shapes/
    # dtypes are identical for the initial and calibrated parameter sets
    # (only the values differ), so the second call is a compiled-cache
    # hit rather than a second full compile of the whole simulation.
    sim_fn = jax.jit(
        lambda p: simulate_truncated(p, geometry, forcing, ice_state, init_state, args.chunk_steps)
    )
    metrics_initial = compute_metrics(sim_fn, base_params, obs)
    metrics_final = compute_metrics(sim_fn, params_final, obs)
    print_metrics_table(
        "Evaluation metrics (computed at the same (step, depth) pairs the loss scores; "
        "NSE/KGE/R2: 1.0 = perfect fit):",
        metrics_initial, metrics_final, active_vars,
    )

    out_path = os.path.join(os.getcwd(), "calibration_result.csv")
    pd.DataFrame(results).T.rename_axis("parameter").to_csv(out_path)
    print(f"\nSaved calibrated parameter table to {out_path}")

    metrics_rows = []
    for name in metrics_final:
        row = dict(variable=name, n=metrics_final[name]["n"], targeted=name in active_vars)
        for m in ("rmse", "nse", "kge", "r2"):
            row[f"{m}_initial"] = metrics_initial[name][m]
            row[f"{m}_calibrated"] = metrics_final[name][m]
        metrics_rows.append(row)
    metrics_path = os.path.join(os.getcwd(), "calibration_metrics.csv")
    pd.DataFrame(metrics_rows).set_index("variable").to_csv(metrics_path)
    print(f"Saved evaluation metrics (RMSE/NSE/KGE/R2, initial vs. calibrated) to {metrics_path}")


if __name__ == "__main__":
    main()
