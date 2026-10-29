# CLAUDE.md — Lake-M3-autodiff handoff

Handoff notes for continuing this project with Claude on another machine.
Written 2026-10-28 at the end of a long working session. Owner: Robert Ladwig
(rladwig@ecos.au.dk). Read this first, then the module docstrings — they are
detailed and explain most design choices in place.

## Goal

Lake-M3 is a vertical 1D lake model (temperature + O2, DOC-R/L, POC-R/L).
The original numpy/pandas version is `src/processBased_lakeModel_functions.py`
(`run_wq_model()`, driven by `src/run_M3.py`) and is kept as the reference.
The project re-implements the whole model in JAX for automatic
differentiation: gradient-based calibration, a machine-learning correction of
the eddy diffusivity (MCL), and gradient-based attribution.

Test lakes: `Ravn/` (Danish lake, 1.7 km², 32 m; main case) and `Mendota/`
(Wisconsin, 39.9 km²). Each lake folder has `run_config.csv`,
`model_params.csv`, `lake_config.csv`, `ice_and_snow.csv`, forcing and
observation files. The user runs the heavy jobs on a 14-core Linux machine
(`/home/au740615/Documents/projects/Lake-M3-autodiff`, conda env `lake-m3`).

## Layout

```
src/
  jax_lakeModel_functions.py   the JAX model (state = LakeState dict pytree)
  run_M3_jax.py                continuous forward run -> <lake>/res_lake1_jax_full.npz
  calibrate_M3_jax.py          gradient calibration (Adam, truncated BPTT; --windows years)
  apply_calibration.py         writes calibration_result.csv into model_params.csv (keeps backups)
  windows.py                   windowed (yearly, parallel) simulation infrastructure
  benchmark_windows.py         Stage-0 speed benchmark for windowed runs
  diagnose_temperature_errors.py  profile-based error diagnosis (A-F analyses, decision rules)
  plot_output.py               standard plots + metrics (+ year x month layer bias map)
  data_tools.py                observation loading + data prep CLI (fix-datetime, fix-surface,
                               integrate-buoy, integrate-ntl, flag-hf) + mcl path helpers
  processBased_lakeModel_functions.py, run_M3.py, ancillary_functions.py   numpy reference
mcl/                           MCL scripts + trained artefacts (<name>_<lake>.pkl/.npz)
attribution/                   SCS-CN catchment model + POC-labile gradient attribution
tests/                         test_full_step_mcl.py, test_windows.py, test_plot_output_loader.py
Richardson-seiche mixing scheme mathematical description.{docx,pdf}
```

Run scripts from the project root, e.g. `python src/run_M3_jax.py Ravn`.

## Model essentials (JAX)

- `full_step(state, forcing, geometry, params)` = one hourly step: kd_light,
  OC loading, heating, ice, O2 boundary, wind mixing (`mixing_step`), eddy
  diffusivity, diffusion, POC settling, convection, NPP + production/
  consumption, outflow, momentum (uvel/vvel, Goudsmit-style).
- `full_step_mcl(..., kz_override)` is a verbatim copy with a hook for the
  MCL correction; only MCL scripts use it. Physics changes must go into both;
  `tests/test_full_step_mcl.py` checks they agree.
- `run_config.csv: diffusion_method` ∈ hendersonSellers | richardson |
  richardson_seiche (Ravn uses richardson_seiche: resolved-shear Ri closure in
  the surface layer, seiche energy budget + Osborn interior mixing below; see
  the docx/pdf and the claude.ai doc "Richardson-seiche mixing scheme").
- `default_params()` builds params from model_params.csv. Never pass the
  `diffusion_method` string through `jax.jit` arguments — close over it.
- LakeState key order must stay identical across scan steps (pytree).
- Gradients: truncated BPTT, stop_gradient every `--chunk-steps` (2000 h);
  stable gradient horizon ~7500 h. Custom JVPs bound some derivatives
  (`_safe_div`, `_ri_safe_div`).

## Data conventions

- Temperature observations: `run_config: u_ini_file` (Ravn `L0001-HD.csv`).
  It merges sparse manual profiles and the buoy record; column
  `high_frequency` (1 = buoy, 0 = profile; set by `data_tools.py flag-hf`).
  `--temp-obs profiles` (calibrate/train) uses only manual profiles. The user
  considers buoy depths probably wrong -> work with profiles only.
- Ravn manual profiles: ~19/yr 1989-2014, then only even years (2016-2024).
- WQ obs: `L0001-WQ.csv` (do, doc, kd from Secchi). Ravn precipitation column
  is hourly amounts.
- Result .npz files store `start_time` (newer runs); plot scripts prefer it
  over run_config (older files: pass `--start-time`).
- MCL artefacts live in `mcl/` as `<name>_<lake>.pkl/.npz`; a bare name passed
  to `--out/--nn-params(-out)` resolves there (`data_tools.mcl_path`).
- Data edits always kept backups (`*.pre_*`, `*.bak`).

## Windowed calibration (new, working)

`calibrate_M3_jax.py --windows years` computes the loss on independent yearly
windows (1 Feb + 8016 h, 28-day unscored spin-up), each started from the
state of a continuous reference run on that date, run in parallel
(`--devices 14 --single-thread`: pmap over CPU devices + vmap). Start states
are refreshed every `--refresh-every 5` iterations; `--test-years every:4`
holds years out; the end reports initial vs best on the same start states,
then the usual continuous-run metrics. Needs `--params`.
- Speed (14 cores): ~7 min/iteration vs ~40 min for the continuous
  1989-2024 gradient; reference run ~8 min.
- `--check-window-gradient` compares windowed vs continuous gradient (cosine);
  result now also saved to `<lake>/calibration_gradient_check.csv`. The cosine
  has NOT yet been seen for a full-size run (output lost in log spam).
- With `--devices > 1` XLA prints harmless "may be stuck / unstuck
  false-positive" messages; now hidden by default (`TF_CPP_MIN_LOG_LEVEL=3`),
  `--xla-warnings` shows them. Unverified on the Linux machine; fallback
  `2> xla.log`.
- Tests: `python src/windows.py Ravn --test --devices 14 --single-thread`,
  `python tests/test_windows.py`.

## Key findings so far

1. Round-off amplification: a 1e-15 perturbation grows to ~0.1 °C RMS and
   0.6-1.6 °C peaks within one year (threshold processes: layer-wise
   mixed-layer deepening in `mixing_step`, `Zg = ceil(..*100)/100` rounding,
   convection threshold, ice switches). Windowed/batched runs therefore never
   match continuous runs exactly; judge by gradient direction and final
   continuous metrics. "Smoothing threshold processes" is a planned option.
2. Fixed NaN gradient: `bulk_fluxes` with Tair == Twater and RH = 100 % over
   ice (L = inf). Now carries the inverse Obukhov length.
3. Step-1 diagnosis (`diagnose_temperature_errors.py`) on the profiles-only
   1989-2024 run before windowed calibration: 82 % of error was vertical
   distribution; summer mixing too deep (thermocline +6 m, hypolimnion +2.3 °C,
   turnover 18 d early). MCL had no measurable effect (correction stayed ~0).
4. Windowed calibration run 1 (start: earlier calibrated values incl.
   at_factor 0.86; params sw_factor, wind_factor, k_max, seiche_alpha):
   windowed loss 0.339 -> 0.096, test years 0.390 -> 0.093; continuous temp
   RMSE 2.52 -> 1.34 °C, NSE 0.67 -> 0.91. Diagnosis after: thermocline +1.7 m,
   surface-bottom dT bias ~0, Schmidt ratio 0.94, turnover 1.6 d early; lake
   0.5 °C too cold (winter -0.9) — attributed to at_factor 0.86.
   Result: wind_factor 0.69, seiche_alpha 6.2e-6, k_max 4.5e-4, sw_factor 1.0.
5. Run 2 (start: defaults; params at_factor, sw_factor, wind_factor,
   seiche_alpha, k_max; 15 iters, not converged): loss 0.122, continuous RMSE
   1.48 / NSE 0.887 — worse than run 1. Now under-mixed: k_max -82 % (5.6e-5),
   sw_factor 1.09 -> summer surface/meta +1.8 °C, too stable (Schmidt 1.28),
   onset 17 d early, inverse winter stratification 37 % vs 16 % obs;
   turnover timing better. at_factor stayed ~1.0.
   => Forcing factors and mixing trade off (non-identifiable from profiles);
   result depends on start point.
6. Forcing record: step changes RH 1998, precip 1998, wind 2018; metalimnion
   residual step ~2000 -> check weather-station history.
7. Run 4 (wind_factor 1 with lake_config WindSpeed 1.2, mixing-only params
   W_str, seiche_alpha, k_max, sbl_drho): badly over-mixed (NSE 0.42) and
   gradients exploded (1e9-1e15, cosine -0.20). Effective wind ~0.76x measured
   (sheltering) fits Ravn; run 3 stays the best set.
8. Gradient explosion located (2026-10-29): spring near 4 degC, loop
   temperature -> N2 -> Osborn K = Gamma*eps/N2 -> diffusion (dK/dN2 = -K/N2),
   amplified by mixing_step's KP = KE/POE (POE ~ tiny density step). Fixed with
   bounded custom derivatives (A1 `_osborn_div`, A2 `_capped`, B1 `_kp_ratio`;
   forward bit-identical) and removal of the Zg cm-rounding (B2; changes
   forward by ~0.06 degC RMS, within the round-off noise floor).
   Ravn lake_config WindSpeed is now 1 (was 1.2); wind_factor 0.759 = run 3.
9. kd model has no skill (NSE ~ -0.07); Ravn DOC NSE NaN in plot_output
   (constant obs?) — open.

## Agreed next steps

1. Restore run-1 mixing parameters (apply_calibration backups), set
   at_factor = 1 and sw_factor = 1, then calibrate mixing only:
   ```
   python src/calibrate_M3_jax.py Ravn --variables temp --temp-obs profiles \
     --params wind_factor,seiche_alpha,k_max --windows years \
     --devices 14 --single-thread --test-years every:4 --check-window-gradient --iters 25
   ```
   Check whether Cd, alpha, sbl_drho were reset to defaults before run 2.
2. Apply, run `python src/run_M3_jax.py Ravn`, then
   `python src/diagnose_temperature_errors.py Ravn --run PB=Ravn/res_lake1_jax_full.npz`.
3. Step 3 (planned): better optimiser (backtracking / L-BFGS), parameter
   uncertainty from the residual Jacobian, maybe onset date / thermocline
   depth as loss terms.
4. Phase 3/4 of windowing: start windows from observed winter profiles,
   spin-up length test; windowed MCL training (`train_M3_mcl_jax.py`).
5. Optional: smooth threshold processes (remove Zg rounding, soft convection,
   continuous mixed-layer depth; behind a `smooth_thresholds` switch).

## Working conventions

- The user prefers concise, direct explanations, tables for comparisons,
  and testing before reporting. Keep the numpy reference untouched.
- Back up data files before editing; never delete user files without asking.
- When adding physics, update `full_step` and `full_step_mcl` together and
  run `tests/test_full_step_mcl.py`.
