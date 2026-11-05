# mcl/

Everything for the experimental MCL (machine-learning kz correction) mode:

Scripts (they add ../src to the import path themselves; run from anywhere):
- `train_M3_mcl_jax.py`: train the LSTM kz correction
- `run_M3_mcl_jax.py`: run an already trained correction
- `run_M3_calibrate_then_mcl.py`: calibration (src/calibrate_M3_jax.py) then training
- `plot_mcl_kz.py`, `plot_ri_diagnostics.py`: plots of training results
- `mcl_regime.py`, `train_mcl_regime.py`: sparse-data alternative to the LSTM --
  ~20 coefficients (4 depth bands relative to the mixed layer x constant,
  season sin/cos, Schmidt stability, ice), trained on yearly windows of all
  profile years in parallel (train / validation / test years), with a
  noise-aware stopping rule. Writes `mcl_regime_<lake>.npz` (+ `_coef.csv`,
  `_history.csv`, `_windowed_rmse.csv`, `_continuous_rmse.csv`) here and the
  continuous corrected run to `<lake>/res_lake1_mcl_regime.npz`.
  `--apply mcl/mcl_regime_<lake>.npz` reruns a trained correction only; so does
  `python mcl/run_M3_mcl_jax.py <lake> --nn-params mcl_regime.npz` (any .npz is
  treated as a regime file). Tests: `python tests/test_mcl_regime.py`.
  Default since 2026-11-04 (`--band-mode thermocline`): 5 bands (mixed layer,
  above core, thermocline core, below core, hypolimnion; core = N^2-weighted
  centre) and an additive mixing term in the three thermocline bands
  (k_ref 5e-6, max 4e-5 m2/s) whose mixing power is capped at `--f-max`
  (0.2 %) of the 24-h wind power and is zero under ice; loss adds
  `--grad-weight` x MSE of 1-m temperature differences (thermocline sharpness);
  reports summer thermocline metrics (max dT/dz, depth bias, bias relative to
  the core). `--band-mode mixed_layer --no-additive` = the first version
  (old `mcl_regime_<lake>.npz` files still load).
  Since 2026-11-05 two heads (`--heads kz,tke`, default): the kz head (above /
  without the mixed-layer band, interior mixing -> sharpness) and a TKE head:
  entrainment efficiency eta in [0.3, 1.3] on the wind energy in mixing_step
  (-> thermocline depth), gated to stratified conditions (Schmidt > ~30 J m-2).
  Extra features: 7-day wind power, Lake Number, net heat input (TKE head).
  `--energy-link` (default with both heads): additive thermocline mixing can
  only use gamma x the wind energy withheld from deepening in the same step.
  Loss adds `--depth-weight` x MSE(thermocline core depth). Separate learning
  rates (`--lr-kz 0.02`, `--lr-tke 0.01`). `--grad-ri-max 1e4` loosens the
  surface-layer Ri derivative bound while training (with the model's bound 1
  the TKE gradient has the wrong sign in summer). Ablations: `--heads tke`,
  `--heads kz`. Tests: `python tests/test_mcl_two_heads.py`.


Trained artefacts for all lakes, with the
lake folder's name as suffix:

- `<name>_<lake>.pkl`: trained network weights + settings (written by
  `train_M3_mcl_jax.py --nn-params-out`, read by `run_M3_mcl_jax.py --nn-params`)
- `<name>_<lake>.npz`: training result, baseline vs hybrid fields (written by
  `train_M3_mcl_jax.py --out`, read by `plot_mcl_kz.py` / `plot_ri_diagnostics.py`)

A bare file name given to those options resolves here automatically
(`--nn-params mcl_add.pkl` on Ravn -> `mcl/mcl_add_Ravn.pkl`). A path with a
folder is used as given, e.g. to run Mendota-trained weights on Ravn:

    python mcl/run_M3_mcl_jax.py Ravn --nn-params mcl/mcl_nn_params_Mendota.pkl

Plot scripts find the matching lake folder from the `_<lake>` suffix
(override with `--data-dir`).
