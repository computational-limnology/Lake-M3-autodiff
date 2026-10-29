# mcl/

Everything for the experimental MCL (machine-learning kz correction) mode:

Scripts (they add ../src to the import path themselves; run from anywhere):
- `train_M3_mcl_jax.py`: train the LSTM kz correction
- `run_M3_mcl_jax.py`: run an already trained correction
- `run_M3_calibrate_then_mcl.py`: calibration (src/calibrate_M3_jax.py) then training
- `plot_mcl_kz.py`, `plot_ri_diagnostics.py`: plots of training results


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
