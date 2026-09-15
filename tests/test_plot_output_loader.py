import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plot_output


def test_load_result_from_npz(tmp_path):
    out_path = tmp_path / "res_lake1_jax_full.npz"
    data = {
        "temp": np.ones((2, 3)),
        "o2": np.ones((2, 3)) * 2,
        "docr": np.ones((2, 3)) * 3,
        "docl": np.ones((2, 3)) * 4,
        "pocr": np.ones((2, 3)) * 5,
        "pocl": np.ones((2, 3)) * 6,
        "time": np.array([0.0, 1.0]),
        "depth": np.array([0.0, 1.0, 2.0]),
    }
    np.savez(out_path, **data)

    res = plot_output.load_result(str(out_path))

    assert set(res.keys()) >= {"temp", "o2", "docr", "docl", "pocr", "pocl", "time", "depth"}
    assert res["temp"].shape == (2, 3)
    assert res["depth"].shape == (3,)


def test_load_result_from_directory_pickle(tmp_path):
    sim_dir = tmp_path / "run"
    sim_dir.mkdir()
    pickle_path = sim_dir / "res_lake1.pkl"
    data = {
        "temp": np.ones((2, 3)),
        "o2": np.ones((2, 3)) * 2,
        "docr": np.ones((2, 3)) * 3,
        "docl": np.ones((2, 3)) * 4,
        "pocr": np.ones((2, 3)) * 5,
        "pocl": np.ones((2, 3)) * 6,
        "time": np.array([0.0, 1.0]),
        "depth": np.array([0.0, 1.0, 2.0]),
    }
    import pickle

    with pickle_path.open("wb") as f:
        pickle.dump(data, f)

    res = plot_output.load_result(str(sim_dir))

    assert res["temp"].shape == (2, 3)


def test_prepare_plot_arrays_keeps_temperature_unscaled_by_volume():
    res = {
        "temp": np.array([[10.0, 12.0], [9.0, 11.0]]),
        "o2": np.array([[20.0, 24.0], [18.0, 22.0]]),
        "volume": np.array([2.0, 4.0]),
        "time": np.array([0.0, 1.0]),
        "depth": np.array([0.0, 1.0]),
    }

    arrays = plot_output._prepare_plot_arrays(res, res["time"], res["depth"], 2)

    assert np.allclose(arrays["temp"], res["temp"])
    assert np.allclose(arrays["o2"], np.array([[10.0, 6.0], [9.0, 5.5]]))


def test_temperature_profiles_use_shallowest_and_deepest_observed_depths():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "Depth_meter": [10.0, 0.0],
            "Water_Temperature_celsius": [4.0, 8.0],
        }
    )

    figure = plot_output._plot_temperature_profiles(
        temp=np.array([[8.0, 6.0, 4.0], [9.0, 7.0, 5.0]]),
        time_values=np.array([0.0, 86400.0]),
        depth_values=np.array([0.0, 5.0, 10.0]),
        observations=observations,
        start_time=pd.Timestamp("2020-01-01"),
    )

    assert [axis.get_title() for axis in figure.axes[:2]] == [
        "Temperature at 0 m observed depth",
        "Temperature at 10 m observed depth",
    ]
