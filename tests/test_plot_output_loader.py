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


def test_load_result_from_directory_prefers_full_output(tmp_path):
    np.savez(tmp_path / "res_lake1_jax_temp.npz", temp=np.ones((2, 3)))
    np.savez(
        tmp_path / "res_lake1_jax_full.npz",
        temp=np.ones((4, 3)),
        o2=np.ones((4, 3)),
    )

    res = plot_output.load_result(str(tmp_path))

    assert res["temp"].shape == (4, 3)
    assert "o2" in res


def test_load_water_quality_observations_by_variable(tmp_path):
    observations_path = tmp_path / "L0001-WQ.csv"
    pd.DataFrame(
        {
            "datetime": ["2020-01-01", "2020-01-01"],
            "depth": [0.0, 5.0],
            "observation": [10.0, 8.0],
            "variable": ["do", "doc"],
        }
    ).to_csv(observations_path, index=False)

    observations = plot_output._load_water_quality_observations(
        str(tmp_path), str(observations_path)
    )

    assert list(observations) == ["do", "doc"]
    assert observations["do"].iloc[0]["value"] == 10.0
    assert observations["doc"].iloc[0]["Depth_meter"] == 5.0


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


def test_prepare_plot_arrays_uses_external_volume_for_wq_fields():
    res = {
        "temp": np.array([[10.0, 12.0], [9.0, 11.0]]),
        "o2": np.array([[20.0, 24.0], [18.0, 22.0]]),
        "docr": np.array([[6.0, 8.0], [4.0, 10.0]]),
    }
    volume = np.array([2.0, 4.0])

    arrays = plot_output._prepare_plot_arrays(
        res, np.array([0.0, 1.0]), np.array([0.0, 1.0]), 2, volume
    )

    assert np.allclose(arrays["temp"], res["temp"])
    assert np.allclose(arrays["o2"], np.array([[10.0, 6.0], [9.0, 5.5]]))
    assert np.allclose(arrays["docr"], np.array([[3.0, 2.0], [2.0, 2.5]]))


def test_temperature_profiles_use_shallowest_and_deepest_observed_depths():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
            ),
            "Depth_meter": [10.0, 0.0, 10.0, 0.0],
            "Water_Temperature_celsius": [4.0, 8.0, 5.0, 9.0],
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
    assert figure._suptitle.get_text() == (
        "Observed depths during modeled period: upper 0 m, lower 10 m"
    )


def test_temperature_profiles_filter_observations_to_modeled_period():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2019-12-31", "2020-01-01", "2020-01-01",
                    "2020-01-02", "2020-01-02", "2020-01-03",
                ]
            ),
            "Depth_meter": [20.0, 5.0, 0.0, 5.0, 0.0, 30.0],
            "Water_Temperature_celsius": [1.0, 5.0, 8.0, 6.0, 9.0, 2.0],
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
        "Temperature at 5 m observed depth",
    ]
    assert all(len(axis.collections[0].get_offsets()) == 2 for axis in figure.axes[:2])


def test_water_quality_profiles_use_same_depth_rule():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-01", "2020-01-15", "2020-06-01", "2020-06-15",
                    "2021-01-01", "2021-01-15", "2021-06-01", "2021-06-15",
                    "2020-07-01",
                ]
            ),
            "Depth_meter": [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 5.0, 10.0],
            "value": [10.0, 9.0, 8.0, 7.0, 9.0, 8.0, 7.0, 6.0, 4.0],
        }
    )

    figure = plot_output._plot_observed_series(
        model_data=np.ones((2, 3)),
        time_values=np.array([
            0.0,
                (pd.Timestamp("2021-06-15") - pd.Timestamp("2020-01-01")).total_seconds(),
        ]),
        depth_values=np.array([0.0, 5.0, 10.0]),
        observations=observations,
        value_column="value",
        title="Dissolved oxygen",
        y_label="O2 (mg/L)",
        start_time=pd.Timestamp("2020-01-01"),
    )

    assert [axis.get_title() for axis in figure.axes[:2]] == [
        "Dissolved oxygen at 0 m observed depth",
        "Dissolved oxygen at 5 m observed depth",
    ]
    scatter_axis = figure.axes[2]
    assert scatter_axis.get_title() == "Dissolved oxygen: modeled vs observed"
    assert len(scatter_axis.collections) == 1
    assert len(scatter_axis.collections[0].get_offsets()) == len(observations)
    assert len(scatter_axis.lines) == 1
    assert scatter_axis.lines[0].get_label() == "1:1"
    assert figure.axes[3].get_ylabel() == "Observed depth (m)"


def test_lower_depth_requires_two_observations_in_each_upper_data_year():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01",
                    "2020-01-15", "2020-06-15", "2021-01-15",
                ]
            ),
            "Depth_meter": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
            "value": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        }
    )

    assert plot_output._select_temperature_depths(observations) is None

    observations = pd.concat(
        [
            observations,
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2021-06-15"]),
                    "Depth_meter": [5.0],
                    "value": [3.0],
                }
            ),
        ],
        ignore_index=True,
    )

    assert plot_output._select_temperature_depths(observations) == (0.0, 5.0)


def test_temperature_profiles_choose_lower_depth_with_upper_year_overlap():
    observations = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-01", "2020-01-15", "2020-06-01", "2020-06-15",
                    "2021-01-01", "2021-01-15", "2021-06-01", "2021-06-15",
                ]
            ),
            "Depth_meter": [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 5.0],
            "Water_Temperature_celsius": [8.0, 7.5, 6.0, 5.5, 7.0, 6.5, 5.0, 4.5],
        }
    )
    observations = pd.concat(
        [
            observations,
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2020-07-01"]),
                    "Depth_meter": [10.0],
                    "Water_Temperature_celsius": [4.0],
                }
            ),
        ],
        ignore_index=True,
    )

    figure = plot_output._plot_temperature_profiles(
        temp=np.array([[8.0, 6.0, 4.0], [9.0, 7.0, 5.0]]),
        time_values=np.array([
            0.0,
            (pd.Timestamp("2021-06-15") - pd.Timestamp("2020-01-01")).total_seconds(),
        ]),
        depth_values=np.array([0.0, 5.0, 10.0]),
        observations=observations,
        start_time=pd.Timestamp("2020-01-01"),
    )

    assert [axis.get_title() for axis in figure.axes[:2]] == [
        "Temperature at 0 m observed depth",
        "Temperature at 5 m observed depth",
    ]


def test_build_time_labels_interprets_model_times_as_seconds():
    ticks, labels = plot_output._build_time_labels(
        np.array([0.0, 86400.0]), pd.Timestamp("2020-01-01")
    )

    assert np.array_equal(ticks, np.array([0, 1]))
    assert labels == ["2020-01-01", "2020-01-02"]
