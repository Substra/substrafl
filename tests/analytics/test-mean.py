import numpy as np
import pandas as pd
import pytest

from substrafl.analytics.mean import Mean


@pytest.fixture()
def data():
    data = {
        "1": [2, 3, 4, 100, 11, 10],
        "2": [20000, 25000, np.nan, 30000, 23456, 65000],
        "4": [1000, 2300, 1200, 2000, 1100, np.nan],
    }

    data_df = pd.DataFrame(data)
    return data_df


@pytest.fixture()
def data_split(data):
    split_data = [
        pd.DataFrame({k: v[0:2] for k, v in data.items()}),
        pd.DataFrame({k: v[2:5] for k, v in data.items()}),
        pd.DataFrame({k: [v[5]] for k, v in data.items()}),
    ]
    return split_data


def test_local_mean(data):
    """Test local_mean method."""
    # pylint: disable=unexpected-keyword-arg
    state = Mean().local_mean(datasamples=data, _skip=True)

    local_means = data.mean(skipna=True)
    size = data.count()

    assert isinstance(state, dict)
    assert "mean" in state
    assert "n_samples" in state
    assert (state["mean"] == local_means).all()
    assert (state["n_samples"] == size).all()


def test_aggregate_mean(data, data_split):
    """Test aggregate_mean method."""
    local_means = data.mean(skipna=True)

    shared_state = [
        {
            "mean": center_data.mean(numeric_only=True, skipna=True),
            "n_samples": center_data.count(),
        }
        for center_data in data_split
    ]

    avg_state = Mean().aggregate_mean(shared_state, _skip=True)

    assert isinstance(avg_state, dict)
    assert "global_mean" in avg_state
    assert np.isclose(avg_state["global_mean"], local_means).all()
