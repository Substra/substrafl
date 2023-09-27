"""Units test for the variance strategy."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import moment

from substrafl.analytics.variance import StrategyVariance


class TestVarianceStrategy(unittest.TestCase):
    """Test the substrafl Variance strategy."""

    def setUp(self):
        """Build all necessary quantities for all tests.

        Builds ground truths.
        """
        data = {
            "1": [2, 3, 4, 100, 11, 10],
            "2": [20000, 25000, np.nan, 30000, 23456, 65000],
            "4": [1000, 2300, 1200, 2000, 1100, np.nan],
        }
        self.df = pd.DataFrame(data)
        self.global_mean = self.df.mean()
        self.global_variance = self.df.var(numeric_only=True, ddof=0)
        self.global_moment0 = moment(self.df.select_dtypes(include=np.number), 0, nan_policy="omit")
        self.global_moment1 = moment(self.df.select_dtypes(include=np.number), 1, nan_policy="omit")
        self.global_moment2 = moment(self.df.select_dtypes(include=np.number), 2, nan_policy="omit")
        self.size = self.df.count()
        self.split_data = [
            pd.DataFrame({k: v[0:2] for k, v in data.items()}),
            pd.DataFrame({k: v[2:5] for k, v in data.items()}),
            pd.DataFrame({k: [v[5]] for k, v in data.items()}),
        ]


def test_local_m1_and_m2(self):
    """Test local_m1_and_m2 method."""
    # pylint: disable=unexpected-keyword-arg
    state = StrategyVariance().local_moments_1_and_2(datasamples=self.df, _skip=True)
    assert isinstance(state, dict)
    assert "moment2" in state
    assert "moment1" in state
    assert "moment0" in state
    assert "n_samples" in state
    assert "mean" in state

    assert (state["n_samples"] == self.size).all()
    assert (state["mean"] == self.global_mean).all()
    assert (state["moment2"] == self.global_moment2).all()
    assert (state["moment1"] == self.global_moment1).all()
    assert (state["moment0"] == self.global_moment0).all()


def test_aggregate_variance(self):
    """Test aggregate_variance method."""
    shared_states = [
        {
            "moment0": moment(center_data.select_dtypes(include=np.number), 0, nan_policy="omit"),
            "moment1": moment(center_data.select_dtypes(include=np.number), 1, nan_policy="omit"),
            "moment2": moment(center_data.select_dtypes(include=np.number), 2, nan_policy="omit"),
            "mean": center_data.mean(numeric_only=True, skipna=True),
            "n_samples": center_data.select_dtypes(include=np.number).count(),
        }
        for center_data in self.split_data
    ]
    # pylint: disable=unexpected-keyword-arg
    avg_state = StrategyVariance().aggregate_variance(shared_states, _skip=True)
    assert isinstance(avg_state, dict)
    assert "global_mean" in avg_state
    assert "global_variance" in avg_state
    assert "global_n_samples" in avg_state
    assert np.isclose(avg_state["global_mean"], self.global_mean).all()
    assert np.isclose(avg_state["global_variance"], self.global_variance).all()
    assert np.isclose(avg_state["global_n_samples"], self.size).all()
