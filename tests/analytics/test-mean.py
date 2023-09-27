"""Units test for the mean strategy."""
import unittest

import numpy as np
import pandas as pd
from analytics.mean import Mean


class TestMean(unittest.TestCase):
    """Test the substrafl Mean."""

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
        self.local_means = self.df.mean(skipna=True)
        self.size = self.df.count()
        split_data = [
            pd.DataFrame({k: v[0:2] for k, v in data.items()}),
            pd.DataFrame({k: v[2:5] for k, v in data.items()}),
            pd.DataFrame({k: [v[5]] for k, v in data.items()}),
        ]
        self.shared_state = [
            {
                "mean": center_data.mean(numeric_only=True, skipna=True),
                "n_samples": center_data.count(),
            }
            for center_data in split_data
        ]

    def test_local_mean(self):
        """Test local_mean method."""
        # pylint: disable=unexpected-keyword-arg
        state = Mean().local_mean(datasamples=self.df, _skip=True)
        assert isinstance(state, dict)
        assert "mean" in state
        assert "n_samples" in state
        assert (state["mean"] == self.local_means).all()
        assert (state["n_samples"] == self.size).all()

    def test_aggregate_mean(self):
        """Test aggregate_mean method."""
        # pylint: disable=unexpected-keyword-arg
        avg_state = Mean().aggregate_mean(self.shared_state, _skip=True)

        assert isinstance(avg_state, dict)
        assert "global_mean" in avg_state
        assert np.isclose(avg_state["global_mean"], self.local_means).all()
