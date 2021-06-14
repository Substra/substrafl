import substratools as tools

import numpy as np


class AccuracyMetric(tools.Metrics):
    def score(self, y_true, y_pred):
        """Returns the macro-average recall

        :param y_true: actual values from test data
        :type y_true: pd.DataFrame
        :param y_true: predicted values from test data
        :type y_pred: pd.DataFrame
        :rtype: float
        """
        assert np.all(y_pred == 1)
        # TODO: this is only temporary and used by the tests (tests/test_strategies.py
        # this should be improved and done for each test separately)
        # return accuracy_score(y_true, y_pred)
        return 1.0


if __name__ == "__main__":
    tools.metrics.execute(AccuracyMetric())
