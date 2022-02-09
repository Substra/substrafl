from pathlib import Path

import numpy as np
import substratools as tools
from sklearn.metrics import roc_auc_score


class AUC(tools.Metrics):
    def score(self, y_true, y_pred):
        """AUC"""
        indexes = np.loadtxt(Path(y_true[0]) / "index.csv", delimiter=",", dtype=str)
        indexes = indexes[np.argsort(indexes[:, 0])]

        y_true = np.array([int(x == "Tumor") for x in indexes[:, 1]])
        metric = roc_auc_score(y_true, y_pred)

        return float(metric)


if __name__ == "__main__":
    tools.metrics.execute(AUC())
