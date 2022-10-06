import numpy as np
import substratools as tools
from sklearn.metrics import roc_auc_score


def score(inputs, outputs, task_properties):
    """AUC"""

    y_pred = get_predictions(inputs["predictions"])
    y_true = inputs["datasamples"].y_true

    metric = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0

    tools.save_performance(float(metric), outputs["performance"])


def get_predictions(path):
    return np.load(path)


if __name__ == "__main__":
    tools.function.execute_cli([score])
