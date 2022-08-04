import substratools as tools
from sklearn.metrics import roc_auc_score


class AUC(tools.Metrics):
    def score(self, y_true, y_pred):
        """AUC"""
        metric = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0

        return float(metric)


if __name__ == "__main__":
    tools.metrics.execute(AUC())
