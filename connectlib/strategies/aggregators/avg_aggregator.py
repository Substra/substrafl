import pickle

import numpy as np

from typing import List, Dict
from pathlib import Path

from .aggregator import Aggregator


class AvgAggregator(Aggregator):
    def aggregate(self, inmodels: List[Dict[str, np.array]], rank):

        if not inmodels:
            raise FileNotFoundError("This algo needs input models")

        # get keys
        weights_keys = inmodels[0].keys()

        # average weights
        averaged_weights = {}
        for key in weights_keys:
            weights = np.stack([model[key] for model in inmodels])
            averaged_weights[key] = np.mean(weights, axis=0)

        return averaged_weights

    def load_model(self, path):
        with Path(path).open("rb") as f:
            weights = pickle.load(f)
        return weights

    def save_model(self, model: Dict[str, np.array], path: str):
        with Path(path).open("wb") as f:
            pickle.dump(model, f)
