import pickle

import numpy as np

from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

import substratools as tools


class Aggregator(tools.AggregateAlgo):
    @abstractmethod
    def aggregate_states(self, states: List[Dict[str, np.array]]):
        raise NotImplementedError

    # Substra methods
    def aggregate(self, inmodels: List[Dict[str, np.array]], rank):
        return self.aggregate_states(inmodels)

    def load_model(self, path: str):
        with Path(path).open("rb") as f:
            weights = pickle.load(f)
        return weights

    def save_model(self, model: Dict[str, np.array], path: str):
        with Path(path).open("wb") as f:
            pickle.dump(model, f)
