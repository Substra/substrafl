from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np

from connectlib.remote import remote_data

Weights = Dict[str, np.ndarray]


class Algo:
    def __init__(self, seed: int = 42, *args, **kwargs):
        self.seed = seed
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def delayed_init(self, seed: int, *args, **kwargs):
        raise NotImplementedError

    @remote_data
    @abstractmethod
    def train(self, x: Any, y: Any, num_updates: int, shared_state: Weights) -> Weights:
        raise NotImplementedError

    @remote_data
    @abstractmethod
    def predict(self, x: Any, shared_state: Weights) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path):
        # load local state
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path):
        # save local state
        raise NotImplementedError
