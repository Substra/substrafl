import json
import substratools
import cloudpickle
import pickle
import tempfile

import numpy as np

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Callable, Type


class Algo(substratools.CompositeAlgo):
    SEED: int

    def preprocessing(self, x: Any, y: Optional[Any] = None) -> Tuple[np.array, np.array]:
        return x, y

    @abstractmethod
    def perform_update(self, x: np.array, y: np.array):
        raise NotImplementedError

    @abstractmethod
    def test(self, x: np.array):
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self) -> Dict[str, np.array]:
        raise NotImplementedError

    @weights.setter
    @abstractmethod
    def weights(self, weights: Dict[str, np.array]):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path):
        raise NotImplementedError

    # Substra methods
    def train(
        self,
        X: Any,
        y: Any,
        head_model: Optional["Algo"],
        trunk_model: Optional[Dict[str, np.array]],
        rank: int,
    ):

        if head_model is None:
            head_model = self

        if trunk_model is None:
            print("trunk model is None")
            # raise TypeError("you need to run InitAggregator first")
        else:
            head_model.weights = trunk_model

        X, y = self.preprocessing(X, y)
        head_model.perform_update(X, y)

        return head_model, head_model.weights

    def predict(
        self, X: Any, head_model: Optional["Algo"], trunk_model: Optional[Dict[str, np.array]]
    ):
        assert head_model is not None
        assert trunk_model is not None

        head_model.weights = trunk_model

        X = self.preprocessing(X)
        return head_model.test(X)

    def load_trunk_model(self, path: str) -> Dict[str, np.array]:
        with Path(path).open("rb") as f:
            weights = pickle.load(f)
        return weights

    def save_trunk_model(self, model: Dict[str, np.array], path: str):
        with Path(path).open("wb") as f:
            pickle.dump(model, f)

    def load_head_model(self, path: str) -> "Algo":
        self.load(Path(path))
        return self

    def save_head_model(self, model: "Algo", path: str):
        model.save(Path(path))


@dataclass
class RegisteredAlgo:
    algo_cls: Algo
    algo_dir: Path
    parameters_path: Path
    cloudpickle_path: Path


def register(cls: Algo) -> Callable[..., RegisteredAlgo]:
    def pickle_algo(*args, **kwargs) -> RegisteredAlgo:
        parameters = {"args": args, "kwargs": kwargs}

        algo_dir = Path(tempfile.mkdtemp())
        parameters_path = algo_dir / "parameters.json"
        cloudpickle_path = algo_dir / "cloudpickle"

        with parameters_path.open("w") as f:
            # TypeError if types are not [str, int, float, bool, None]
            json.dump(parameters, f)

        with cloudpickle_path.open("wb") as f:
            cloudpickle.dump(cls, f)

        return RegisteredAlgo(cls, algo_dir, parameters_path, cloudpickle_path)

    return pickle_algo
