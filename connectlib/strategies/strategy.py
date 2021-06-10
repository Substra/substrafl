import json

from abc import ABC, abstractmethod
from typing import TypeVar, List
from pathlib import Path

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    def __init__(self, seed: int = 42, *args, **kwargs):
        self.seed = seed
        self.args = args
        self.kwargs = kwargs

    def delayed_init(self, seed: int, *args, **kwargs):
        pass

    @abstractmethod
    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
    ):
        # TODO: when substratools testtuples are patched to accept algo_key, we will not need train_data_nodes anymore
        raise NotImplementedError

    def load(self, path: Path):
        pass

    def save(self, path: Path):
        with path.open("w") as f:
            json.dump({"args": self.args, "kwargs": {**self.kwargs, "seed": self.seed}}, f)
