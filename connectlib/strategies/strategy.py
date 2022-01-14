import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import TypeVar

from connectlib.algorithms import Algo
from connectlib.nodes import AggregationNode
from connectlib.nodes import TestDataNode
from connectlib.nodes import TrainDataNode

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

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
        raise NotImplementedError

    def load(self, path: Path):
        pass

    def save(self, path: Path):
        with path.open("w") as f:
            json.dump({"args": self.args, "kwargs": {**self.kwargs, "seed": self.seed}}, f)
