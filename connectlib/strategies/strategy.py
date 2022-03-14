import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import TypeVar

from connectlib.algorithms.algo import Algo
from connectlib.nodes.aggregation_node import AggregationNode
from connectlib.nodes.test_data_node import TestDataNode
from connectlib.nodes.train_data_node import TrainDataNode

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    """Base strategy to be inherited from connectlib strategies."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        raise NotImplementedError

    def load(self, path: Path):
        pass

    def save(self, path: Path):
        with path.open("w") as f:
            json.dump({"args": self.args, "kwargs": {**self.kwargs, "seed": self.seed}}, f)
