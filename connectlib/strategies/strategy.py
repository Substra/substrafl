from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
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
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
    ):
        """Perform one round of the strategy

        Args:
            algo (Algo): algo with the code to execute on the node
            train_data_nodes (typing.List[TrainDataNode]): list of the train nodes
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralised strategy, unused otherwise
            round_idx (int): index of the round
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Predict function of the strategy: evaluate the model.
        Gets the model for a train node and evaluate it on the
        test nodes.

        Args:
            algo (Algo): algo with the code to execute on the node
            test_data_nodes (typing.List[TestDataNode]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNode]): list of nodes on which the model has been trained
            round_idx (int): index of the round
        """
        raise NotImplementedError
