from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import TypeVar

from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.schemas import StrategyName

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    """Base strategy to be inherited from substrafl strategies."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        raise NotImplementedError

    @abstractmethod
    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
        clean_models: bool,
    ):
        """Perform one round of the strategy

        Args:
            algo (Algo): algo with the code to execute on the organization
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralized strategy, unused otherwise
            round_idx (int): index of the round
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
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
        Gets the model for a train organization and evaluate it on the
        test nodes.

        Args:
            algo (Algo): algo with the code to execute on the organization
            test_data_nodes (typing.List[TestDataNode]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNode]): list of nodes on which the model has
                been trained
            round_idx (int): index of the round
        """
        raise NotImplementedError
