from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import TypeVar

from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.schemas import StrategyName

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    """Base strategy to be inherited from SubstraFL strategies."""

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

    def initialization_round(
        self,
        train_data_nodes: List[TrainDataNode],
        clean_models: bool,
        round_idx: Optional[int] = 0,
        additional_orgs_permissions: Optional[set] = None,
    ):
        next_local_states = []

        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of Algo instead of a key for an algo
            next_local_state = node.init_states(
                self.algo.initialize(  # type: ignore
                    _algo_name=f"Initializing with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
                authorized_ids=set([node.organization_id]) | additional_orgs_permissions,
                clean_models=clean_models,
            )
            next_local_states.append(next_local_state)
        self._local_states = next_local_states

    @abstractmethod
    def perform_round(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Perform one round of the strategy

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralized strategy, unused otherwise
            round_idx (int): index of the round
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        raise NotImplementedError

    @abstractmethod
    def perform_predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Predict function of the strategy: evaluate the model.
        Gets the model for a train organization and evaluate it on the
        test nodes.

        Args:
            test_data_nodes (typing.List[TestDataNode]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNode]): list of nodes on which the model has
                been trained
            round_idx (int): index of the round
        """
        raise NotImplementedError

    def build_graph(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[List[AggregationNode]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: int,
        clean_models: Optional[bool],
    ):
        """_summary_

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralized strategy, unused otherwise
            evaluation_strategy (Optional[EvaluationStrategy]): _description_
            num_rounds (int): _description_
            clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
                if you want to download or re-use intermediary models. This causes the disk space to fill
                quickly so should be set to True unless needed. Defaults to True.
        """
        additional_orgs_permissions = (
            evaluation_strategy.test_data_nodes_org_ids if evaluation_strategy is not None else set()
        )

        # create computation graph.
        for round_idx in range(0, num_rounds + 1):
            if round_idx == 0:
                self.initialization_round(
                    train_data_nodes=train_data_nodes,
                    additional_orgs_permissions=additional_orgs_permissions,
                    clean_models=clean_models,
                )
            else:
                self.perform_round(
                    train_data_nodes=train_data_nodes,
                    aggregation_node=aggregation_node,
                    additional_orgs_permissions=additional_orgs_permissions,
                    round_idx=round_idx,
                    clean_models=clean_models,
                )

            if evaluation_strategy is not None and next(evaluation_strategy):
                self.perform_predict(
                    train_data_nodes=train_data_nodes,
                    test_data_nodes=evaluation_strategy.test_data_nodes,
                    round_idx=round_idx,
                )
