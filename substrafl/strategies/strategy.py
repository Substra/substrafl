from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import TypeVar

from substrafl import exceptions
from substrafl.algorithms.algo import Algo
from substrafl.compute_plan_builder import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.strategies.schemas import StrategyName

SharedState = TypeVar("SharedState")


class Strategy(ComputePlanBuilder):
    """Base strategy to be inherited from SubstraFL strategies."""

    def __init__(self, algo: Algo, *args, **kwargs):
        """
        All child class arguments need to be passed to it through its ``args`` and ``kwargs``
        in order to use them when instantiating it as a RemoteStruct in each process.

        Example:

            .. code-block:: python

                class MyStrat(Strategy):
                    def __init__(self, algo, my_custom_arg):
                        super().__init__(algo=algo, my_custom_arg=my_custom_arg)

        Args:
            algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)

        Raises:
            exceptions.IncompatibleAlgoStrategyError: Raise an error if the strategy name is not in ``algo.strategies``.
        """

        super().__init__(algo=algo, *args, **kwargs)

        self.algo = algo

        if self.name not in algo.strategies:
            raise exceptions.IncompatibleAlgoStrategyError(
                f"The algo {self.algo.__class__.__name__} is not compatible with the strategy "
                f"{self.__class__.__name__}, "
                f"named {self.name}. Check the algo strategies property: algo.strategies to see the list of compatible "
                "strategies."
            )

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
        *,
        train_data_nodes: List[TrainDataNode],
        clean_models: bool,
        round_idx: Optional[int] = 0,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Call the initialize function of the algo on each train node.

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            round_idx (typing.Optional[int]): index of the round. Defaults to 0.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization. Default to None
        """
        next_local_states = []

        for node in train_data_nodes:
            # define train tasks (do not submit yet)
            # for each train task give description of Algo instead of a key for an algo
            next_local_state = node.init_states(
                operation=self.algo.initialize(
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
        *,
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
        """Perform the prediction of the algo on each test nodes.
        Gets the model for a train organization and compute the prediction on the
        test nodes.

        Args:
            test_data_nodes (typing.List[TestDataNode]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNode]): list of nodes on which the model has
                been trained
            round_idx (int): index of the round
        """
        raise NotImplementedError

    def build_compute_plan(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[List[AggregationNode]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: int,
        clean_models: Optional[bool] = True,
    ) -> None:
        """Build the compute plan of the strategy.
        The built graph will be stored by side effect in the given train_data_nodes,
        aggregation_nodes and evaluation_strategy.
        This function create a graph be first calling the initialization_round method of the strategy
        at round 0, and then call the perform_round method for each new round.
        If the current round is part of the evaluation strategy, the perform_predict method is
        called to complete the graph.

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralized strategy, unused otherwise
            evaluation_strategy (Optional[EvaluationStrategy]): evaluation strategy to follow for testing models.
            num_rounds (int): Number of times to repeat the compute plan sub-graph (define in perform round).
            clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
                if you want to download or re-use intermediary models. This causes the disk space to fill
                quickly so should be set to True unless needed. Defaults to True.

        Returns:
            None
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
                if round_idx == num_rounds:
                    clean_models = False  # Enforce to keep at least the outputs of the last round.

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

    def save_local_state(self, path: Path) -> None:
        self.algo.save_local_state(path)

    def load_local_state(self, path: Path) -> Any:
        return self.algo.load_local_state(path)
