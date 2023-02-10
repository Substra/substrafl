from abc import abstractmethod
from typing import List
from typing import Optional

from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.strategies.strategy import Strategy


class CentralizedStrategy(Strategy):
    """Centralized strategy to be inherited from SubstraFL centralized strategies.
    Centralized strategies share the computational graph, computed from the ``initialization_round``,
    ``perform_round`` and ``predict_function``. They differentiate from one another by their aggregation
    function, computed in ``compute_aggregated_states``."""

    def __init__(self, *args, **kwargs):
        super(CentralizedStrategy, self).__init__(*args, **kwargs)

    @abstractmethod
    def compute_aggregated_states(self, *args, **kwargs):
        """A centralized strategy compute aggregated states to share to each node.

        Returns:
            AggregatedStates: Return the computed aggregated states of the aggregation.
        """
        raise NotImplementedError

    def initialization_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Initialize the strategy by performing a local update
        (train on n mini-batches) of the models on each train data node

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict methods
            train_data_nodes (typing.List[TrainDataNode]): List of the nodes on which to perform
                local updates.
            aggregation_node (AggregationNode): Node without data, used to perform
                operations on the shared states of the models
            round_idx (int): Round number, it starts at 0.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        # Initialization of the strategy by performing a local update on each train data organization
        assert self._local_states is None
        assert self._shared_states is None
        self._perform_local_updates(
            algo=algo,
            train_data_nodes=train_data_nodes,
            current_aggregation=None,
            round_idx=round_idx,
            aggregation_id=aggregation_node.organization_id,
            additional_orgs_permissions=additional_orgs_permissions or set(),
            clean_models=clean_models,
        )

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """One round of the centralized strategy consists in:
            - aggregate the model shared_states
            - set the model weights to the aggregated weights on each train data nodes
            - perform a local update (train on n mini-batches) of the models on each train data nodes

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict methods
            train_data_nodes (typing.List[TrainDataNode]): List of the nodes on which to perform
                local updates.
            aggregation_node (AggregationNode): Node without data, used to perform
                operations on the shared states of the models
            round_idx (int): Round number, it starts at 0.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        if aggregation_node is None:
            raise ValueError(f"In {self.name} strategy aggregation node cannot be None")

        current_aggregation = aggregation_node.update_states(
            self.compute_aggregated_states(shared_states=self._shared_states, _algo_name="Aggregating"),  # type: ignore
            round_idx=round_idx,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
            clean_models=clean_models,
        )

        self._perform_local_updates(
            algo=algo,
            train_data_nodes=train_data_nodes,
            current_aggregation=current_aggregation,
            round_idx=round_idx,
            aggregation_id=aggregation_node.organization_id,
            additional_orgs_permissions=additional_orgs_permissions or set(),
            clean_models=clean_models,
        )

    def _perform_local_updates(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        current_aggregation: Optional[SharedStateRef],
        round_idx: int,
        aggregation_id: str,
        additional_orgs_permissions: set,
        clean_models: bool,
    ):
        """Perform a local update (train on n mini-batches) of the models
        on each train data nodes.

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict methods
            train_data_nodes (typing.List[TrainDataNode]): List of the organizations on which to perform
            local updates current_aggregation (SharedStateRef, Optional): Reference of an aggregation operation to be
            passed as input to each local training
            round_idx (int): Round number, it starts at 1.
            aggregation_id (str): Id of the aggregation node the shared state is given to.
            additional_orgs_permissions (set): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
        """

        next_local_states = []
        next_shared_states = []

        for i, node in enumerate(train_data_nodes):
            # define composite tasks (do not submit yet)
            # for each composite task give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.update_states(
                algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=current_aggregation,
                    _algo_name=f"Training with {algo.__class__.__name__}",
                ),
                local_state=self._local_states[i] if self._local_states is not None else None,
                round_idx=round_idx,
                authorized_ids=set([node.organization_id]) | additional_orgs_permissions,
                aggregation_id=aggregation_id,
                clean_models=clean_models,
            )
            # keep the states in a list: one/organization
            next_local_states.append(next_local_state)
            next_shared_states.append(next_shared_state)

        self._local_states = next_local_states
        self._shared_states = next_shared_states

    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Predict function for test_data_nodes on which the model have been trained on.

        Args:
            algo (Algo): algo to use for computing the predictions.
            test_data_nodes (List[TestDataNode]): test data nodes to intersect with train data
                nodes to evaluate the model on.
            train_data_nodes (List[TrainDataNode]): train data nodes the model has been trained
                on.
            round_idx (int): round index.

        Raises:
            NotImplementedError: Cannot test on a node we did not train on for now.
        """

        for test_data_node in test_data_nodes:
            matching_train_nodes = [
                train_node
                for train_node in train_data_nodes
                if train_node.organization_id == test_data_node.organization_id
            ]
            if len(matching_train_nodes) == 0:
                node_index = 0
            else:
                node_index = train_data_nodes.index(matching_train_nodes[0])

            assert self._local_states is not None, "Cannot predict if no training has been done beforehand."
            local_state = self._local_states[node_index]

            test_data_node.update_states(
                operation=algo.predict(
                    data_samples=test_data_node.test_data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Testing with {algo.__class__.__name__}",
                ),
                traintask_id=local_state.key,
                round_idx=round_idx,
            )  # Init state for testtask
