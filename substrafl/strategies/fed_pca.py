from typing import List
from typing import Optional

import numpy as np
from numpy import linalg

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import EmptySharedStatesError
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName
from substrafl.strategies import FedAvg


class FedPCA(FedAvg):
    """Federated Principal Component Analysis strategy."""

    def __init__(self):
        super(FedPCA, self).__init__()

        # current local and share states references of the client
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.FEDERATED_PCA

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """One round of the Federated PCA"""
        if aggregation_node is None:
            raise ValueError("In FedAvg strategy aggregation node cannot be None")

        if round_idx == 0:
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

        elif round_idx < 2:
                function_to_execute = self.avg_shared_states
            else:
                function_to_execute = self.avg_shared_states_with_qr

            current_aggregation = aggregation_node.update_states(
                function_to_execute(shared_states=self._shared_states, _algo_name="Aggregating"),  # type: ignore
                round_idx=round_idx,
                authorized_ids={train_data_node.organization_id for train_data_node in train_data_nodes},
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

    @remote
    def avg_shared_states_with_qr(self, shared_states: List[FedAvgSharedState]) -> FedAvgAveragedState:

        if not shared_states:
            raise EmptySharedStatesError(
                "Your shared_states is empty. Please ensure that "
                "the train method of your algorithm returns a FedAvgSharedState object."
            )

        assert all(
            [
                len(shared_state.parameters_update) == len(shared_states[0].parameters_update)
                for shared_state in shared_states
            ]
        ), "Not the same number of layers for every input parameters."

        n_all_samples = sum([state.n_samples for state in shared_states])

        averaged_states = []
        for idx in range(len(shared_states[0].parameters_update)):
            states = [ state.parameters_update[idx] * (state.n_samples / n_all_samples)
                                            for state in shared_states]
            averaged_state_before_qr = np.sum(states, axis=0)
            averaged_state_after_qr, _ = linalg.qr(averaged_state_before_qr.T)
            averaged_state_after_qr = averaged_state_after_qr.T
            averaged_states.append(averaged_state_after_qr)

        return FedAvgAveragedState(avg_parameters_update=averaged_states)
