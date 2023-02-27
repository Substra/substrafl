from typing import List
from typing import Optional

import numpy as np
from numpy import linalg

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import EmptySharedStatesError
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName
from substrafl.strategies import FedAvg


class FedPCA(FedAvg):
    """Federated Principal Component Analysis strategy.
    This strategy should be used only using TorchFedPCAAlgo

    The goal of this strategy is to perform a principal component analysis (PCA) by
    computing the eigen vectors with highest eigen values of the covariance matrices
    of the data samples. We assume we have clients indexed by $j$, with $n_j$ data samples each.
    We note $N = \sum_j n_j$ the total number of samples. We denote $D$ the dimension of
    the data, and $K$ the number of eigen vectors computed.

    It is based on the Federated sub-space iteration algorithm described
    here https://doi.org/10.1093/bioadv/vbac026 (algorithm 3 of the paper)

    During the process, the local covariance matrices of each center are not shared.
    However, the column-wise mean of each dataset is shared across the centers.

    The Federated Subspace iteration is divded in three steps:

    Step 1: for d= 1, ..., D, each center computes the mean of the $d$ component of
    their dataset and share it to the central aggregator. The central aggregator averages
    the local mean and send to all the clients the global column-wise means of data.

    Step 2: Each center normalize their dataset by substracting the global mean column-wise
    and compute the covariance matrix of their local data after mean-subtraction. We denote by $C_j$
    the local covariance matrices.

    We initialize eig_0: a matrix of size $D \times K$ corresponding to the $K$ eigen
    vectors we want to compute
    Step 3: For a given number of rounds (rounds are labeled by $r$) we perform the following:
        Step 3.1: each center $j$ computes  $eig^r_j = C_j \dot eig^{r-1}_j$
        Step 3.2: the aggregator computes $eig^r = \frac{1}{N}\sum_j n_j eig^r_j
        Step 3:3: the aggregator performs a QR decomposition: eig^r \mapsto QR(eig^r)
                    and sahres $eig^r$ to all the clients

    $eig^r$ will converge to the $K$ eigen-vectors of the global covariance matrix with
    the high eign-values.
    """

    def __init__(self):
        super(FedPCA, self).__init__()

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
            raise ValueError(f"In {self.name} strategy aggregation node cannot be None")

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
        parameters_update_len = len(shared_states[0].parameters_update)
        assert all(
            [len(shared_state.parameters_update) == parameters_update_len for shared_state in shared_states]
        ), "Not the same number of layers for every input parameters."

        n_all_samples = sum([state.n_samples for state in shared_states])

        averaged_states = []
        for idx in range(parameters_update_len):
            states = [state.parameters_update[idx] * (state.n_samples / n_all_samples) for state in shared_states]
            averaged_state_before_qr = np.sum(states, axis=0)
            averaged_state_after_qr, _ = linalg.qr(averaged_state_before_qr.T)
            averaged_state_after_qr = averaged_state_after_qr.T
            averaged_states.append(averaged_state_after_qr)
        return FedAvgAveragedState(avg_parameters_update=averaged_states)
