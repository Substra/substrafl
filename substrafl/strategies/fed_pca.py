import logging
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import EmptySharedStatesError
from substrafl.nodes import AggregationNodeProtocol
from substrafl.nodes import TestDataNodeProtocol
from substrafl.nodes import TrainDataNodeProtocol
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote import remote
from substrafl.strategies.schemas import FedPCAAveragedState
from substrafl.strategies.schemas import FedPCASharedState
from substrafl.strategies.schemas import StrategyName
from substrafl.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class FedPCA(Strategy):
    """Federated Principal Component Analysis strategy.

    The goal of this strategy is to perform a **principal component analysis** (PCA) by
    computing the eigen vectors with highest eigen values of the covariance matrices
    regarding the provided data.

    We assume we have clients indexed by :math:`j`, with :math:`n_j` samples each.

    We note :math:`N = \\sum_j n_j` the total number of samples. We denote :math:`D` the dimension of
    the data, and :math:`K` the number of eigen vectors computed.

    This strategy implementation is based on the **federated iteration algorithm** described
    by:

        *Anne Hartebrodt, Richard Röttger,* **Federated horizontally partitioned principal component analysis
        for biomedical applications,** *Bioinformatics Advances, Volume 2, Issue 1, 2022, vbac026,*
        https://doi.org/10.1093/bioadv/vbac026 *(algorithm 3 of the paper)*

    During the process, the local covariance matrices of each center are not shared.
    However, the column-wise mean of each dataset is shared across the centers.

    The federated iteration is divided in **three steps**.

    Step 1:
        - For :math:`d= 1, ..., D`, each center computes the mean of the :math:`d` component of
          their dataset and share it to the central aggregator. The central aggregator averages
          the local mean and send to all the clients the global column-wise means of data.

    Step 2:
        - Each center normalize their dataset by subtracting the global mean column-wise
          and compute the covariance matrix of their local data after mean-subtraction. We denote by :math:`C_j`
          the local covariance matrices.

        We initialize :math:`eig_0`: a matrix of size :math:`D \\times K` corresponding to the :math:`K` eigen
        vectors we want to compute

    Step 3, for a given number of rounds (rounds are labeled by :math:`r`) we perform the following:
        Step 3.1:
            - Each center :math:`j` computes  :math:`eig^r_j = C_j.eig^{r-1}_j`
        Step 3.2:
            - The aggregator computes :math:`eig^r = \\frac{1}{N}\\sum_j n_j eig^r_j`
        Step 3:3:
            - The aggregator performs a QR decomposition: :math:`eig^r \\mapsto QR(eig^r)`
              and shares :math:`eig^r` to all the clients

            :math:`eig^r` will converge to the :math:`K` eigen-vectors of the global covariance matrix with
            the highest eigen-values.
    """

    def __init__(
        self,
        algo: Algo,
        metric_functions: Optional[Union[Dict[str, Callable], List[Callable], Callable]] = None,
    ):
        """
        Args:
            algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
            metric_functions (Optional[Union[Dict[str, Callable], List[Callable], Callable]]):
                list of Functions that implement the different metrics. If a Dict is given, the keys will be used to
                register the result of the associated function. If a Function or a List is given, function.__name__
                will be used to store the result.
        """
        super().__init__(algo=algo, metric_functions=metric_functions)

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

    def perform_evaluation(
        self,
        test_data_nodes: List[TestDataNodeProtocol],
        train_data_nodes: List[TrainDataNodeProtocol],
        round_idx: int,
    ) -> None:
        """Perform evaluation on test_data_nodes. Perform prediction before round 3 is not take into account
        as all objects to compute prediction are not initialize before the second round.

        Args:
            test_data_nodes (List[TestDataNodeProtocol]): test data nodes to perform the prediction from the algo on.
            train_data_nodes (List[TrainDataNodeProtocol]): train data nodes the model has been trained
                on.
            round_idx (int): round index.
        """
        if round_idx < 3:
            logger.warning(f"Evaluation ignored before round 3 for {self.name} (pre-processing rounds).")
            return

        for test_data_node in test_data_nodes:
            matching_train_nodes = [
                train_data_node
                for train_data_node in train_data_nodes
                if train_data_node.organization_id == test_data_node.organization_id
            ]
            if len(matching_train_nodes) == 0:
                node_index = 0
            else:
                node_index = train_data_nodes.index(matching_train_nodes[0])

            assert self._local_states is not None, "Cannot evaluate if no training has been done beforehand."
            local_state = self._local_states[node_index]

            test_data_node.update_states(
                traintask_id=local_state.key,
                operation=self.evaluate(
                    data_samples=test_data_node.data_sample_keys,
                    _algo_name=f"Evaluating with {self.__class__.__name__}",
                ),
                round_idx=round_idx,
            )  # Init state for testtask

    def perform_round(
        self,
        train_data_nodes: List[TrainDataNodeProtocol],
        aggregation_node: AggregationNodeProtocol,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ) -> None:
        """The Federated Principal Component Analysis strategy uses the first two rounds as pre-processing rounds.

        Three type of rounds:
            - Compute the average mean on all centers at round 1.
            - Compute the local covariance matrix of each center at round 2.
            - Use the local covariance matrices to compute the orthogonal matrix for every next rounds.

        Args:
            train_data_nodes (typing.List[TrainDataNodeProtocol]): List of the nodes on which to perform
                local updates.
            aggregation_node (AggregationNodeProtocol): Node without data, used to perform
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

        if round_idx == 1:
            # Initialization of the strategy by performing a local update on each train data organization
            assert self._shared_states is None
            self._perform_local_updates(
                train_data_nodes=train_data_nodes,
                current_aggregation=None,
                round_idx=round_idx,
                aggregation_id=aggregation_node.organization_id,
                additional_orgs_permissions=additional_orgs_permissions or set(),
                clean_models=clean_models,
            )
            # At round 1, we only want to send the average of the shared states.
            function_to_execute = self.avg_shared_states
        else:
            # For the next round, we want to send the orthogonal matrix as shared states
            # to average. We use avg_shared_states_with_qr to compute it.
            function_to_execute = self.avg_shared_states_with_qr

        current_aggregation = aggregation_node.update_states(
            operation=function_to_execute(shared_states=self._shared_states, _algo_name="Aggregating"),
            round_idx=round_idx,
            authorized_ids={train_data_node.organization_id for train_data_node in train_data_nodes},
            clean_models=clean_models,
        )

        self._perform_local_updates(
            train_data_nodes=train_data_nodes,
            current_aggregation=current_aggregation,
            round_idx=round_idx,
            aggregation_id=aggregation_node.organization_id,
            additional_orgs_permissions=additional_orgs_permissions or set(),
            clean_models=clean_models,
        )

    @remote
    def avg_shared_states(self, shared_states: List[FedPCASharedState]) -> FedPCAAveragedState:
        """Compute the weighted average of all elements returned by the train
        methods of the user-defined algorithm.
        The average is weighted by the proportion of the number of samples.

        Example:

            .. code-block:: python

                shared_states = [
                    {"parameters_update": [3, 6, 1], "n_samples": 20},
                    {"parameters_update": [6, 3, 1], "n_samples": 40},
                ]
                result = {"parameters_update": [5, 4, 1]}

        Args:
            shared_states (typing.List[FedPCASharedState]): The list of the
                shared_state returned by the train method of the algorithm for each organization.

        Raises:
            EmptySharedStatesError: The train method of your algorithm must return a shared_state
            TypeError: Each shared_state must contains the key **n_samples**
            TypeError: Each shared_state must contains at least one element to average
            TypeError: All the elements of shared_states must be similar (same keys)
            TypeError: All elements to average must be of type np.ndarray

        Returns:
            FedPCAAveragedState: A dict containing the weighted average of each input parameters
            without the passed key "n_samples".
        """
        if len(shared_states) == 0:
            raise EmptySharedStatesError(
                "Your shared_states is empty. Please ensure that "
                "the train method of your algorithm returns a FedPCASharedState object."
            )

        parameters_update_len = len(shared_states[0].parameters_update)
        assert all(
            [len(shared_state.parameters_update) == parameters_update_len for shared_state in shared_states]
        ), "Not the same number of layers for every input parameters."

        n_all_samples = sum([state.n_samples for state in shared_states])

        averaged_states = []
        for idx in range(parameters_update_len):
            states = [state.parameters_update[idx] * (state.n_samples / n_all_samples) for state in shared_states]
            averaged_states.append(np.sum(states, axis=0))

        return FedPCAAveragedState(avg_parameters_update=averaged_states)

    @remote
    def avg_shared_states_with_qr(self, shared_states: List[FedPCASharedState]) -> FedPCAAveragedState:
        """Compute the weighted average of all elements returned by the train
        methods of the user-defined algorithm and factorize the obtained matrix
        with a QR decomposition, where Q is orthonormal and R is upper-triangular.

        The returned FedPCAAveragedState the **Q matrix only**.

        Args:
            shared_states (typing.List[FedPCASharedState]): The list of the
                shared_state returned by the train method of the algorithm for each organization.

        Raises:
            EmptySharedStatesError: The train method of your algorithm must return a shared_state

        Returns:
            FedPCAAveragedState: returned the Q matrix as a FedPCAAveragedState.
        """
        if not shared_states:
            raise EmptySharedStatesError(
                "Your shared_states is empty. Please ensure that "
                "the train method of your algorithm returns a FedPCASharedState object."
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
            averaged_state_after_qr, _ = np.linalg.qr(averaged_state_before_qr.T)

            averaged_states.append(averaged_state_after_qr.T)
        return FedPCAAveragedState(avg_parameters_update=averaged_states)

    def _perform_local_updates(
        self,
        train_data_nodes: List[TrainDataNodeProtocol],
        current_aggregation: Optional[SharedStateRef],
        round_idx: int,
        aggregation_id: str,
        additional_orgs_permissions: set,
        clean_models: bool,
    ):
        """Perform a local update (train on n mini-batches) of the models
        on each train data nodes.

        Args:
            train_data_nodes (typing.List[TrainDataNodeProtocol]): List of the organizations on which to perform
            local updates current_aggregation (SharedStateRef, Optional): Reference of an aggregation operation to
                be passed as input to each local training
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
            # define train tasks (do not submit yet)
            # for each train task give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.update_states(
                operation=self.algo.train(
                    node.data_sample_keys,
                    shared_state=current_aggregation,
                    _algo_name=f"Training with {self.algo.__class__.__name__}",
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
