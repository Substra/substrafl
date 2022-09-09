from typing import List
from typing import Optional

import numpy as np

from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote
from substrafl.schemas import ScaffoldAveragedStates
from substrafl.schemas import ScaffoldSharedState
from substrafl.schemas import StrategyName
from substrafl.strategies.strategy import Strategy


class Scaffold(Strategy):
    """Scaffold strategy.
    Paper: https://arxiv.org/pdf/1910.06378.pdf
    Scaffold is Federated Averaging with control variates.
    By adding auxiliary variables in the clients and server the authors of
    the related paper prove better bounds on the convergence assuming certain
    hypothesis, in particular with non-iid data.

    A round consists in performing a predefined number of forward/backward
    passes on each client, aggregating updates by computing their means and
    distributing the consensus update to all clients. In Scaffold, strategy is
    performed in a centralized way, where a single server or
    ``AggregationNode`` communicates with a number of clients ``TrainDataNode``
    and ``TestDataNode``.

    Args:
        aggregation_lr (float, Optional): Global aggregation rate applied on the averaged weight updates
            (`eta_g` in the paper). Defaults to 1. Must be >=0.
    """

    def __init__(self, aggregation_lr: float = 1):
        super(Scaffold, self).__init__()

        if aggregation_lr < 0:
            raise ValueError("aggregation_lr must be >=0")
        self._aggregation_lr = aggregation_lr
        # current local and share states references of the client used for training
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.SCAFFOLD

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
        clean_models: bool,
    ):
        """One round of the Scaffold strategy consists in:
            - if ``round_idx==1``: initialize the strategy by performing a local update
                (train on n mini-batches) of the models on each train data nodes
            - aggregate the model shared_states
            - set the model weights to the aggregated weights on each train data nodes
            - perform a local update (train on n mini-batches) of the models on each train data nodes

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict methods
            train_data_nodes (typing.List[TrainDataNode]): List of the organizations on which to perform
            local updates aggregation_node (AggregationNode): Node without data, used to perform
                operations on the shared states of the models
            round_idx (int): Round number, it starts at 1.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
        """
        if aggregation_node is None:
            raise ValueError("In Scaffold strategy aggregation node cannot be None")

        if round_idx == 1:
            # Initialization of the strategy by performing a local update on each train data organization
            assert self._local_states is None
            assert self._shared_states is None
            self._perform_local_updates(
                algo=algo,
                train_data_nodes=train_data_nodes,
                current_aggregation=None,
                round_idx=0,
                aggregation_id=aggregation_node.organization_id,
                clean_models=clean_models,
            )

        current_aggregation = aggregation_node.update_states(
            self.avg_shared_states(shared_states=self._shared_states, _algo_name="Aggregating"),  # type: ignore
            round_idx=round_idx,
            authorized_ids=list(set([train_data_node.organization_id for train_data_node in train_data_nodes])),
            clean_models=clean_models,
        )

        self._perform_local_updates(
            algo=algo,
            train_data_nodes=train_data_nodes,
            current_aggregation=current_aggregation,
            round_idx=round_idx,
            aggregation_id=aggregation_node.organization_id,
            clean_models=clean_models,
        )

    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):

        for test_data_node in test_data_nodes:
            matching_train_nodes = [
                train_data_node
                for train_data_node in train_data_nodes
                if train_data_node.organization_id == test_data_node.organization_id
            ]
            if len(matching_train_nodes) == 0:
                raise NotImplementedError("Cannot test on a organization we did not train on for now.")

            train_data_node = matching_train_nodes[0]
            organization_index = train_data_nodes.index(train_data_node)
            assert self._local_states is not None, "Cannot predict if no training has been done beforehand."
            local_state = self._local_states[organization_index]

            test_data_node.update_states(
                operation=algo.predict(
                    data_samples=test_data_node.test_data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Testing with {algo.__class__.__name__}",
                ),
                traintuple_id=local_state.key,
                round_idx=round_idx,
            )  # Init state for testtuple

    def _check_shared_states(self, shared_states: List[ScaffoldSharedState]):
        """Check the Scaffold assumptions: server_control_variate, parameters_update and server_control_variate have the
        same length for all the shared states, and all server_control_variate are equal.

        Args:
            shared_states (List[ScaffoldSharedState]): Shared state returned by the train method of the algorithm for
                each client (e.g. algorithms.pytorch.scaffold.train)
        """
        assert shared_states, "shared_states should contain at least one element"

        for shared_state in shared_states:

            assert isinstance(
                shared_state, ScaffoldSharedState
            ), "shared_state should be an instance of ScaffoldSharedState"

            assert len(shared_state.control_variate_update) == len(
                shared_states[0].control_variate_update
            ), "the length of control_variate_update should be the same for each shared_state"
            assert len(shared_state.parameters_update) == len(
                shared_states[0].parameters_update
            ), "the length of parameters_update should be the same for each shared_state"
            assert len(shared_state.server_control_variate) == len(
                shared_states[0].server_control_variate
            ), "the length of server_control_variate should be the same for each shared_state"

            for c, ci in zip(shared_states[0].server_control_variate, shared_state.server_control_variate):
                np.testing.assert_array_equal(
                    c, ci, err_msg="all server_control_variate in the shared_states are not equal"
                )

        assert (
            len(shared_states[0].control_variate_update)
            == len(shared_states[0].server_control_variate)
            == len(shared_states[0].parameters_update)
        ), "the length of server_control_variate, parameters_update and server_control_variate should be the same"

    def _weight_arrays(
        self,
        client_weight: np.ndarray,
        states_to_aggregate: List[List[np.ndarray]],
        layer_idx: int,
    ) -> List[np.ndarray]:
        """apply the `client_weight` to the arrays of the layer `layer_idx` in `states_to_aggregate`
        Args:
            client_weight (numpy.ndarray): array of shape (num_clients,). Contains the weight of
                each client (n_samples / n_all_samples).
            states_to_aggregate (typing.List[typing.List[numpy.ndarray]]): List of the states on
                which the weights are applied.
            layer_idx (int): the selected layer

        Returns:
            typing.List[numpy.ndarray]: the weighted arrays of the layer `layer_idx`
        """
        weighted_arrays = []
        assert len(client_weight) == len(
            states_to_aggregate
        ), "n_samples_per_client and states_to_aggregate should have the same length"

        for client_idx in range(len(states_to_aggregate)):
            layer_values = states_to_aggregate[client_idx][layer_idx]
            weighted_array = client_weight[client_idx] * layer_values
            weighted_arrays.append(weighted_array)

        return weighted_arrays

    def _update_server_control_variate(
        self,
        server_control_variate: List[np.ndarray],
        control_variate_updates: List[List[np.ndarray]],
        client_weight: np.ndarray,
    ) -> List[np.ndarray]:
        """Updates the server control variate with the weighted average of the control variate updates,
        according to Scaffold paper's Algo steps 16.1 + 17.1: `c = c + sum([client_weight*control_variate_update])`

        Args:
            server_control_variate (typing.List[numpy.ndarray]): the server control variate
            control_variate_updates (typing.List[typing.List[numpy.ndarray]]): the control variate updates from
                the shared state of each client.
            client_weight (numpy.ndarray): array of shape (num_clients,). Contains the weight of each client
                (n_samples / n_all_samples).

        Returns:
            typing.List[numpy.ndarray]: the updated server_control_variate
        """
        updated_server_control_variate = []
        # aggregate the arrays at pos i of each client state
        for layer_idx in range(len(control_variate_updates[0])):
            weighted_arrays = self._weight_arrays(
                client_weight=client_weight,
                states_to_aggregate=control_variate_updates,
                layer_idx=layer_idx,
            )
            # get array of position i for each client state and multiply by the client weight

            weighted_arrays.append(server_control_variate[layer_idx])
            updated_server_control_variate.append(np.sum(weighted_arrays, axis=0))

        return updated_server_control_variate

    def _avg_weight_update(
        self,
        weight_updates: List[List[np.ndarray]],
        client_weight: np.ndarray,
    ) -> List[np.ndarray]:
        """Computes the weighted average of the weight updates and applies the aggregation learning rate,
        according to Scaffold paper's Algo steps 16.2 + 17.2:
        `delta_x = global_lr * sum([client_weight*parameters_update])`

        Args:
            weight_updates (typing.List[typing.List[numpy.ndarray]]): the weight updates of the clients
            client_weight (numpy.ndarray): array of shape (num_clients,). Contains the weight of each client
                (n_samples / n_all_samples).

        Returns:
            typing.List[numpy.ndarray]: the averaged weight updates
        """
        averaged_weight_update = []
        # aggregate the arrays at pos i of each client state
        for layer_idx in range(len(weight_updates[0])):
            weighted_arrays = self._weight_arrays(
                client_weight=client_weight,
                states_to_aggregate=weight_updates,
                layer_idx=layer_idx,
            )
            # we apply global_lr here so we don't have to pass it to the algo (step 17.2)
            averaged_weight_update.append(self._aggregation_lr * np.sum(weighted_arrays, axis=0))

        return averaged_weight_update

    @remote
    def avg_shared_states(self, shared_states: List[ScaffoldSharedState]) -> ScaffoldAveragedStates:
        """Performs the aggregation of the shared states returned by the train
        methods of the user-defined algorithm, according to the server operations of the Scaffold Algo.

        1. Computes the weighted average of the weight updates and applies the aggregation learning rate
        2. Updates the server control variate with the weighted average of the control variate updates

        The average is weighted by the proportion of the number of samples.

        Args:
            shared_states (typing.List[ScaffoldSharedState]): Shared state returned by the train method of
                the algorithm for each client (e.g. algorithms.pytorch.scaffold.train)

        Returns:
            ScaffoldAveragedStates: averaged weight updates and updated server control variate
        """
        # TODO: Do separate function for pytorch to avoid converting in np

        self._check_shared_states(shared_states=shared_states)

        # remove "n_samples" from shared_states and store it in all_samples
        n_samples_per_client = np.array([state.n_samples for state in shared_states])
        client_weight = n_samples_per_client / np.sum(n_samples_per_client)

        # all values should be the same: take the first one
        server_control_variate = shared_states[0].server_control_variate

        averaged_states = ScaffoldAveragedStates(
            server_control_variate=self._update_server_control_variate(
                server_control_variate=server_control_variate,
                control_variate_updates=[shared_state.control_variate_update for shared_state in shared_states],
                client_weight=client_weight,
            ),
            avg_parameters_update=self._avg_weight_update(
                weight_updates=[shared_state.parameters_update for shared_state in shared_states],
                client_weight=client_weight,
            ),
        )

        return averaged_states

    def _perform_local_updates(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        current_aggregation: Optional[SharedStateRef],
        round_idx: int,
        aggregation_id: str,
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
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
        """

        next_local_states = []
        next_shared_states = []

        for i, node in enumerate(train_data_nodes):
            # define composite tuples (do not submit yet)
            # for each composite tuple give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.update_states(
                algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=current_aggregation,
                    _algo_name=f"Training with {algo.__class__.__name__}",
                ),
                local_state=self._local_states[i] if self._local_states is not None else None,
                round_idx=round_idx,
                authorized_ids=list(set([node.organization_id, aggregation_id])),
                clean_models=clean_models,
            )
            # keep the states in a list: one/organization
            next_local_states.append(next_local_state)
            next_shared_states.append(next_shared_state)

        self._local_states = next_local_states
        self._shared_states = next_shared_states
