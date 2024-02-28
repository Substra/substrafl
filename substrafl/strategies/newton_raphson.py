from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import DampingFactorValueError
from substrafl.exceptions import EmptySharedStatesError
from substrafl.exceptions import SharedStatesError
from substrafl.nodes import AggregationNodeProtocol
from substrafl.nodes import TestDataNodeProtocol
from substrafl.nodes import TrainDataNodeProtocol
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote import remote
from substrafl.strategies.schemas import NewtonRaphsonAveragedStates
from substrafl.strategies.schemas import NewtonRaphsonSharedState
from substrafl.strategies.schemas import StrategyName
from substrafl.strategies.strategy import Strategy


class NewtonRaphson(Strategy):
    """Newton-Raphson strategy.

    Newton-Raphson strategy is based on Newton-Raphson distributed method. It leads to a faster convergence than
    a standard FedAvg strategy, however it can only be used on convex problems.

    See https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization for more details.

    In a local step, the first order derivative (gradients) and the second order derivative (Hessian Matrix) of loss
    with respect to weights  is calculated for each node.

    Hessians and gradients are averaged on the aggregation node.

    Updates of the weights are then calculated using the formula:

    .. math::

        update = -\\eta * H^{-1}_\\theta.\\Delta_\\theta

    Where :math:`H` is the Hessian of the loss with respect to :math:`\\theta` and :math:`\\Delta_\\theta` is
    the gradients of the loss with respect to :math:`\\theta`  and :math:`0 < \\eta <= 1` is the damping factor.
    """

    def __init__(
        self,
        algo: Algo,
        damping_factor: float,
        metric_functions: Optional[Union[Dict[str, Callable], List[Callable], Callable]] = None,
    ):
        """
        Args:
            algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
            damping_factor (float): Must be between 0 and 1. Multiplicative coefficient of the parameters update.
                Smaller value for :math:`\\eta` will increase the stability but decrease the speed of convergence of
                the gradient descent. Recommended value: ``damping_factor=0.8``.
            metric_functions (Optional[Union[Dict[str, Callable], List[Callable], Callable]]):
                list of Functions that implement the different metrics. If a Dict is given, the keys will be used to
                register the result of the associated function. If a Function or a List is given, function.__name__
                will be used to store the result.
        """
        super().__init__(algo=algo, damping_factor=damping_factor, metric_functions=metric_functions)

        # States
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

        if not (damping_factor > 0 and damping_factor <= 1):
            raise DampingFactorValueError("damping_factor must be greater than 0 and less than or equal to 1")

        self._damping_factor = damping_factor

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.NEWTON_RAPHSON

    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNodeProtocol],
        aggregation_node: AggregationNodeProtocol,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """One round of the Newton-Raphson strategy consists in:

            - if ``round_ids==0``: initialize the strategy by performing a local update
              of the models on each train data nodes
            - aggregate the model shared_states
            - set the model weights to the aggregated weights on each train data nodes
            - perform a local update of the models on each train data nodes

        Args:
            train_data_nodes (typing.List[TrainDataNodeProtocol]): List of the nodes on which to perform
                local updates
            aggregation_node (AggregationNodeProtocol): node without data, used to perform operations
                on the shared states of the models
            round_idx (int): Round number, it starts at 0.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        if aggregation_node is None:
            raise ValueError("In Newton-Raphson strategy aggregation node cannot be None")

        if round_idx == 1:
            # First round of the strategy by performing a local update on each train data node
            # We consider this step as part of the initialization and tag it as round 0.
            assert self._shared_states is None

            self._perform_local_updates(
                train_data_nodes=train_data_nodes,
                current_aggregation=None,
                round_idx=0,
                aggregation_id=aggregation_node.organization_id,
                additional_orgs_permissions=additional_orgs_permissions or set(),
                clean_models=clean_models,
            )

        current_aggregation = aggregation_node.update_states(
            operation=self.compute_averaged_states(
                shared_states=self._shared_states,
                _algo_name="Aggregating",
            ),
            round_idx=round_idx,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
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
    def compute_averaged_states(
        self,
        shared_states: List[NewtonRaphsonSharedState],
    ) -> NewtonRaphsonAveragedStates:
        """Given the gradients and the Hessian (the second order derivative of loss with respect to weights),
        updates of the weights are calculated using the formula:

        .. math::

            update = -\\eta * H^{-1}_\\theta.\\Delta_\\theta

        Where :math:`H` is the Hessian of the loss with respect to :math:`\\theta` and :math:`\\Delta_\\theta` is
        the gradients of the loss with respect to :math:`\\theta`  and :math:`0 < \\eta <= 1` is the damping factor.

        The average is weighted by the number of samples.

        Example:

            .. code::

                shared_states = [
                    {"gradients": [1, 1, 1], "hessian": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "n_samples": 2},
                    {"gradients": [2, 2, 2], "hessian": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "n_samples": 1}
                ]
                damping_factor = 1

                average = {"gradients": [4/3, 4/3, 4/3], "hessian": [[4/3, 0, 0],[0, 4/3, 0], [0, 0, 4/3]]}

        Args:
            shared_states (List[NewtonRaphsonSharedState]): The list of the shared_state returned by
                the train method of the algorithm for each node.
        Raises:
            EmptySharedStatesError: The train method of your algorithm must return a shared_state
            TypeError: Each shared_state must contains the key **n_samples**, **gradients** and **hessian**
            TypeError: Each shared_state must contains at least one element to average
            TypeError: All the elements of shared_states must be similar (same keys)
            TypeError: All elements to average must be of type np.array

        Returns:
            NewtonRaphsonAveragedStates: A dict containing the parameters updates of each input parameters.
        """

        self._check_shared_states(shared_states)

        n_all_samples = sum([state.n_samples for state in shared_states])

        total_hessians, total_gradient_one_d = None, None
        for idx, state in enumerate(shared_states):
            # Compute average coefficient of the hessian and of the gradients
            sample_coefficient = state.n_samples / n_all_samples

            if idx == 0:
                total_hessians = state.hessian * sample_coefficient
                total_gradient_one_d = (
                    np.concatenate([grad.reshape(-1) for grad in state.gradients]) * sample_coefficient
                )
            else:
                total_hessians += state.hessian * sample_coefficient
                total_gradient_one_d += (
                    np.concatenate([grad.reshape(-1) for grad in state.gradients]) * sample_coefficient
                )

        parameters_update = -self._damping_factor * np.linalg.solve(total_hessians, total_gradient_one_d)
        parameters_update = self._unflatten_array(parameters_update, state.gradients)

        return NewtonRaphsonAveragedStates(parameters_update=parameters_update)

    def _unflatten_array(
        self,
        array_one_d: np.ndarray,
        list_of_array: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Turns a 1-d array into a list of arrays with the same shapes as list_of_array

        Args:
            array_one_d (numpy.ndarray): 1-d array to be unflattened.
            list_of_array (List[numpy.ndarray]): list of arrays with the same shapes as
                the intended new list.

        Returns:
            List[numpy.ndarray]
                The initial array_one_d unflatten with the same shape of list_of_array
        """
        assert len(array_one_d.shape) == 1  # The array to unflatten have to be a 1 dimensional array

        result = []
        current_index = 0

        for array in list_of_array:
            num_params = len(array.ravel())
            result.append(np.array(array_one_d[current_index : current_index + num_params].reshape(array.shape)))
            current_index += num_params

        return result

    def _perform_local_updates(
        self,
        train_data_nodes: List[TrainDataNodeProtocol],
        current_aggregation: Optional[SharedStateRef],
        round_idx: int,
        aggregation_id: str,
        additional_orgs_permissions: set,
        clean_models: bool,
    ):
        """Perform a local update of the model on each train data nodes.

        Args:
            train_data_nodes (typing.List[TrainDataNodeProtocol]): List of the nodes on which to
                perform local updates
            current_aggregation (SharedStateRef, Optional): Reference of an aggregation operation to
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
            # keep the states in a list: one/node
            next_local_states.append(next_local_state)
            next_shared_states.append(next_shared_state)

        self._local_states = next_local_states
        self._shared_states = next_shared_states

    def perform_evaluation(
        self,
        test_data_nodes: List[TestDataNodeProtocol],
        train_data_nodes: List[TrainDataNodeProtocol],
        round_idx: int,
    ):
        """Perform evaluation on test_data_nodes.

        Args:
            test_data_nodes (List[TestDataNodeProtocol]): test data nodes to perform the prediction from the algo on.
            train_data_nodes (List[TrainDataNodeProtocol]): train data nodes the model has been trained
                on.
            round_idx (int): round index.
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

            assert self._local_states is not None, "Cannot evaluate if no training has been done beforehand."
            local_state = self._local_states[node_index]

            test_data_node.update_states(
                operation=self.evaluate(
                    data_samples=test_data_node.data_sample_keys,
                    _algo_name=f"Evaluating with {self.__class__.__name__}",
                ),
                traintask_id=local_state.key,
                round_idx=round_idx,
            )  # Init state for testtask

    def _check_shared_states(self, shared_states: List[NewtonRaphsonSharedState]):
        """Check the Newton Raphson assumptions.

        Args:
            shared_states (List[NewtonRaphsonSharedState]): Shared states returned by the train method of the algorithm
                for each client (e.g. algorithms.pytorch.newton_raphson.train)
        """
        if len(shared_states) == 0:
            raise EmptySharedStatesError(
                "Your shared_states is empty. Please ensure that "
                "the train method of your algorithm returns a NewtonRaphsonSharedState object."
            )

        for shared_state in shared_states:
            if not isinstance(shared_state, NewtonRaphsonSharedState):
                raise SharedStatesError("Shared_state should be an instance of NewtonRaphsonSharedState")
            if not len(shared_state.hessian) == sum([grad.size for grad in shared_state.gradients]):
                raise SharedStatesError("Hessian and gradients should be computed with the same number of parameters")
