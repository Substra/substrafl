from typing import List
from typing import Optional

import numpy as np

from substrafl.exceptions import EmptySharedStatesError
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote import remote
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName
from substrafl.strategies.centralized_strategies.centralized_strategy import CentralizedStrategy


class FedAvg(CentralizedStrategy):
    """Federated averaging strategy.

    Federated averaging is the simplest federating strategy.
    A round consists in performing a predefined number of forward/backward
    passes on each client, aggregating updates by computing their means and
    distributing the consensus update to all clients. In FedAvg, strategy is
    performed in a centralized way, where a single server or
    ``AggregationNode`` communicates with a number of clients ``TrainDataNode``
    and ``TestDataNode``.

    Formally, if :math:`w_t` denotes the parameters of the model at round
    :math:`t`, a single round consists in the following steps:

    .. math::

      \\Delta w_t^{k} = \\mathcal{O}^k_t(w_t| X_t^k, y_t^k, m)
      \\Delta w_t = \\sum_{k=1}^K \\frac{n_k}{n} \\Delta w_t^k
      w_{t + 1} = w_t + \\Delta w_t

    where :math:`\\mathcal{O}^k_t` is the local optimizer algorithm of client
    :math:`k` taking as argument the averaged weights as well as the
    :math:`t`-th batch of data for local worker :math:`k` and the number of
    local updates :math:`m` to perform, and where :math:`n_k` is the number of
    samples for worker :math:`k`, :math:`n = \\sum_{k=1}^K n_k` is the total
    number of samples.
    """

    def __init__(self):
        super(FedAvg, self).__init__()

        # current local and share states references of the client
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.FEDERATED_AVERAGING

    @remote
    def compute_aggregated_states(self, shared_states: List[FedAvgSharedState]) -> FedAvgAveragedState:
        """Compute the weighted average of all elements returned by the train
        methods of the user-defined algorithm.
        The average is weighted by the proportion of the number of samples.

        Example:

            .. code-block:: python

                shared_states = [
                    {"weights": [3, 3, 3], "gradient": [4, 4, 4], "n_samples": 20},
                    {"weights": [6, 6, 6], "gradient": [1, 1, 1], "n_samples": 40},
                ]
                result = {"weights": [5, 5, 5], "gradient": [2, 2, 2]}

        Args:
            shared_states (typing.List[FedAvgSharedState]): The list of the
                shared_state returned by the train method of the algorithm for each organization.

        Raises:
            EmptySharedStatesError: The train method of your algorithm must return a shared_state
            TypeError: Each shared_state must contains the key **n_samples**
            TypeError: Each shared_state must contains at least one element to average
            TypeError: All the elements of shared_states must be similar (same keys)
            TypeError: All elements to average must be of type np.ndarray

        Returns:
            FedAvgAveragedState: A dict containing the weighted average of each input parameters
            without the passed key "n_samples".
        """
        if len(shared_states) == 0:
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

        averaged_states = list()
        for idx in range(len(shared_states[0].parameters_update)):
            states = list()
            for state in shared_states:
                states.append(state.parameters_update[idx] * (state.n_samples / n_all_samples))
            averaged_states.append(np.sum(states, axis=0))

        return FedAvgAveragedState(avg_parameters_update=averaged_states)
