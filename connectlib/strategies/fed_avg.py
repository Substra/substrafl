from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from connectlib.algorithms import Algo
from connectlib.nodes import AggregationNode
from connectlib.nodes import TestDataNode
from connectlib.nodes import TrainDataNode
from connectlib.nodes.references.local_state import LocalStateRef
from connectlib.nodes.references.shared_state import SharedStateRef
from connectlib.remote import remote
from connectlib.strategies.strategy import Strategy


class FedAVG(Strategy):
    """Federated averaging strategy.
    Federated averaging is the simplest federating strategy.
    A round consists in a performing a predefined number of forward/backward
    passes on each client, aggregating updates by computing their means and
    distributing the consensus update to all clients. In FedAvg, strategy is
    performed in a centralized way, where a single server or
    :term:`aggregation node` communicates with a number of clients :term:`TrainDataNode`
    and :term:`TestDataNode`.

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

    **Adventages**:
    - decoupling of model training from the need for direct access to the raw
    training data
    - significant reduction of privacy and security risks by (limiting the attack
    surface only to the nodes)

    **Disadventages**:
    - each :term:`node` is connected to :term:`aggregation node`. The communication
    rounds between each node and aggregation node might be frequent which might be expensive
    - some trust of the server coordinating the training is still required.
    """

    def __init__(self):
        super(FedAVG, self).__init__()

        # States
        self.local_states: Optional[List[LocalStateRef]] = None
        self.avg_shared_state: Optional[SharedStateRef] = None

    @remote
    def avg_shared_states(self, shared_states: List[Dict[str, Union[int, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """Compute the weighted average of all elements returned by the train
        methods of the user defined algorithm.
        The average is weighted by the number of samples.

        E.g.: shared_states = [
            {"weights": [3, 3, 3], "gradient": [4, 4, 4], "n_samples": 20},
            {"weights": [6, 6, 6], "gradient": [1, 1, 1], "n_samples": 40},
        ]

        result = {"weights": [5, 5, 5], "gradient": [2, 2, 2]}

        Args:
            shared_states (List[Weights]): The list of the shared_state returned by
            the train method of the algorithm for each node.

        Raises:
            TypeError: The train method of your algorithm must return a shared_state
            TypeError: Each shared_state must contains the key **n_samples**
            TypeError: Each shared_state must contains at least one element to average
            TypeError: All the elements of shared_states must be similar (same keys)
            TypeError: All elements to average must be of type np.array

        Returns:
            Weights:
        """
        # get keys
        # TODO: for now an ugly conversion to list to be able to remove the element, improve
        # shared_states -> list of weights at each client
        if len(shared_states) == 0:
            raise TypeError(
                "Your shared_states is empty. Please ensure that "
                "the train method of your algorithm always returns nothing. "
                "It must returns a dict containing n_samples(int) and at least one other key (np.array)."
            )
        if not (all(["n_samples" in shared_state.keys() for shared_state in shared_states])):
            raise TypeError(
                "n_samples must be a key from all your shared_state. "
                "This must be set in the returned element of the train method from your algorithm. "
                "It must be a dict containing n_samples(int) and at least one other key (np.array)."
            )
        if len(shared_states[0].keys()) == 1:
            raise TypeError(
                "shared_state must contains at least one element to average."
                "This must be set it in the returned element of the train method from your algorithm. "
                "It must be a dict containing n_samples(int) and at least one other key (np.array)."
            )
        for shared_state in shared_states:
            for k, v in shared_state.items():
                if k != "n_samples" and not (isinstance(v, np.ndarray)):
                    raise TypeError(
                        "Except for the `n_samples`, the types of your shared_state must be numpy array. "
                        f"'{k}' is of type '{type(v)}' ."
                        "It must be a dict containing n_samples(int) and at least one other key (np.array)."
                    )

        averaged_states = {}
        all_samples = np.array([state.pop("n_samples") for state in shared_states])
        n_all_samples = np.sum(all_samples)

        for key in shared_states[0].keys():
            # take each of the states,and multiply by the number of samples for each client and sum
            # For now each value of shared_states is an np.array.
            states = []
            for n_samples, state in zip(all_samples, shared_states):
                states.append(state[key] * n_samples)
            states = np.sum(states, axis=0)
            # divide by the sum of all the samples
            averaged_states[key] = states / n_all_samples

        return averaged_states

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        """One round of the Federated Averaging strategy:
            - if they exist, set the model weights to the aggregated weights on each train data nodes
            - perform a local update (train on n minibatches) of the models on each train data nodes
            - aggregate the model shared_states

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local updates
            aggregation_node (AggregationNode): Node without data, used to perform operations on the shared states
                of the models
        """
        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = None
            if self.local_states is not None:
                previous_local_state = self.local_states[i]

            # define composite tuples (do not submit yet)
            # for each composite tuple give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.update_states(
                algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=self.avg_shared_state,
                ),
                local_state=previous_local_state,
            )
            # keep the states in a list: one/node
            next_local_states.append(next_local_state)
            states_to_aggregate.append(next_shared_state)

        avg_shared_state = aggregation_node.update_states(
            self.avg_shared_states(shared_states=states_to_aggregate)  # type: ignore
        )

        self.local_states = next_local_states
        self.avg_shared_state = avg_shared_state

    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
    ):

        for test_node in test_data_nodes:
            matching_train_nodes = [
                train_node for train_node in train_data_nodes if train_node.node_id == test_node.node_id
            ]
            if len(matching_train_nodes) == 0:
                raise NotImplementedError("Cannot test on a node we did not train on for now.")

            train_node = matching_train_nodes[0]
            node_index = train_data_nodes.index(train_node)
            previous_local_state = self.local_states[node_index] if self.local_states is not None else None
            assert previous_local_state is not None

            # Since the training round ends on an aggregation on the aggregation node
            # we need to get the aggregated gradients back to the test node
            traintuple_id_ref, _ = train_node.update_states(
                # here we could also use algo.train or whatever method marked as @remote_data
                # in the algo because fake_traintuple is true so the method name and the method
                # are not used
                operation=algo.predict(
                    data_samples=[train_node.data_sample_keys[0]],
                    shared_state=self.avg_shared_state,
                    fake_traintuple=True,
                ),
                local_state=previous_local_state,
            )

            test_node.update_states(traintuple_id=traintuple_id_ref.key)  # Init state for testtuple
