import numpy as np

from typing import List

from connectlib.algorithms import Algo
from connectlib.algorithms.algo import Weights
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.strategies.strategy import Strategy
from connectlib.remote import remote


class FedAVG(Strategy):
    def __init__(self, num_updates: int):
        self.num_updates = num_updates
        super(FedAVG, self).__init__(num_updates=num_updates, seed=42)

        # States
        self.local_states = None
        self.avg_shared_state = None

    @remote
    def avg_shared_states(self, shared_states: List[Weights]) -> Weights:
        # get keys
        keys = shared_states[0].keys()

        # average weights
        averaged_states = {}
        for key in keys:
            states = np.stack([state[key] for state in shared_states])
            averaged_states[key] = np.mean(states, axis=0)

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
        - aggregate the model gradients

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local updates
            aggregation_node (AggregationNode): Node without data, used to perform operations on the shared states of the models
        """
        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )

            # define composite tuples (do not submit yet)
            # for each composite tuple give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.compute(
                algo.train(  # type: ignore
                    node.data_sample_keys,  # TODO: change this, we don't give all the dataset to train on for one strategy round
                    shared_state=self.avg_shared_state,
                    num_updates=self.num_updates,
                ),
                local_state=previous_local_state,
            )
            # keep the states in a list: one/node
            next_local_states.append(next_local_state)
            states_to_aggregate.append(next_shared_state)

        avg_shared_state = aggregation_node.compute(
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
        # TODO: REMOVE WHEN HACK IS FIXED
        traintuple_ids = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )

            assert previous_local_state is not None

            traintuple_id_ref, _ = node.compute(
                # here we could also use algo.train or whatever method marked as @remote_data
                # in the algo
                # because fake_traintuple is true so the method name and the method
                # are not used
                algo.predict(  # type: ignore
                    [node.data_sample_keys[0]],
                    shared_state=self.avg_shared_state,
                    fake_traintuple=True,
                ),
                local_state=previous_local_state,
            )
            traintuple_ids.append(traintuple_id_ref.key)

        for i, node in enumerate(test_data_nodes):
            node.compute(traintuple_ids[i])  # compute testtuple
