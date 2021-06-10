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
        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )

            next_local_state, next_shared_state = node.compute(
                algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=self.avg_shared_state,
                    num_updates=self.num_updates,
                ),
                local_state=previous_local_state,
            )
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

            traintuple_id, _ = node.compute(
                algo.predict(  # type: ignore
                    [node.data_sample_keys[0]],
                    shared_state=self.avg_shared_state,
                    num_updates=self.num_updates,
                    fake_traintuple=True,
                ),
                local_state=previous_local_state,
            )
            traintuple_ids.append(traintuple_id)

        for i, node in enumerate(test_data_nodes):
            testtuple = node.compute(traintuple_ids[i])
