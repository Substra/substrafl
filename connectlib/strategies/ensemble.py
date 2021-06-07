from typing import List

from connectlib.algorithms import Algo
from connectlib.algorithms.algo import Weights
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.strategies.strategy import Strategy
from connectlib.remote import remote


class Ensemble(Strategy):
    def __init__(self, num_updates: int):
        self.num_updates = num_updates

        super(Ensemble, self).__init__(num_updates=num_updates, seed=42)

        # States
        self.local_states = None
        self.shared_states = None
        self.stacked_shared_states = None

    @remote
    def stack_states(self, shared_states: List[Weights]) -> List[Weights]:
        return shared_states

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        next_local_states = []
        next_shared_states = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )
            previous_shared_state = (
                self.shared_states[i] if self.shared_states is not None else None
            )

            next_local_state, next_shared_state = node.compute(
                algo.train(
                    node.data_sample_keys,
                    shared_state=previous_shared_state,
                    num_updates=self.num_updates,
                ),
                local_state=previous_local_state,
            )
            next_local_states.append(next_local_state)
            next_shared_states.append(next_shared_state)

        self.stacked_shared_states = aggregation_node.compute(
            self.stack_states(shared_states=next_shared_states)
        )

        self.local_states = next_local_states
        self.shared_states = next_shared_states

    def predict(self, algo: Algo, test_data_nodes: List[TestDataNode]):
        raise NotImplementedError
