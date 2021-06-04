import uuid
import substra

import numpy as np

from typing import List, Optional, Dict

from connectlib.operations import AggregateOp
from connectlib.operations.blueprint import blueprint
from connectlib.nodes import TrainDataNode, AggregationNode
from connectlib.nodes.references import LocalStateRef, SharedStateRef, AlgoRef
from connectlib.nodes.register import register_aggregate_op
from connectlib.strategies.strategy import Strategy

SharedState = Dict[str, np.array]


@blueprint
class AvgAggregateOp(AggregateOp):
    def __call__(self, shared_states: List[SharedState]) -> SharedState:
        print(shared_states)
        # get keys
        keys = shared_states[0].keys()

        # average weights
        averaged_states = {}
        for key in keys:
            states = np.stack([state[key] for state in shared_states])
            averaged_states[key] = np.mean(states, axis=0)

        return averaged_states


class FedAVG(Strategy):
    def __init__(self, num_rounds: int, num_updates: int):
        self.num_rounds = num_rounds
        self.num_updates = num_updates

        self.avg_agg_op = None

    def perform_round(
        self,
        algo: AlgoRef,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        local_states: Optional[List[LocalStateRef]],
        shared_state: Optional[SharedStateRef],
    ):
        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = local_states[i] if local_states is not None else None

            next_local_state, next_shared_state = node.compute(
                algo,
                local_state=previous_local_state,
                shared_state=shared_state,
            )
            next_local_states.append(next_local_state)
            states_to_aggregate.append(next_shared_state)

        avg_shared_state = aggregation_node.compute(
            AvgAggregateOp(), shared_states=states_to_aggregate
        )

        return next_local_states, avg_shared_state
