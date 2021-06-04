import uuid
import substra

import numpy as np

from typing import List, Optional, Dict

from connectlib.operations import AggregateOp
from connectlib.operations.blueprint import blueprint
from connectlib.nodes import TrainDataNode, AggregationNode
from connectlib.nodes.pointers import LocalStatePointer, SharedStatePointer, AlgoPointer
from connectlib.nodes.register import register_aggregate_op
from connectlib.strategies.strategy import Strategy

SharedState = Dict[str, np.array]


@blueprint
class AvgAggregateOp(AggregateOp):
    def __call__(self, shared_states: List[SharedState]) -> SharedState:
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

        self.avg_op = None

    def perform_round(
        self,
        client: substra.Client,
        algo: AlgoPointer,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        local_states: Optional[List[LocalStatePointer]],
        shared_state: Optional[SharedStatePointer],
    ):
        if self.avg_op is None:
            authorized_ids = [aggregation_node.node_id] + [
                node.node_id for node in train_data_nodes
            ]
            permissions = substra.sdk.schemas.Permissions(
                public=False, authorized_ids=authorized_ids
            )
            self.avg_op = register_aggregate_op(
                client, blueprint=AvgAggregateOp(), permisions=permissions
            )

        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = local_states[i] if local_states is not None else None

            next_local_state, next_shared_state = node.add(
                algo,
                local_state_pointer=previous_local_state,
                shared_state_pointer=shared_state,
            )
            next_local_states.append(next_local_state)
            states_to_aggregate.append(next_shared_state)

        avg_shared_state = aggregation_node.add(
            self.avg_op, shared_state_pointers=states_to_aggregate
        )

        return next_local_states, avg_shared_state
