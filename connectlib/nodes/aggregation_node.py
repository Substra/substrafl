import uuid

from typing import Optional, List, TypeVar

from connectlib.nodes.pointers import SharedStatePointer, AggregatePointer
from connectlib.nodes import Node

SharedState = TypeVar("SharedState")


class AggregationNode(Node):
    def add(
        self,
        operation_pointer: AggregatePointer,
        shared_state_pointers: Optional[List[SharedStatePointer]] = None,
    ) -> SharedStatePointer:
        op_id = uuid.uuid4().hex

        aggregate_tuple = {
            "algo_key": operation_pointer.key,
            "worker": self.node_id,
            "in_models_ids": [pointer.key for pointer in shared_state_pointers]
            if shared_state_pointers is not None
            else None,
            "tag": "aggregate",
            "aggregatetuple_id": op_id,
        }
        self.tuples.append(aggregate_tuple)

        return SharedStatePointer(key=op_id)
