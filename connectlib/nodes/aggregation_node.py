import substra
import uuid

from typing import TypeVar, Dict

from connectlib.nodes.references import SharedStateRef
from connectlib.nodes import Node
from connectlib.remote.methods import RemoteStruct, AggregateOperation
from connectlib.remote.register import register_aggregate_node_op

SharedState = TypeVar("SharedState")

OperationKey = str


class AggregationNode(Node):
    CACHE: Dict[RemoteStruct, OperationKey] = {}

    def compute(self, operation: AggregateOperation) -> SharedStateRef:
        if not isinstance(operation, AggregateOperation):
            raise TypeError(
                "operation must be a AggregateOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote?",
            )

        op_id = uuid.uuid4().hex

        aggregate_tuple = {
            "algo_key": operation.remote_struct,
            "worker": self.node_id,
            "in_models_ids": [ref.key for ref in operation.shared_states]
            if operation.shared_states is not None
            else None,
            "tag": "aggregate",
            "aggregatetuple_id": op_id,
        }
        self.tuples.append(aggregate_tuple)

        return SharedStateRef(key=op_id)

    def register_operations(
        self, client: substra.Client, permissions: substra.sdk.schemas.Permissions
    ):
        for tuple in self.tuples:
            if isinstance(tuple["algo_key"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["algo_key"]

                if remote_struct not in self.CACHE:
                    operation_key = register_aggregate_node_op(
                        client, remote_struct=remote_struct, permisions=permissions
                    )
                    self.CACHE[remote_struct] = operation_key

                else:
                    operation_key = self.CACHE[remote_struct]

                tuple["algo_key"] = operation_key
