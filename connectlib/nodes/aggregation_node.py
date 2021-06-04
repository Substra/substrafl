import substra
import uuid

from typing import Optional, List, TypeVar, Dict

from connectlib.nodes.references import SharedStateRef
from connectlib.nodes.register import register_aggregate_op
from connectlib.operations import AggregateOp
from connectlib.operations.blueprint import Blueprint
from connectlib.nodes import Node

SharedState = TypeVar("SharedState")

OperationKey = str


class AggregationNode(Node):
    CACHE: Dict[Blueprint[AggregateOp], OperationKey] = {}

    def compute(
        self,
        operation: Blueprint[AggregateOp],
        shared_states: Optional[List[SharedStateRef]] = None,
    ) -> SharedStateRef:
        if not isinstance(operation, Blueprint):
            raise TypeError(
                "operation must be a Blueprint",
                f"Given: {type(operation)}",
                "Have you decorated your AggregateOp with @blueprint?",
            )
        if not isinstance(operation.cls, AggregationNode):
            raise TypeError(
                "operation must be a Blueprint of an AggregateOp",
                f"Given: {type(operation.cls)}",
            )

        op_id = uuid.uuid4().hex

        aggregate_tuple = {
            "algo_key": operation,
            "worker": self.node_id,
            "in_models_ids": [ref.key for ref in shared_states]
            if shared_states is not None
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
            if isinstance(tuple["algo_key"], Blueprint):
                blueprint: Blueprint[AggregateOp] = tuple["algo_key"]

                if blueprint not in self.CACHE:
                    operation_key = register_aggregate_op(
                        client, blueprint=blueprint, permisions=permissions
                    ).key
                    self.CACHE[blueprint] = operation_key

                else:
                    operation_key = self.CACHE[blueprint]

                tuple["algo_key"] = operation_key
