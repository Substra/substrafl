import uuid
from typing import List, Optional, TypeVar

import substra

from connectlib.nodes import Node
from connectlib.nodes.references import SharedStateRef
from connectlib.remote.methods import AggregateOperation, RemoteStruct
from connectlib.remote.register import register_aggregation_node_op

SharedState = TypeVar("SharedState")

OperationKey = str


class AggregationNode(Node):
    """The node which applies operations to the shared states which are received from TrainDataNode data operations.
    The result is sent to the TrainDataNode and/or TestDataNode data operations.

    Inherits from :class:`connectlib.nodes.node.Node`

    """

    def update_states(self, operation: AggregateOperation) -> SharedStateRef:
        """Adding an aggregated tuple to the list of operations to be executed by the node during the compute plan.
        This is done in a static way, nothing is submitted to substra.
        This is why the algo key is a RemoteStruct (connectlib local reference of the algorithm)
        and not a substra algo_key as nothing has been submitted yet.

        Args:
            operation (AggregateOperation): Automatically generated structure returned by
            :func:`connectlib.remote.methods.remote` decoractor. This allows to register an
            operation and execute it later on.

        Raises:
            TypeError: operation must be an AggregateOperation, make sure to docorate your (user defined) aggregate
            function of the strategy with @remote.

        Returns:
            SharedStateRef: Identification for the result of this operation.
        """
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
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        dependencies: Optional[List[str]] = None,
    ):
        """Define the algorithms for each operation and submit the aggregated tuple to substra.

        Go through every operation in the computation graph, check what algorithm they use (identified by their RemoteStruct),
        submit it to substra and save the genearated algo_key to self.CACHE.
        If two tuples depend on the same algorithm, the algorithm won't be added twice to substra as
        self.CACHE keeps the submitted algo keys in memory.

        Args:
            client (substra.Client): [description]
            permissions (substra.sdk.schemas.Permissions): [description]
            dependencies (Optional[List[str]], optional): [description]. Defaults to None.
        """
        for tuple in self.tuples:
            if isinstance(tuple["algo_key"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["algo_key"]

                if remote_struct not in self.CACHE:
                    operation_key = register_aggregation_node_op(
                        client,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        dependencies=dependencies,
                    )
                    self.CACHE[remote_struct] = operation_key

                else:
                    operation_key = self.CACHE[remote_struct]

                tuple["algo_key"] = operation_key
