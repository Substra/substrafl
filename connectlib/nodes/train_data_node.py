import substra
import uuid

from typing import List, Optional, Tuple

from connectlib.nodes.references import (
    LocalStateRef,
    SharedStateRef,
)
from connectlib.nodes import Node
from connectlib.remote.methods import RemoteStruct, DataOperation
from connectlib.remote.register import register_data_node_op


class TrainDataNode(Node):
    def __init__(
        self,
        node_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
    ):
        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys

        super(TrainDataNode, self).__init__(node_id)

    def compute(
        self,
        operation: DataOperation,
        local_state: Optional[LocalStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        if not isinstance(operation, DataOperation):
            raise TypeError(
                "operation must be a DataOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote_data?",
            )

        op_id = uuid.uuid4().hex

        # Create the composite traintuple Substra spec
        # The only difference is the algo_key: for now it contains the remote struct
        # When 'register_operations' is called, the algo is created and the field contains
        # the actual algo key
        # This is done to avoid registering duplicate algos to the platform
        composite_traintuple = {
            "algo_key": operation.remote_struct,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": operation.data_samples,
            # in_head_model_id is a user-defined id (last composite_traintuple id)
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": operation.shared_state.key
            if operation.shared_state is not None
            else None,  # user-defined id (last central node task id)
            "tag": "train",
            "composite_traintuple_id": op_id,
            "metadata": dict(),  # TODO: might add info here so that on the platform we see what the tuple does ?
        }

        self.tuples.append(composite_traintuple)

        return LocalStateRef(op_id), SharedStateRef(op_id)

    def register_operations(
        self, client: substra.Client, permissions: substra.sdk.schemas.Permissions
    ):
        """Define the algorithms for each operation

        Go through every operation in the computation graph, check what algorithm they use
        and build the algorithm cache.

        Args:
            client (substra.Client): Substra client for the node.
            permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
        """
        for tuple in self.tuples:
            if tuple.get("out_trunk_model_permissions", None) is None:
                tuple["out_trunk_model_permissions"] = permissions

            if isinstance(tuple["algo_key"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["algo_key"]

                if remote_struct not in self.CACHE:
                    operation_key = register_data_node_op(
                        client, remote_struct=remote_struct, permisions=permissions
                    )
                    self.CACHE[remote_struct] = operation_key

                else:
                    operation_key = self.CACHE[remote_struct]

                tuple["algo_key"] = operation_key
