import substra
import uuid

from typing import List, Optional, Tuple, Dict

from connectlib.nodes.references import (
    LocalStateRef,
    SharedStateRef,
)
from connectlib.nodes import Node
from connectlib.remote.methods import RemoteStruct, DataOperation
from connectlib.remote.register import register_data_node_op

OperationKey = str


class TrainDataNode(Node):
    CACHE: Dict[RemoteStruct, OperationKey] = {}

    def __init__(
        self,
        node_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
        objective_name: str,
    ):
        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys
        self.objective_name = objective_name

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

        train_tuple = {
            "algo_key": operation.remote_struct,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": operation.data_samples,
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": operation.shared_state.key
            if operation.shared_state is not None
            else None,
            "tag": "train",
            "composite_traintuple_id": op_id,
        }

        self.tuples.append(train_tuple)

        return LocalStateRef(op_id), SharedStateRef(op_id)

    def register_operations(
        self, client: substra.Client, permissions: substra.sdk.schemas.Permissions
    ):
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
