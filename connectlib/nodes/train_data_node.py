import substra
import uuid

from typing import List, Optional, Tuple, Union, Dict, Type

from connectlib.nodes.references import (
    LocalStateRef,
    SharedStateRef,
)
from connectlib.algorithms import Algo
from connectlib.operations.blueprint import Blueprint
from connectlib.operations import RemoteTrainOp
from connectlib.nodes import Node
from connectlib.nodes.register import register_remote_data_node_op

OperationKey = str


class TrainDataNode(Node):
    CACHE: Dict[Blueprint[Union[Algo, RemoteTrainOp]], OperationKey] = {}

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
        operation: Blueprint[Union[Type[Algo], Type[RemoteTrainOp]]],
        data_sample_keys: Optional[List[str]] = None,
        local_state: Optional[LocalStateRef] = None,
        shared_state: Optional[SharedStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        if not isinstance(operation, Blueprint):
            raise TypeError(
                "operation must be a Blueprint",
                f"Given: {type(operation)}",
                "Have you decorated your Algo or RemoteTrainOp with @blueprint?",
            )
        if not issubclass(operation.cls, Algo) and not issubclass(
            operation.cls, RemoteTrainOp
        ):
            raise TypeError(
                "operation must be a Blueprint of an Algo or RemoteTrainOp",
                f"Given: {operation.cls}",
            )

        if data_sample_keys is None:
            data_sample_keys = self.data_sample_keys

        op_id = uuid.uuid4().hex

        train_tuple = {
            "algo_key": operation,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": data_sample_keys,
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": shared_state.key if shared_state is not None else None,
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

            if isinstance(tuple["algo_key"], Blueprint):
                blueprint: Blueprint[Union[Algo, RemoteTrainOp]] = tuple["algo_key"]

                if blueprint not in self.CACHE:
                    operation_key = register_remote_data_node_op(
                        client, blueprint=blueprint, permisions=permissions
                    )
                    self.CACHE[blueprint] = operation_key

                else:
                    operation_key = self.CACHE[blueprint]

                tuple["algo_key"] = operation_key
