import substra
import uuid

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from connectlib.nodes.pointers import (
    LocalStatePointer,
    SharedStatePointer,
    RemoteTrainPointer,
    AlgoPointer,
)
from connectlib.nodes import Node


@dataclass
class TrainDataNode(Node):
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

    def add(
        self,
        operation_pointer: Union[AlgoPointer, RemoteTrainPointer],
        data_sample_keys: Optional[List[str]] = None,
        local_state_pointer: Optional[LocalStatePointer] = None,
        shared_state_pointer: Optional[SharedStatePointer] = None,
    ) -> Tuple[LocalStatePointer, SharedStatePointer]:
        if data_sample_keys is None:
            data_sample_keys = self.data_sample_keys

        op_id = uuid.uuid4().hex

        train_tuple = {
            "algo_key": operation_pointer.key,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": data_sample_keys,
            "in_head_model_id": local_state_pointer.key
            if local_state_pointer is not None
            else None,
            "in_trunk_model_id": shared_state_pointer.key
            if shared_state_pointer is not None
            else None,
            "tag": "train",
            "composite_traintuple_id": op_id,
        }

        self.tuples.append(train_tuple)

        return LocalStatePointer(op_id), SharedStatePointer(op_id)

    def set_permissions(self, permissions: substra.sdk.schemas.Permissions):
        for tuple in self.tuples:
            tuple["out_trunk_model_permissions"] = permissions
