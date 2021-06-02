from dataclasses import dataclass
from typing import List, Optional, Tuple

from connectlib.nodes.pointers.local_state import LocalStatePointer
from connectlib.nodes.pointers.shared_state import SharedStatePointer
from connectlib.operations import RemoteTrainDataOp


@dataclass
class Node:
    node_id: str
    data_manager_key: str
    data_sample_keys: List[str]
    objective_name: str

    def submit(self,
               operation: RemoteTrainDataOp,
               data_sample_keys: Optional[List[str]] = None,
               local_state_pointer: Optional[LocalStatePointer] = None,
               shared_state_pointer: Optional[SharedStatePointer] = None) \
            -> Tuple[SharedStatePointer, LocalStatePointer]:
        pass
