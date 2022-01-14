from typing import Dict
from typing import List

from connectlib.remote.methods import RemoteStruct

OperationKey = str


class Node:
    CACHE: Dict[RemoteStruct, OperationKey] = {}

    def __init__(self, node_id: str):
        self.node_id = node_id

        self.tuples: List[Dict] = []
