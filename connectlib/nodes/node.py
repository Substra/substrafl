from typing import Dict
from typing import List

OperationKey = str


class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id

        self.tuples: List[Dict] = []
