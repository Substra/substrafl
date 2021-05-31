from dataclasses import dataclass
from typing import List


@dataclass
class NodeSpec:
    node_id: str
    data_manager_key: str
    data_sample_keys: List[str]
    objective_name: str

    def perform_update(self):
        pass
