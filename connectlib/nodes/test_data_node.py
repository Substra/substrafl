import substra
import uuid

from typing import List, Optional, Tuple

from connectlib.nodes import Node


class TestDataNode(Node):
    # TODO: improve on comments and docstrings
    def __init__(
        self,
        node_id: str,
        data_manager_key: str,
        test_data_sample_keys: List[str],
        objective_key: str,  # key to the metric, use substra.Client().add_objective()
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys
        self.objective_key = objective_key

        super(TestDataNode, self).__init__(node_id)

    def compute(
        self,
        traintuple_id: str,
    ):
        self.tuples.append(
            {
                "objective_key": self.objective_key,
                "traintuple_id": traintuple_id,
                "data_manager_key": self.data_manager_key,
                "test_data_sample_keys": self.test_data_sample_keys,
            }
        )

    def register_operations(
        self, client: substra.Client, permissions: substra.sdk.schemas.Permissions
    ):
        pass
