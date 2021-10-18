import substra

from typing import List

from connectlib.nodes import Node


class TestDataNode(Node):
    # TODO: improve on comments and docstrings
    def __init__(
        self,
        node_id: str,
        data_manager_key: str,
        test_data_sample_keys: List[str],
        metric_keys: List[str],  # key to the metric, use substra.Client().add_metric()
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys
        self.metric_keys = metric_keys

        super(TestDataNode, self).__init__(node_id)

    def compute(
        self,
        traintuple_id: str,
    ):
        self.tuples.append(
            {
                "metric_keys": self.metric_keys,
                "traintuple_id": traintuple_id,
                "data_manager_key": self.data_manager_key,
                "test_data_sample_keys": self.test_data_sample_keys,
            }
        )

    def register_operations(
        self, client: substra.Client, permissions: substra.sdk.schemas.Permissions
    ):
        pass
