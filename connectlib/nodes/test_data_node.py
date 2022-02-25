from typing import List

import substra

from connectlib.dependency import Dependency
from connectlib.nodes import Node


class TestDataNode(Node):
    """A node on which you will test your algorithm.
    A TestDataNode must also be a train data node for now.

    Inherits from :class:`connectlib.nodes.node.Node`

    Args:
        node_id (str): The substra node ID (shared with other nodes if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        test_data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
        metric_keys (List[str]):  Substra metric keys to the metrics, use substra.Client().add_metric()
    """

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

    def update_states(
        self,
        traintuple_id: str,
    ):
        """Creating a test tuple based on the node characteristic.

        Args:
            traintuple_id (str): The substra parant id
        """
        self.tuples.append(
            {
                "metric_keys": self.metric_keys,
                "traintuple_id": traintuple_id,
                "data_manager_key": self.data_manager_key,
                "test_data_sample_keys": self.test_data_sample_keys,
            }
        )

    def register_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        dependencies: Dependency,
    ):
        pass
