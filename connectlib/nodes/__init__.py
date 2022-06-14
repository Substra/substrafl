from connectlib.nodes.node import Node  # isort:skip
from connectlib.nodes.node import OperationKey  # isort:skip

from connectlib.nodes.aggregation_node import AggregationNode
from connectlib.nodes.test_data_node import TestDataNode
from connectlib.nodes.train_data_node import TrainDataNode

# This is needed for auto doc to find that Node module's is organizations.organization, otherwise when
# trying to link Node references from one page to the Node documentation page, it fails.
AggregationNode.__module__ = "organizations.aggregation_node"
Node.__module__ = "organizations.organization"

__all__ = ["Node", "AggregationNode", "TrainDataNode", "TestDataNode", "OperationKey"]
