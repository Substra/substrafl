from substrafl.nodes.schemas import OperationKey  # isort:skip

from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.aggregation_node import SimuAggregationNode
from substrafl.nodes.protocol import AggregationNodeProtocol
from substrafl.nodes.protocol import TestDataNodeProtocol
from substrafl.nodes.protocol import TrainDataNodeProtocol
from substrafl.nodes.test_data_node import SimuTestDataNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import SimuTrainDataNode
from substrafl.nodes.train_data_node import TrainDataNode

# This is needed for auto doc to find that Node module's is organizations.organization, otherwise when
# trying to link Node references from one page to the Node documentation page, it fails.
AggregationNode.__module__ = "organizations.aggregation_node"

__all__ = [
    "TestDataNodeProtocol",
    "TrainDataNodeProtocol",
    "AggregationNodeProtocol",
    "AggregationNode",
    "TrainDataNode",
    "TestDataNode",
    "SimuAggregationNode",
    "SimuTrainDataNode",
    "SimuTestDataNode",
    "OperationKey",
]
