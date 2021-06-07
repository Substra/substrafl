import datetime
import substra

from typing import List, Type

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode
from connectlib.strategies import FedAVG
from connectlib.operations.blueprint import Blueprint



class Orchestrator:
    def __init__(self, algo: Blueprint[Type[Algo]], strategy: FedAVG):
        self.algo = algo
        self.strategy = strategy

    def run(
        self,
        client: substra.Client,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        self.strategy.perform_round(
            algo=self.algo,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=None,
            shared_state=None,
        )

        authorized_ids = [aggregation_node.node_id] + [
            node.node_id for node in train_data_nodes
        ]
        permissions = substra.sdk.schemas.Permissions(
            public=False, authorized_ids=authorized_ids
        )

        # Register all operations in substra
        composite_traintuples = []
        for node in train_data_nodes:
            node.register_operations(client, permissions)
            composite_traintuples += node.tuples

        aggregation_node.register_operations(client, permissions)

        compute_plan = client.add_compute_plan(
            {
                "composite_traintuples": composite_traintuples,
                "aggregatetuples": aggregation_node.tuples,
                "tag": str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
            }
        )
