import datetime
import substra

from typing import List

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode
from connectlib.strategies import Strategy


class Orchestrator:
    def __init__(self, algo: Algo, strategy: Strategy, num_rounds: int):
        self.algo = algo
        self.strategy = strategy
        self.num_rounds = num_rounds

    def run(
        self,
        client: substra.Client,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        for _ in range(self.num_rounds):
            self.strategy.perform_round(
                algo=self.algo,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
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

        return compute_plan
