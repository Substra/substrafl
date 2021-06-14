import datetime
import substra

from typing import List

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
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
        test_data_nodes: List[TestDataNode],
    ):
        for _ in range(self.num_rounds):
            self.strategy.perform_round(
                algo=self.algo,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
            )
        self.strategy.predict(
            algo=self.algo, train_data_nodes=train_data_nodes, test_data_nodes=test_data_nodes
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

        testtuples = []
        for node in test_data_nodes:
            node.register_operations(client, permissions)
            testtuples += node.tuples

        aggregation_node.register_operations(client, permissions)

        compute_plan = client.add_compute_plan(
            {
                "composite_traintuples": composite_traintuples,
                "aggregatetuples": aggregation_node.tuples,
                "testtuples": testtuples,
                "tag": str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
            }
        )

        return compute_plan
