import datetime
import substra

from typing import List, Optional

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.strategies import Strategy


class Orchestrator:
    def __init__(
        self,
        algo: Algo,
        strategy: Strategy,
        num_rounds: int,
        dependencies: Optional[List[str]] = None,
    ):
        """The orchestrator class takes an algo and strategy and runs the
        federated learning experiment.

        Args:
            algo (Algo): Model, with its train and predict functions
            strategy (Strategy): Federated learning strategy to train the model
            num_rounds (int): Number of rounds of the strategy
        """
        self.algo = algo
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.dependencies = dependencies

    def run(
        self,
        client: substra.Client,
        train_data_nodes: List[TrainDataNode],
        test_data_nodes: List[TestDataNode],
        aggregation_node: AggregationNode,
    ):
        """Run the experiment

          Args:
              client (substra.Client): Substra client
              train_data_nodes (List[TrainDataNode]): List of the nodes where training on data occurs
              aggregation_node (AggregationNode): aggregation node if there is one
              test_data_nodes (List[TestDataNode]): List of the TestDataNodes

          Returns:
        "      [type]: [description]
        """
        # TODO: aggregation_node should be optional

        # create computation graph
        for _ in range(self.num_rounds):
            self.strategy.perform_round(
                algo=self.algo,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
            )

        self.strategy.predict(  # TODO rename 'predict' into 'predict_and_score' ? the outputs are metrics here
            algo=self.algo,
            train_data_nodes=train_data_nodes,
            test_data_nodes=test_data_nodes,
        )

        # Computation graph is created
        # TODO: static checks on the graph

        authorized_ids = list(
            set(
                [aggregation_node.node_id] + [node.node_id for node in train_data_nodes]
            )
        )
        permissions = substra.sdk.schemas.Permissions(
            public=False, authorized_ids=authorized_ids
        )

        # Register all operations in substra
        # Define the algorithms we need and submit them
        composite_traintuples = []
        for train_node in train_data_nodes:
            train_node.register_operations(
                client, permissions, dependencies=self.dependencies
            )
            composite_traintuples += train_node.tuples

        testtuples = []
        for test_node in test_data_nodes:
            test_node.register_operations(
                client, permissions, dependencies=self.dependencies
            )
            testtuples += test_node.tuples

        aggregation_node.register_operations(
            client, permissions, dependencies=self.dependencies
        )

        compute_plan = client.add_compute_plan(
            substra.sdk.schemas.ComputePlanSpec(
                composite_traintuples=composite_traintuples,
                aggregatetuples=aggregation_node.tuples,
                testtuples=testtuples,
                tag=str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                clean_models=True,  # set it to False if users need the intermediary models
            ),
            auto_batching=False,
        )

        return compute_plan
