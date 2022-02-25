import copy
import datetime
import logging
from typing import List
from typing import Optional
from typing import Tuple

import substra

from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.nodes import AggregationNode
from connectlib.nodes import TrainDataNode
from connectlib.strategies import Strategy

logger = logging.getLogger(__name__)


def _register_operations(
    client: substra.Client,
    train_data_nodes: List[TrainDataNode],
    aggregation_node: AggregationNode,
    evaluation_strategy: Optional[EvaluationStrategy],
    dependencies: Dependency,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Register the operations in Substra: define the algorithms we need and submit them

    Args:
        client (substra.Client): substra client
        train_data_nodes (List[TrainDataNode]):list of train data nodes
        aggregation_node (AggregationNode): the aggregation node
        evaluation_strategy (Optional[EvaluationStrategy]): the evaluation strategy if there is one
        dependencies (Dependency): dependencies of the train algo

    Returns:
        Tuple[List[dict], List[dict], List[dict]]: composite_traintuples, aggregation_tuples, testtuples specifications
    """
    # `register_operations` methods from the different nodes store the id of the already registered
    # algorithm so we don't add them twice
    operation_cache = dict()

    authorized_ids = list(set([aggregation_node.node_id] + [node.node_id for node in train_data_nodes]))
    permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=authorized_ids)

    composite_traintuples = []
    for train_node in train_data_nodes:
        operation_cache = train_node.register_operations(
            client, permissions, cache=operation_cache, dependencies=dependencies
        )
        composite_traintuples += train_node.tuples

    testtuples = []
    if evaluation_strategy is not None:
        for test_node in evaluation_strategy.test_data_nodes:
            # The test nodes do not have any operation to register: no algo on the testtuple
            testtuples += test_node.tuples

    # The aggregation operation is defined in the strategy, its dependencies are
    # the strategy dependencies
    # We still need to pass the information of the editable mode.
    operation_cache = aggregation_node.register_operations(
        client, permissions, cache=operation_cache, dependencies=Dependency(editable_mode=dependencies.editable_mode)
    )
    aggregation_tuples = aggregation_node.tuples

    return composite_traintuples, aggregation_tuples, testtuples


def execute_experiment(
    client: substra.Client,
    algo: Algo,
    strategy: Strategy,
    train_data_nodes: List[TrainDataNode],
    aggregation_node: AggregationNode,
    num_rounds: int,
    evaluation_strategy: Optional[EvaluationStrategy] = None,
    dependencies: Optional[Dependency] = None,
) -> substra.sdk.models.ComputePlan:
    """Run a complete experiment. This will train (on the `train_data_nodes`) and test (on the `test_data_nodes`)
    your `algo` with the specified `strategy` `n_rounds` times and return the compute plan object from the connect
    platform.

    In connectlib, operations are linked to each other statically before being submitted to substra.

    The execution of :
        * the `self.perform_round` method from the passed strategy **num_rounds** times
        * the `self.predict` methods from the passed strategy
    generate the static graph of operations.

    Each element necessary for those operations (CompositeTrainTuples, TestTuples and Algorithms)
    is registered to the connect platform thanks to the specified client.

    Finally, the compute plan is sent and executed.

    Args:
        client (substra.Client): A substra client to interact with the connect platform
        algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
        strategy (Strategy): The strategy by which your algorithm will be executed
        train_data_nodes (List[TrainDataNode]): List of the nodes where training on data occurs
        evaluation_strategy (EvaluationStrategy, optional): If None performance will not be measured at all.
            Otherwise measuring of performance will follow the EvaluationStrategy. Defaults to None.
        aggregation_node (AggregationNode): The aggregation node, where all the shared tasks occur
        num_rounds (int): The number of time your strategy will be executed
        dependencies (Dependency, optional): Dependencies of the algorithm. It must be defined from
            the connectlib Dependency class. Defaults None.

    Returns:
        ComputePlan: The generated compute plan
    """
    if dependencies is None:
        dependencies = Dependency()
    train_data_nodes = copy.deepcopy(train_data_nodes)
    aggregation_node = copy.deepcopy(aggregation_node)
    strategy = copy.deepcopy(strategy)
    evaluation_strategy = copy.deepcopy(evaluation_strategy)

    train_node_ids = [train_data_node.node_id for train_data_node in train_data_nodes]

    if len(train_node_ids) != len(set(train_node_ids)):
        raise ValueError("Training multiple algorithms on the same node is not supported right now.")

    if evaluation_strategy is not None:
        if evaluation_strategy.num_rounds is None:
            evaluation_strategy.num_rounds = num_rounds
        elif evaluation_strategy.num_rounds != num_rounds:
            raise ValueError(
                "num_rounds set in evaluation_strategy does not match num_rounds set in the experiment: "
                f"{evaluation_strategy.num_rounds} is not {num_rounds}"
            )
    # Reseting evaluation strategy
    evaluation_strategy.restart_rounds()

    logger.info("Building the compute plan.")

    # create computation graph
    for _ in range(num_rounds):
        strategy.perform_round(
            algo=algo,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
        )

        if evaluation_strategy is not None and next(evaluation_strategy):
            strategy.predict(
                algo=algo,
                train_data_nodes=train_data_nodes,
                test_data_nodes=evaluation_strategy.test_data_nodes,
            )

    # Computation graph is created
    logger.info("Submitting the algorithm to Connect.")
    composite_traintuples, aggregation_tuples, testtuples = _register_operations(
        client=client,
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        evaluation_strategy=evaluation_strategy,
        dependencies=dependencies,
    )

    # Execute the compute plan
    logger.info("Submitting the compute plan to Connect.")
    compute_plan = client.add_compute_plan(
        substra.sdk.schemas.ComputePlanSpec(
            composite_traintuples=composite_traintuples,
            aggregatetuples=aggregation_tuples,
            testtuples=testtuples,
            tag=str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
            clean_models=True,  # set it to False if users need the intermediary models
        ),
        auto_batching=False,
    )

    logger.info(("The compute plan has been submitted to Connect, its key is {0}.").format(compute_plan.key))

    return compute_plan
