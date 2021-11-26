import datetime
import traceback
from pathlib import Path
from typing import List, Optional

import substra

from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.nodes import AggregationNode, TestDataNode, TrainDataNode
from connectlib.strategies import Strategy


def execute_experiment(
    # TODO: this is getting hard to read, we need to think of an other way to call execute_experiment.
    # This should be done when we add the algo strategy layer.
    # E.g.: create a dependencies pydantic class with three args
    client: substra.Client,
    algo: Algo,
    strategy: Strategy,
    train_data_nodes: List[TrainDataNode],
    test_data_nodes: List[TestDataNode],
    aggregation_node: AggregationNode,
    num_rounds: int,
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
        test_data_nodes (List[TestDataNode]): List of the nodes where testing on data occurs
        aggregation_node (AggregationNode): The aggregation node, where all the shared tasks occur
        num_rounds (int): The number of time your strategy will be executed
        dependencies (Dependency, optional): The list of public dependencies used by your algorithm. Defaults to None.

    Returns:
        [ComputePlan]: The generated compute plan
    """

    # create computation graph
    for _ in range(num_rounds):
        strategy.perform_round(
            algo=algo,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
        )

    strategy.predict(  # TODO rename 'predict' into 'predict_and_score' ? the outputs are metrics here
        algo=algo,
        train_data_nodes=train_data_nodes,
        test_data_nodes=test_data_nodes,
    )

    # Computation graph is created
    # TODO: static checks on the graph

    authorized_ids = list(
        set([aggregation_node.node_id] + [node.node_id for node in train_data_nodes])
    )
    permissions = substra.sdk.schemas.Permissions(
        public=False, authorized_ids=authorized_ids
    )

    # Register all operations in substra
    # Define the algorithms we need and submit them
    composite_traintuples = []
    for train_node in train_data_nodes:
        train_node.register_operations(client, permissions, dependencies=dependencies)
        composite_traintuples += train_node.tuples

    testtuples = []
    for test_node in test_data_nodes:
        test_node.register_operations(client, permissions, dependencies=dependencies)
        testtuples += test_node.tuples

    aggregation_node.register_operations(client, permissions, dependencies=dependencies)

    # Execute the compute plan
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


def execute_experiment_caller() -> Path:
    """Find the path of the file that called the :func:`~connectlib.run_experiment` function via the traceback.

    Returns:
        Optional[Path]: the path of the file that called :func:`~connectlib.run_experiment`
    """
    traces = traceback.extract_stack()
    execute_experiment_trace_indexes = [
        ind
        for ind, trace in enumerate(traces)
        if trace.filename.endswith("connectlib/experiment.py")
        and trace.name == "execute_experiment"
    ]

    callers = [traces[ind - 1].filename for ind in execute_experiment_trace_indexes]

    if len(callers) == 0:
        raise AttributeError(
            "This function is to be used inside the :func:`~run_experiment` function. "
            "But no trace of it has been found."
        )
    if len(callers) > 1:
        nl = "\n\t"
        raise AttributeError(
            """For now, connectlib doesn't support nested call of :func:~`connectlib.run_experiment` function. """
            f"""But you called it {len(callers)} times from the following files {nl}{nl.join(callers)}"""
        )

    return Path(callers[0])
