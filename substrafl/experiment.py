import copy
import datetime
import json
import logging
import uuid
from pathlib import Path
from platform import python_version
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import substra
import substratools

import substrafl
from substrafl import exceptions
from substrafl.algorithms.algo import Algo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import KeyMetadataError
from substrafl.exceptions import LenMetadataError
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.node import OperationKey
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


def _register_operations(
    client: substra.Client,
    train_data_nodes: List[TrainDataNode],
    aggregation_node: Optional[AggregationNode],
    evaluation_strategy: Optional[EvaluationStrategy],
    dependencies: Dependency,
) -> Tuple[List[dict], List[dict], List[dict], Dict[RemoteStruct, OperationKey]]:
    """Register the operations in Substra: define the algorithms we need and submit them

    Args:
        client (substra.Client): substra client
        train_data_nodes (typing.List[TrainDataNode]): list of train data nodes
        aggregation_node (typing.Optional[AggregationNode]): the aggregation node for
            centralized strategies
        evaluation_strategy (typing.Optional[EvaluationStrategy]): the evaluation strategy
            if there is one dependencies
        (Dependency): dependencies of the train algo

    Returns:
        typing.Tuple[typing.List[dict], typing.List[dict], typing.List[dict], typing.Dict[RemoteStruct, OperationKey]]:
        composite_traintuples, aggregation_tuples, testtuples specifications, operation_cache
    """
    # `register_operations` methods from the different organizations store the id of the already registered
    # algorithm so we don't add them twice
    operation_cache = dict()
    predict_algo_cache = dict()

    train_data_organizations_id = {train_data_node.organization_id for train_data_node in train_data_nodes}
    aggregation_organization_id = {aggregation_node.organization_id} if aggregation_node is not None else set()

    authorized_ids = list(train_data_organizations_id | aggregation_organization_id)
    permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=authorized_ids)

    composite_traintuples = []
    for train_data_node in train_data_nodes:
        operation_cache = train_data_node.register_operations(
            client,
            permissions,
            cache=operation_cache,
            dependencies=dependencies,
        )

        composite_traintuples += train_data_node.tuples

    predicttuples = []
    testtuples = []
    if evaluation_strategy is not None:
        for test_data_node in evaluation_strategy.test_data_nodes:
            predict_algo_cache = test_data_node.register_predict_operations(
                client,
                permissions,
                traintuples=composite_traintuples,
                cache=predict_algo_cache,
                dependencies=dependencies,
            )

            predicttuples += test_data_node.predicttuples
            testtuples += test_data_node.testtuples

    # The aggregation operation is defined in the strategy, its dependencies are
    # the strategy dependencies
    # We still need to pass the information of the editable mode.
    aggregation_tuples = []
    if aggregation_node is not None:
        operation_cache = aggregation_node.register_operations(
            client,
            permissions,
            cache=operation_cache,
            dependencies=Dependency(editable_mode=dependencies.editable_mode),
        )

        aggregation_tuples = aggregation_node.tuples

    return composite_traintuples, aggregation_tuples, predicttuples, testtuples, operation_cache


def _save_experiment_summary(
    experiment_folder: Path,
    compute_plan_key: str,
    strategy: Strategy,
    num_rounds: int,
    algo: Algo,
    operation_cache: Dict[RemoteStruct, OperationKey],
    train_data_nodes: TrainDataNode,
    aggregation_node: Optional[AggregationNode],
    evaluation_strategy: EvaluationStrategy,
    timestamp: str,
    additional_metadata: Optional[Dict],
):
    """Saves the experiment summary in `experiment_folder`, with the name format `{timestamp}_{compute_plan.key}.json`

    Args:
        experiment_folder (typing.Union[str, pathlib.Path]): path to the folder where the experiment summary is saved.
        compute_plan_key (str): compute plan key
        strategy (substrafl.strategies.Strategy): strategy
        num_rounds (int): num_rounds
        algo (substrafl.algorithms.Algo): algo
        operation_cache (typing.Dict[RemoteStruct, OperationKey]): operation_cache
        train_data_nodes (TrainDataNode): train_data_nodes
        aggregation_node (typing.Optional[AggregationNode]): aggregation_node
        evaluation_strategy (EvaluationStrategy): evaluation_strategy
        timestamp (str): timestamp with "%Y_%m_%d_%H_%M_%S" format
        additional_metadata (dict, Optional): Optional dictionary of metadata to be shown on the Substra WebApp.
    """
    # create the experiment folder if it doesn't exist
    experiment_folder = Path(experiment_folder)
    experiment_folder.mkdir(exist_ok=True)
    experiment_summary = dict()

    # add attributes of interest and summaries of the classes to the experiment summary
    experiment_summary["compute_plan_key"] = compute_plan_key
    experiment_summary["strategy"] = type(strategy).__name__
    experiment_summary["num_rounds"] = num_rounds
    experiment_summary["algo"] = algo.summary()
    experiment_summary["algo_keys"] = {
        operation_cache[remote_struct]: remote_struct.summary() for remote_struct in operation_cache
    }
    experiment_summary["train_data_nodes"] = [train_data_node.summary() for train_data_node in train_data_nodes]
    experiment_summary["test_data_nodes"] = []
    if evaluation_strategy is not None:
        experiment_summary["test_data_nodes"] = [
            test_data_node.summary() for test_data_node in evaluation_strategy.test_data_nodes
        ]

    experiment_summary["aggregation_node"] = aggregation_node.summary() if aggregation_node is not None else None
    if additional_metadata is not None:
        experiment_summary["additional_metadata"] = additional_metadata

    # Save the experiment summary
    summary_file = experiment_folder / f"{timestamp}_{compute_plan_key}.json"
    summary_file.write_text(json.dumps(experiment_summary, indent=4))
    logger.info(("Experiment summary saved to {0}").format(summary_file))


def _check_evaluation_strategy(
    evaluation_strategy: EvaluationStrategy,
    num_rounds: int,
):
    if evaluation_strategy.num_rounds is None:
        evaluation_strategy.num_rounds = num_rounds
    elif evaluation_strategy.num_rounds != num_rounds:
        raise ValueError(
            "num_rounds set in evaluation_strategy does not match num_rounds set in the experiment: "
            f"{evaluation_strategy.num_rounds} is not {num_rounds}"
        )


def _check_additional_metadata(additional_metadata: Dict):
    unauthorized_keys = set(
        ("substrafl_version", "substra_version", "substratools_version", "python_version", "num_rounds")
    )
    invalid_keys = set(additional_metadata.keys()).intersection(unauthorized_keys)

    if len(invalid_keys) > 0:
        raise KeyMetadataError(
            f"None of: `{'`, `'.join(unauthorized_keys)}` can be used as"
            f" metadata key but `{' '.join(invalid_keys)}` were/was found"
        )

    for metadata_value in additional_metadata.values():
        if len(str(metadata_value)) > 100:
            raise LenMetadataError(
                "The maximum length of a value in the additional_metadata dictionary is 100 characters."
            )


def _get_packages_versions() -> dict:
    """Returns a dict containing substrafl, substra and substratools versions

    Returns:
        dict: substrafl, substra and substratools versions
    """

    return {
        "substrafl_version": substrafl.__version__,
        "substra_version": substra.__version__,
        "substratools_version": substratools.__version__,
        "python_version": python_version(),
    }


def execute_experiment(
    client: substra.Client,
    algo: Algo,
    strategy: Strategy,
    train_data_nodes: List[TrainDataNode],
    num_rounds: int,
    experiment_folder: Union[str, Path],
    aggregation_node: Optional[AggregationNode] = None,
    evaluation_strategy: Optional[EvaluationStrategy] = None,
    dependencies: Optional[Dependency] = None,
    clean_models: bool = True,
    name: Optional[str] = None,
    additional_metadata: Optional[Dict] = None,
) -> substra.sdk.models.ComputePlan:
    """Run a complete experiment. This will train (on the `train_data_nodes`) and test (on the
    `test_data_nodes`) your `algo` with the specified `strategy` `n_rounds` times and return the
    compute plan object from the Substra platform.

    In substrafl, operations are linked to each other statically before being submitted to substra.

    The execution of:

        - the `self.perform_round` method from the passed strategy **num_rounds** times
        - the `self.predict` methods from the passed strategy

    generate the static graph of operations.

    Each element necessary for those operations (CompositeTrainTuples, TestTuples and Algorithms)
    is registered to the Substra platform thanks to the specified client.

    Finally, the compute plan is sent and executed.

    The experiment summary is saved in `experiment_folder`, with the name format `{timestamp}_{compute_plan.key}.json`

    Args:
        client (substra.Client): A substra client to interact with the Substra platform
        algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
        strategy (Strategy): The strategy by which your algorithm will be executed
        train_data_nodes (typing.List[TrainDataNode]): List of the nodes where training on data
            occurs evaluation_strategy (EvaluationStrategy, Optional): If None performance will not be measured at all.
            Otherwise measuring of performance will follow the EvaluationStrategy. Defaults to None.
        aggregation_node (typing.Optional[AggregationNode]): For centralized strategy, the aggregation
            node, where all the shared tasks occurs else None.
        num_rounds (int): The number of time your strategy will be executed
        dependencies (Dependency, Optional): Dependencies of the algorithm. It must be defined from
            the substrafl Dependency class. Defaults None.
        experiment_folder (typing.Union[str, pathlib.Path]): path to the folder where the experiment summary is saved.
        clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
            if you want to download or re-use intermediary models. This causes the disk space to fill
            quickly so should be set to True unless needed. Defaults to True.
        name (str, Optional): Optional name chosen by the user to identify the compute plan. If None,
            the compute plan name is set to the timestamp.
        additional_metadata(dict, Optional): Optional dictionary of metadata to be passed to the Substra WebApp.

    Returns:
        ComputePlan: The generated compute plan
    """
    if dependencies is None:
        dependencies = Dependency()

    if strategy.name not in algo.strategies:
        raise exceptions.IncompatibleAlgoStrategyError(
            f"The algo {algo.__class__.__name__} is not compatible with the strategy {strategy.__class__.__name__},"
            f"named {strategy.name}. Check the algo strategies property: algo.strategies to see the list of compatible"
            "strategies."
        )

    train_data_nodes = copy.deepcopy(train_data_nodes)
    aggregation_node = copy.deepcopy(aggregation_node)
    strategy = copy.deepcopy(strategy)
    evaluation_strategy = copy.deepcopy(evaluation_strategy)

    train_organization_ids = [train_data_node.organization_id for train_data_node in train_data_nodes]

    if len(train_organization_ids) != len(set(train_organization_ids)):
        raise ValueError("Training multiple algorithms on the same organization is not supported right now.")

    if evaluation_strategy is not None:
        _check_evaluation_strategy(evaluation_strategy, num_rounds)
        # Reset the evaluation strategy
        evaluation_strategy.restart_rounds()

    cp_metadata = dict()
    if additional_metadata is not None:
        _check_additional_metadata(additional_metadata)
        cp_metadata.update(additional_metadata)

    # Adding substrafl, substratools and substra versions to the cp metadata
    cp_metadata.update(_get_packages_versions())

    # Adding rounds metadata to cp_meta_data
    cp_metadata["num_rounds"] = num_rounds

    logger.info("Building the compute plan.")

    # create computation graph
    for round_idx in range(1, num_rounds + 1):
        strategy.perform_round(
            algo=algo,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        if evaluation_strategy is not None and next(evaluation_strategy):
            strategy.predict(
                algo=algo,
                train_data_nodes=train_data_nodes,
                test_data_nodes=evaluation_strategy.test_data_nodes,
                round_idx=round_idx,
            )

    # Computation graph is created
    logger.info("Registering the algorithm to Substra.")
    composite_traintuples, aggregation_tuples, predicttuples, testtuples, operation_cache = _register_operations(
        client=client,
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        evaluation_strategy=evaluation_strategy,
        dependencies=dependencies,
    )

    # Execute the compute plan
    logger.info("Registering the compute plan to Substra.")
    timestamp = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # Generate the compute plan key.
    compute_plan_key = str(uuid.uuid4())

    # save the experiment summary in experiment_folder
    _save_experiment_summary(
        experiment_folder=experiment_folder,
        compute_plan_key=compute_plan_key,
        strategy=strategy,
        num_rounds=num_rounds,
        algo=algo,
        operation_cache=operation_cache,
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        evaluation_strategy=evaluation_strategy,
        timestamp=timestamp,
        additional_metadata=additional_metadata,
    )
    compute_plan = client.add_compute_plan(
        substra.sdk.schemas.ComputePlanSpec(
            key=compute_plan_key,
            composite_traintuples=composite_traintuples,
            aggregatetuples=aggregation_tuples,
            predicttuples=predicttuples,
            testtuples=testtuples,
            name=name or timestamp,
            metadata=cp_metadata,
        ),
    )
    logger.info(("The compute plan has been registered to Substra, its key is {0}.").format(compute_plan.key))
    return compute_plan
