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
from substrafl.compute_plan_builder import ComputePlanBuilder
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import KeyMetadataError
from substrafl.exceptions import LenMetadataError
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.node import OperationKey
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote.remote_struct import RemoteStruct

logger = logging.getLogger(__name__)


def _register_operations(
    client: substra.Client,
    train_data_nodes: List[TrainDataNode],
    aggregation_node: Optional[AggregationNode],
    evaluation_strategy: Optional[EvaluationStrategy],
    dependencies: Dependency,
) -> Tuple[List[dict], Dict[RemoteStruct, OperationKey]]:
    """Register the operations in Substra: define the functions we need and submit them

    Args:
        client (substra.Client): substra client
        train_data_nodes (typing.List[TrainDataNode]): list of train data nodes
        aggregation_node (typing.Optional[AggregationNode]): the aggregation node for
            centralized strategies
        evaluation_strategy (typing.Optional[EvaluationStrategy]): the evaluation strategy
            if there is one dependencies
        (Dependency): dependencies of the experiment

    Returns:
        typing.Tuple[typing.List[dict], typing.Dict[RemoteStruct, OperationKey]]:
        tasks, operation_cache
    """
    # `register_operations` methods from the different organizations store the id of the already registered
    # functions so we don't add them twice
    operation_cache = dict()
    predict_function_cache = dict()
    test_function_cache = dict()
    tasks = list()

    train_data_organizations_id = {train_data_node.organization_id for train_data_node in train_data_nodes}
    aggregation_organization_id = {aggregation_node.organization_id} if aggregation_node is not None else set()
    test_data_organizations_ids = (
        evaluation_strategy.test_data_nodes_org_ids if evaluation_strategy is not None else set()
    )

    authorized_ids = list(train_data_organizations_id | aggregation_organization_id | test_data_organizations_ids)
    permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=authorized_ids)

    for train_data_node in train_data_nodes:
        operation_cache = train_data_node.register_operations(
            client=client,
            permissions=permissions,
            cache=operation_cache,
            dependencies=dependencies,
        )

        if train_data_node.init_task is not None:
            tasks += [train_data_node.init_task]

        tasks += train_data_node.tasks

    if evaluation_strategy is not None:
        for test_data_node in evaluation_strategy.test_data_nodes:
            predict_function_cache = test_data_node.register_predict_operations(
                client=client,
                permissions=permissions,
                cache=predict_function_cache,
                dependencies=dependencies,
            )
            test_function_cache = test_data_node.register_test_operations(
                client=client,
                permissions=permissions,
                cache=test_function_cache,
                dependencies=dependencies,
            )

            tasks += test_data_node.predicttasks
            tasks += test_data_node.testtasks

    # The aggregation operation is defined in the strategy, its dependencies are
    # the strategy dependencies
    if aggregation_node is not None:
        operation_cache = aggregation_node.register_operations(
            client=client,
            permissions=permissions,
            cache=operation_cache,
            dependencies=dependencies,
        )

        tasks += aggregation_node.tasks

    return tasks, operation_cache


def _save_experiment_summary(
    experiment_folder: Path,
    compute_plan_key: str,
    strategy: ComputePlanBuilder,
    num_rounds: int,
    operation_cache: Dict[RemoteStruct, OperationKey],
    train_data_nodes: List[TrainDataNode],
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
        operation_cache (typing.Dict[RemoteStruct, OperationKey]): operation_cache
        train_data_nodes (typing.List[TrainDataNode]): train_data_nodes
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
    if num_rounds is not None:
        experiment_summary["num_rounds"] = num_rounds
    experiment_summary["function_keys"] = {
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
    unauthorized_keys = {"substrafl_version", "substra_version", "substratools_version", "python_version"}
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
    *,
    client: substra.Client,
    strategy: ComputePlanBuilder,
    train_data_nodes: List[TrainDataNode],
    experiment_folder: Union[str, Path],
    num_rounds: Optional[int] = None,
    aggregation_node: Optional[AggregationNode] = None,
    evaluation_strategy: Optional[EvaluationStrategy] = None,
    dependencies: Optional[Dependency] = None,
    clean_models: bool = True,
    name: Optional[str] = None,
    additional_metadata: Optional[Dict] = None,
    task_submission_batch_size: int = 500,
) -> substra.sdk.models.ComputePlan:
    """Run a complete experiment. This will train (on the `train_data_nodes`) and test (on the
    `test_data_nodes`) the specified `strategy` `n_rounds` times and return the
    compute plan object from the Substra platform.

    In SubstraFL, operations are linked to each other statically before being submitted to Substra.

    The execution of:

        - the `self.perform_round` method from the passed strategy **num_rounds** times
        - the `self.predict` methods from the passed strategy

    generate the static graph of operations.

    Each element necessary for those operations (Tasks and Functions)
    is registered to the Substra platform thanks to the specified client.

    Finally, the compute plan is sent and executed.

    The experiment summary is saved in `experiment_folder`, with the name format `{timestamp}_{compute_plan.key}.json`

    Args:
        client (substra.Client): A substra client to interact with the Substra platform
        strategy (Strategy): The strategy that will be executed
        train_data_nodes (typing.List[TrainDataNode]): List of the nodes where training on data
            occurs evaluation_strategy (EvaluationStrategy, Optional): If None performance will not be measured at all.
            Otherwise measuring of performance will follow the EvaluationStrategy. Defaults to None.
        aggregation_node (typing.Optional[AggregationNode]): For centralized strategy, the aggregation
            node, where all the shared tasks occurs else None.
        evaluation_strategy: Optional[EvaluationStrategy]
        num_rounds (int): The number of time your strategy will be executed
        dependencies (Dependency, Optional): Dependencies of the experiment. It must be defined from
            the SubstraFL Dependency class. Defaults None.
        experiment_folder (typing.Union[str, pathlib.Path]): path to the folder where the experiment summary is saved.
        clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
            if you want to download or re-use intermediary models. This causes the disk space to fill
            quickly so should be set to True unless needed. Defaults to True.
        name (str, Optional): Optional name chosen by the user to identify the compute plan. If None,
            the compute plan name is set to the timestamp.
        additional_metadata(dict, typing.Optional): Optional dictionary of metadata to be passed to the Substra WebApp.
        task_submission_batch_size(int): The compute plan tasks are submitted by batch. The higher the batch size,
            the faster the submission, a batch size that is too high makes the submission fail.
            Rule of thumb: batch_size = math.floor(120000 / number_of_samples_per_task)

    Returns:
        ComputePlan: The generated compute plan
    """
    if dependencies is None:
        dependencies = Dependency()

    train_data_nodes = copy.deepcopy(train_data_nodes)
    aggregation_node = copy.deepcopy(aggregation_node)
    strategy = copy.deepcopy(strategy)
    evaluation_strategy = copy.deepcopy(evaluation_strategy)

    train_organization_ids = [train_data_node.organization_id for train_data_node in train_data_nodes]

    if len(train_organization_ids) != len(set(train_organization_ids)):
        raise ValueError("Training multiple functions on the same organization is not supported right now.")

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

    logger.info("Building the compute plan.")

    strategy.build_compute_plan(
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        evaluation_strategy=evaluation_strategy,
        num_rounds=num_rounds,
        clean_models=clean_models,
    )

    # Computation graph is created
    logger.info("Registering the functions to Substra.")
    tasks, operation_cache = _register_operations(
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
            tasks=tasks,
            name=name or timestamp,
            metadata=cp_metadata,
        ),
        auto_batching=True,
        batch_size=task_submission_batch_size,
    )
    logger.info(("The compute plan has been registered to Substra, its key is {0}.").format(compute_plan.key))
    return compute_plan
