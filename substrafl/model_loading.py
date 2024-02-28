import json
import logging
import sys
import tarfile
import tempfile
from pathlib import Path
from platform import python_version
from typing import Any
from typing import Optional

import substra
import substratools
from substra.sdk.models import ComputeTaskStatus

import substrafl
from substrafl import exceptions
from substrafl.constants import SUBSTRAFL_FOLDER
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.schemas import TaskType

logger = logging.getLogger(__name__)

REQUIRED_LIBS = [substrafl, substra, substratools]
REQUIRED_KEYS = set([lib.__name__ + "_version" for lib in REQUIRED_LIBS] + ["python_version"])
METADATA_FILE = "metadata.json"
FUNCTION_DICT_KEY = "function_file"
MODEL_DICT_KEY = "model_file"


def _check_environment_compatibility(metadata: dict):
    """Checks whether or not the local environment is compatible with the execution environment of the
    compute plan.

    Args:
        metadata (dict): Compute plan's metadata to download the model from.
    """

    if REQUIRED_KEYS.intersection(set(metadata.keys())) != REQUIRED_KEYS:
        raise NotImplementedError(
            "The given metadata doesn't seems to belong to a substrafl compute plan. "
            f"Please Ensure the {REQUIRED_KEYS} are all in the given metadata."
        )

    environment_warnings = ""

    if python_version() != metadata["python_version"]:
        environment_warnings = f"\n\tfrom a python {python_version()} environment."

    for lib in REQUIRED_LIBS:
        metadata_key = lib.__name__ + "_version"

        if lib.__version__ != metadata[metadata_key]:
            environment_warnings += f"\n\twith {lib.__name__} {lib.__version__}"

    if environment_warnings != "":
        logger.warning(
            "This feature is not retro compatible yet.\n"
            "Running it in the current environment could lead to unexpected behaviors and errors.\n"
            "It is safer to download this model:" + environment_warnings
        )


def _validate_folder_content(folder: Path) -> dict:
    """Checks if the input folder is containing the necessary files to load a task output.

    Args:
        folder (Path): Folder to validate the input from.

    Returns:
        dict: execution environment metadata of the model stored in the given folder
    """
    metadata_path = folder / METADATA_FILE
    model_file = None

    missing = []
    end_of_msg = (
        "This folder should be the result of the `substrafl._download_task_output_files` function "
        f"and contain the {METADATA_FILE}, model and algo files."
        "being the local state of the model to load within memory."
    )

    if not metadata_path.exists():
        missing.append(METADATA_FILE)
    else:
        metadata = json.loads(metadata_path.read_text())

        # if metadata file exist we check that the MODEL_DICT_KEY key is provided
        if MODEL_DICT_KEY not in metadata:
            raise exceptions.LoadMetadataError(
                f"The {METADATA_FILE} file from the specified folder should contain a `{MODEL_DICT_KEY}` key"
                "pointing the downloaded local state of the model to load within memory."
            )

        # if metadata file exist we check that the FUNCTION_DICT_KEY key is provided
        elif FUNCTION_DICT_KEY not in metadata:
            raise exceptions.LoadMetadataError(
                f"The {METADATA_FILE} file from the specified folder should contain an `{FUNCTION_DICT_KEY}` key"
                "pointing the downloaded algo file to load within memory."
            )

        model_file = metadata[MODEL_DICT_KEY]
        algo_path = metadata[FUNCTION_DICT_KEY]

        # And that the pointed files exists
        if not ((folder / model_file).exists()):
            missing.append(model_file)
            end_of_msg += f", `{model_file}`"

        if not ((folder / algo_path).exists()):
            missing.append(algo_path)
            end_of_msg += f", `{algo_path}`"

    if len(missing) > 0:
        raise exceptions.LoadFileNotFoundError(
            ", ".join(missing) + f" not found within the provided input folder `{folder}`.\n" + end_of_msg
        )

    return metadata


def _get_task_from_round(
    client: substra.Client, compute_plan_key: str, round_idx: int, tag: str
) -> substra.models.Task:
    """Return the task:

        - hosted on the given client organization
        - belonging to the given compute plan
        - of the given round_idx

    Args:
        client (substra.Client): Substra client where to fetch the task from.
        compute_plan_key (str): Compute plan key to fetch the task from.
        round_idx (int): Round of the strategy to fetch the task from.
        tag (str): Tag of the task to get.

    Returns:
        substra.models.Task: The task matching the given requirements.
    """
    org_id = client.organization_info().organization_id

    filters = {
        "compute_plan_key": [compute_plan_key],
        "worker": [org_id],
        "metadata": [{"key": "round_idx", "type": "is", "value": str(round_idx)}],
    }
    local_tasks = client.list_task(filters=filters)
    local_tagged_tasks = [t for t in local_tasks if t.tag == tag]

    if len(local_tagged_tasks) == 0:
        raise exceptions.TaskNotFoundError(
            f"The given compute plan `{compute_plan_key}` has no {tag} task of round {round_idx} "
            f"hosted on the organization {org_id}"
        )

    elif len(local_tagged_tasks) > 1:
        raise exceptions.MultipleTaskError(
            f"The given compute plan has {len(local_tagged_tasks)} local {tag} tasks of round {round_idx}. "
        )
    local_task = local_tagged_tasks[0]
    return local_task


def _get_task_from_rank(
    client: substra.Client, compute_plan_key: str, tag: str, rank_idx: Optional[int] = None
) -> substra.models.Task:
    """Return the task:

        - hosted on the given client organization
        - belonging to the given compute plan
        - of the given rank_idx

    Args:
        client (substra.Client): Substra client where to fetch the task from.
        compute_plan_key (str): Compute plan key to fetch the task from.
        rank_idx (int): Rank of the strategy to fetch the task from. If set to None,
            the last ending task will be returned. (Default to None)
        tag (str): Tag of the task to get.

    Returns:
        substra.models.Task: The task matching the given requirements.
    """
    org_id = client.organization_info().organization_id

    if rank_idx is not None:
        filters = {"compute_plan_key": [compute_plan_key], "worker": [org_id], "rank": [rank_idx]}
        local_tasks = client.list_task(filters=filters)
        local_tagged_tasks = [t for t in local_tasks if t.tag == tag]

    else:
        filters = {"compute_plan_key": [compute_plan_key], "worker": [org_id]}
        local_tasks = client.list_task(filters=filters, order_by="end_date")
        local_tagged_tasks = [t for t in local_tasks if t.tag == tag]

    if len(local_tagged_tasks) == 0:
        raise exceptions.TaskNotFoundError(
            f"The given compute plan `{compute_plan_key}` has no {tag} task of rank {rank_idx} "
            f"hosted on the organization {org_id}"
        )

    elif len(local_tagged_tasks) > 1 and rank_idx is not None:
        raise exceptions.MultipleTaskError(
            f"The given compute plan has {len(local_tagged_tasks)} {tag} tasks of rank {rank_idx}. "
        )

    local_task = local_tagged_tasks[0]

    return local_task


def _load_instance(gz_path: Path, extraction_folder: Path, remote: bool) -> Any:
    """Load into memory a serialized (and compressed (.tar.gz)) SubstraFL Remote Struct instance within the
    given path.
    This kind of file is usually the result of the ``substra.Client.download_function`` function applied to
    a task being part of a SubstraFL experiment.

    Args:
        gz_path (Path): A file being the tar.gz compression of a SubstraFL RemoteStruct instance
        extraction_folder (Path): Where to unpack the folder.
        remote (bool): Whether to return the instance or the remote instance from the loaded remote_struct.

    Returns:
        Any: The loaded SubstraFL object into memory.
    """
    with tarfile.open(gz_path, "r:gz") as tar:
        tar.extractall(path=extraction_folder)

    sys.path.append(str(extraction_folder))  # for local dependencies

    remote_struct = RemoteStruct.load(src=extraction_folder / SUBSTRAFL_FOLDER)

    if not remote:
        instance = remote_struct.get_instance()
    else:
        instance = remote_struct.get_remote_instance()

    return instance


def _load_from_files(input_folder: Path, remote: bool = False) -> Any:
    """Loads an instance from a specified folder. This folder should contains:

        - function.tar.gz
        - metadata.json
        - the file specified in metadata.model_file

    Args:
        input_folder (Path): Path to folder containing the required files.
        remote (bool): Wether the the instance to load is to load using a local method or a remote method.

    Raises:
        exceptions.LoadMetadataError: The metadata file must contains the model_file key
        exceptions.LoadFileNotFoundError: At least one of the required file to load the instance is not found
        exceptions.LoadLocalDependencyError: One of the dependency used by the instance is not installed within the
          used environment

    Returns:
        Any: The serialized instance within the input_folder
    """

    folder = Path(input_folder)

    metadata = _validate_folder_content(folder=folder)

    _check_environment_compatibility(metadata=metadata)

    instance = _load_instance(gz_path=folder / metadata[FUNCTION_DICT_KEY], extraction_folder=folder, remote=remote)

    if not remote:
        try:
            loaded_instance = instance.load_local_state(folder / metadata[MODEL_DICT_KEY])
        except ModuleNotFoundError as e:
            raise exceptions.LoadLocalDependencyError(
                "The instance from the given input folder requires the installation of "
                "additional dependencies. Those can be found in "
                f"{str(folder / SUBSTRAFL_FOLDER / 'installable_library')}"
                f"\nFull trace of the error: {e}"
            )

    else:
        loaded_instance = instance.load_shared(folder / metadata[MODEL_DICT_KEY])

    return loaded_instance


def _download_task_output_files(
    *,
    client: substra.Client,
    compute_plan_key: str,
    dest_folder: Path,
    task_type: TaskType,
    identifier: OutputIdentifiers,
    round_idx: Optional[int] = None,
    rank_idx: Optional[int] = None,
) -> None:
    """Download all the files needed to load the SubstraFL instance:

        - hosted on the client organization
        - being part of the given compute plan
        - being the result of the associated strategy after `round_idx` steps or at rank `rank_idx`

    into memory.

    Those files are:

        - the function used for this task
        - the output state of the task
        - a metadata.json

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        dest_folder (Path): Folder where to download the files.
        task_type (TaskType): Type of the task to fetch. Must be a valid SubstraFL TaskType
        identifier (OutputIdentifiers): Identifier of the output to fetch. Must be a valid OutputIdentifiers regarding
            the ``task_type``.
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the rank_idx will be used. (Defaults to None)
        rank_idx (Optional[int], None): Rank of the strategy to fetch the model from. If set to ``None``, the last task
            (with the highest rank) will be used. (Defaults to None)

    Raises:
        NotImplementedError: The given compute plan must have been submitted to Substra through the
            :func:`~substrafl.experiment.execute_experiment` function.
        exceptions.TaskNotFoundError: If no task matches the given requirements.
        exceptions.MultipleTaskError: The experiment to get the model from can't have multiple
            TrainDataNodes hosted on the same organization. In practice this means the presence of multiple
            tasks with the same round number on the same rank.
        exceptions.UnfinishedTaskError: The task from which the files are trying to be downloaded is not done.

    Returns:
        None
    """
    compute_plan = client.get_compute_plan(compute_plan_key)

    _check_environment_compatibility(metadata=compute_plan.metadata)

    folder = Path(dest_folder)
    folder.mkdir(exist_ok=True, parents=True)

    # Get the task associated to user inputs
    if round_idx is not None and rank_idx is not None:
        raise exceptions.ArgumentConflictError("Only one out of round_idx and rank_idx must be specified.")

    if round_idx is not None:
        task = _get_task_from_round(
            client=client,
            compute_plan_key=compute_plan_key,
            round_idx=round_idx,
            tag=task_type,
        )
    else:
        task = _get_task_from_rank(
            client=client,
            compute_plan_key=compute_plan_key,
            rank_idx=rank_idx,
            tag=task_type,
        )
    if task.status is not ComputeTaskStatus.done:
        raise exceptions.UnfinishedTaskError(
            f"Can't download algo files form task {task.key} as it is " f"in status {task.status}"
        )

    function_file = client.download_function(task.function.key, destination_folder=folder)

    # Get the associated model
    model_file = client.download_model_from_task(task.key, folder=folder, identifier=identifier)

    # Environment requirements and local state path
    metadata = {k: v for k, v in compute_plan.metadata.items() if k in REQUIRED_KEYS}
    metadata[MODEL_DICT_KEY] = str(model_file.relative_to(folder))
    metadata[FUNCTION_DICT_KEY] = str(function_file.relative_to(folder))
    metadata_path = folder / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata))


def download_algo_state(
    client: substra.Client,
    compute_plan_key: str,
    round_idx: Optional[int] = None,
    rank_idx: Optional[int] = None,
) -> Any:
    """Download a SubstraFL Algo instance at a given state:

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the rank_idx will be used. (Defaults to None)
        rank_idx (Optional[int], None): Rank of the strategy to fetch the model from. If set to ``None``, the last task
            (with the highest rank) will be used. (Defaults to None)

    Returns:
        Any: The serialized algo instance fetch from the given state.
    """

    with tempfile.TemporaryDirectory() as temp_folder:
        _download_task_output_files(
            client=client,
            compute_plan_key=compute_plan_key,
            dest_folder=temp_folder,
            round_idx=round_idx,
            rank_idx=rank_idx,
            task_type=TaskType.TRAIN,
            identifier=OutputIdentifiers.local,
        )

        algo = _load_from_files(input_folder=temp_folder, remote=False)

    return algo


def download_train_shared_state(
    client: substra.Client,
    compute_plan_key: str,
    round_idx: Optional[int] = None,
    rank_idx: Optional[int] = None,
) -> Any:
    """Download a SubstraFL shared object at a given state:

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the rank_idx will be used. (Defaults to None)
        rank_idx (Optional[int], None): Rank of the strategy to fetch the model from. If set to ``None``, the last task
            (with the highest rank) will be used. (Defaults to None)

    Returns:
        Any: The serialized shared instance fetch from the given state.
    """
    with tempfile.TemporaryDirectory() as temp_folder:
        _download_task_output_files(
            client=client,
            compute_plan_key=compute_plan_key,
            dest_folder=temp_folder,
            round_idx=round_idx,
            rank_idx=rank_idx,
            task_type=TaskType.TRAIN,
            identifier=OutputIdentifiers.shared,
        )
        shared = _load_from_files(input_folder=temp_folder, remote=True)

    return shared


def download_aggregate_shared_state(
    client: substra.Client,
    compute_plan_key: str,
    round_idx: Optional[int] = None,
    rank_idx: Optional[int] = None,
) -> Any:
    """Download a SubstraFL aggregated object at a given state:

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the rank_idx will be used. (Defaults to None)
        rank_idx (Optional[int], None): Rank of the strategy to fetch the model from. If set to ``None``, the last task
            (with the highest rank) will be used. (Defaults to None)

    Returns:
        Any: The serialized aggregated instance fetch from the given state.
    """
    with tempfile.TemporaryDirectory() as temp_folder:
        _download_task_output_files(
            client=client,
            compute_plan_key=compute_plan_key,
            dest_folder=temp_folder,
            round_idx=round_idx,
            rank_idx=rank_idx,
            task_type=TaskType.AGGREGATE,
            identifier=OutputIdentifiers.shared,
        )
        aggregated = _load_from_files(input_folder=temp_folder, remote=True)

    return aggregated
