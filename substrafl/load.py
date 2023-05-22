import json
import logging
import os
import sys
import tarfile
from pathlib import Path
from platform import python_version
from typing import Any
from typing import Optional

import substra
import substratools
from substra.sdk.models import Status

import substrafl
from substrafl import exceptions
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.register.register import SUBSTRAFL_FOLDER
from substrafl.remote.remote_struct import RemoteStruct

logger = logging.getLogger(__name__)

REQUIRED_LIBS = [substrafl, substra, substratools]
REQUIRED_KEYS = set([lib.__name__ + "_version" for lib in REQUIRED_LIBS] + ["python_version", "num_rounds"])
METADATA_FILE = "metadata.json"
ALGO_DICT_KEY = "algo_file"
LOCAL_STATE_DICT_KEY = "local_state_file"


def _check_environment_compatibility(metadata: dict):
    """Checks whether or not the local environment is compatible with the execution environment of the
    compute plan.

    Args:
        metadata (dict): Compute plan's metadata to download the model from.
    """

    if REQUIRED_KEYS.intersection(set(metadata.keys())) != REQUIRED_KEYS:
        raise exceptions.NotImplementedError(
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


def _validate_load_algo_inputs(folder: Path) -> dict:
    """Checks if the input folder is containing the necessary files to load a model with the
    :func:`~substrafl.load.load_algo` function. It can be generated with the
    :func:`~substrafl.load.download_algo_files` function.

    Args:
        folder (Path): Folder generated with the :func:`~substrafl.load.download_algo_files` function.

    Returns:
        dict: execution environment metadata of the model stored in the given folder
    """
    metadata_path = folder / METADATA_FILE
    local_state_file = None

    missing = []
    end_of_msg = (
        "This folder should be the result of the `substrafl.download_algo_files` function "
        f"and contain the {METADATA_FILE}, model and algo files."
        "being the local state of the model to load within memory."
    )

    if not metadata_path.exists():
        missing.append(METADATA_FILE)
    else:
        metadata = json.loads(metadata_path.read_text())

        # if metadata file exist we check that the LOCAL_STATE_DICT_KEY key is provided
        if LOCAL_STATE_DICT_KEY not in metadata:
            raise exceptions.LoadAlgoMetadataError(
                f"The {METADATA_FILE} file from the specified folder should contain a `{LOCAL_STATE_DICT_KEY}` key"
                "pointing the downloaded local state of the model to load within memory."
            )

        # if metadata file exist we check that the ALGO_DICT_KEY key is provided
        elif ALGO_DICT_KEY not in metadata:
            raise exceptions.LoadAlgoMetadataError(
                f"The {METADATA_FILE} file from the specified folder should contain an `{ALGO_DICT_KEY}` key"
                "pointing the downloaded algo file to load within memory."
            )

        local_state_file = metadata[LOCAL_STATE_DICT_KEY]
        algo_path = metadata[ALGO_DICT_KEY]

        # And that the pointed files exists
        if not ((folder / local_state_file).exists()):
            missing.append(local_state_file)
            end_of_msg += f", `{local_state_file}`"

        if not ((folder / algo_path).exists()):
            missing.append(algo_path)
            end_of_msg += f", `{algo_path}`"

    if len(missing) > 0:
        raise exceptions.LoadAlgoFileNotFoundError(
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
    local_train_task = local_tagged_tasks[0]

    return local_train_task


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
            the last ending task will be considered. (Default to None)
        tag (str): Tag of the task to get.

    Returns:
        substra.models.Task: The task matching the given requirements.
    """
    org_id = client.organization_info().organization_id

    if rank_idx is not None:
        filters = {"compute_plan_key": [compute_plan_key], "worker": [org_id], "rank": [rank_idx]}
        local_tasks = client.list_task(filters=filters)
        local_tagged_tasks = [t for t in local_tasks if t.tag == tag]

        if len(local_tagged_tasks) == 0:
            raise exceptions.TaskNotFoundError(
                f"The given compute plan `{compute_plan_key}` has no {tag} task of rank {rank_idx} "
                f"hosted on the organization {org_id}"
            )

        elif len(local_tagged_tasks) > 1:
            raise exceptions.MultipleTaskError(
                f"The given compute plan has {len(local_tagged_tasks)} local {tag} tasks of rank {rank_idx}. "
            )
        local_task = local_tagged_tasks[0]

    else:
        filters = {"compute_plan_key": [compute_plan_key], "worker": [org_id]}
        local_tasks = client.list_task(filters=filters, order_by="end_date")
        local_task = [t for t in local_tasks if t.tag == tag][-1:]

    return local_task


def _load_instance(gz_path: Path, extraction_folder: Path) -> Any:
    """Load into memory a serialized (and compressed (.tar.gz)) SubstraFL Remote Struct instance within the
    given path.
    This kind of file is usually the result of the ``substra.Client.download_function`` function applied to
    a train task being part of a SubstraFL experiment.

    Args:
        gz_path (Path): A file being the tar.gz compression of a SubstraFL RemoteStruct instance
        extraction_folder (Path): Where to unpack the folder.

    Returns:
        Any: The loaded SubstraFL object into memory.
    """

    with tarfile.open(gz_path, "r:gz") as tar:
        tar.extractall(path=extraction_folder)

    sys.path.append(str(extraction_folder))  # for local dependencies

    remote_struct = RemoteStruct.load(src=extraction_folder / SUBSTRAFL_FOLDER)

    instance = remote_struct.get_instance()

    return instance


def download_algo_files(
    *,
    client: substra.Client,
    compute_plan_key: str,
    dest_folder: os.PathLike,
    task_type: str,
    round_idx: Optional[int] = None,
    rank_idx: Optional[int] = None,
):
    """Download all the files needed to load the SubstraFL instance:

        - hosted on the client organization
        - being part of the given compute plan
        - being the result of the associated strategy after `round_idx` steps or at rank `rank_idx`

    into memory.

    Those files are:

        - the function used for this task
        - the output local state of the task
        - a metadata.json

    Important:
        This function supports only strategies with one train task for a given organization and round.

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        dest_folder (os.PathLike): Folder where to download the files.
        task_type (str): Type of the task to fetch. Allowed value are "init", "train" and "aggregate".
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the rank_idx will be used. (Defaults to None)
        rank_idx (Optional[int], None): Rank of the strategy to fetch the model from. If set to ``None``, the last task
            (with the highest rank) will be used. (Defaults to None)

    Raises:
        exceptions.NotImplementedError: The given compute plan must have been submitted to Substra through the
            :func:`~substrafl.experiment.execute_experiment` function.
        exceptions.TrainTaskNotFoundError: If no train task matches the given requirements.
        exceptions.MultipleTrainTaskError: The experiment to get the model from can't have multiple
            TrainDataNodes hosted on the same organization. In practice this means the presence of multiple
            train tasks with the same round number on the same rank.
        exceptions.UnfinishedTrainTaskError: The task from which the files are trying to be downloaded is not done.
    """
    compute_plan = client.get_compute_plan(compute_plan_key)

    _check_environment_compatibility(metadata=compute_plan.metadata)

    folder = Path(dest_folder)
    folder.mkdir(exist_ok=True, parents=True)

    # Get the task associated to user inputs
    if round_idx is not None and rank_idx is not None:
        raise exceptions.ArgumentConflictError("Only one out of round_idx and rank_idx must be specified.")
    elif round_idx is not None:
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
            rank_idx=round_idx,
            tag=task_type,
        )

    if task.status is not Status.done:
        raise exceptions.UnfinishedTaskError(
            f"Can't download algo files form task {task.key} as it is " f"in status {task.status}"
        )

    algo_file = client.download_function(task.function.key, destination_folder=folder)

    # Get the associated head model (local state)
    local_state_file = client.download_model_from_task(task.key, folder=folder, identifier=OutputIdentifiers.local)

    # Environment requirements and local state path
    metadata = {k: v for k, v in compute_plan.metadata.items() if k in REQUIRED_KEYS}
    metadata[LOCAL_STATE_DICT_KEY] = str(local_state_file.relative_to(folder))
    metadata[ALGO_DICT_KEY] = str(algo_file.relative_to(folder))
    metadata_path = folder / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata))


def load_algo(input_folder: os.PathLike) -> Any:
    """Loads an algo from a specified folder. This folder should contains:

        - function.tar.gz
        - metadata.json
        - the file specified in metadata.local_state_file

    This kind of folder can be generated with the :func:`~substrafl.load.download_algo_files`
    function.

    Args:
        input_folder (os.PathLike): Path to folder containing the required files.

    Raises:
        exceptions.LoadAlgoMetadataError: The metadata file must contains the local_state_file key
        exceptions.LoadAlgoFileNotFoundError: At least one of the required file to load the model is not found
        exceptions.LoadAlgoLocalDependencyError: One of the dependency used by the algo is not installed within the
          used environment

    Returns:
        Any: The serialized algo within the input_folder
    """

    folder = Path(input_folder)

    metadata = _validate_load_algo_inputs(folder=folder)

    _check_environment_compatibility(metadata=metadata)

    try:
        algo = _load_instance(algo_path=folder / metadata[ALGO_DICT_KEY], extraction_folder=folder)
        local_state = algo.load(folder / metadata[LOCAL_STATE_DICT_KEY])

    except ModuleNotFoundError as e:
        raise exceptions.LoadAlgoLocalDependencyError(
            "The algo from the given input folder requires the installation of "
            "additional dependencies. Those can be found in "
            f"{str(folder / 'substrafl_internal' / 'installable_library')}"
            f"\nFull trace of the error: {e}"
        )

    return local_state
