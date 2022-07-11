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

import connectlib
from connectlib.exceptions import LoadAlgoFileNotFoundError
from connectlib.exceptions import LoadAlgoLocalDependencyError
from connectlib.exceptions import LoadAlgoMetadataError
from connectlib.exceptions import MultipleTrainTaskError
from connectlib.exceptions import TrainTaskNotFoundError
from connectlib.exceptions import UnfinishedTrainTaskError
from connectlib.remote.register.register import CONNECTLIB_FOLDER
from connectlib.remote.remote_struct import RemoteStruct

logger = logging.getLogger(__name__)

REQUIRED_LIBS = [connectlib, substra, substratools]
REQUIRED_KEYS = set([lib.__name__ + "_version" for lib in REQUIRED_LIBS] + ["python_version", "num_rounds"])
METADATA_FILE = "metadata.json"
ALGO_FILE = "algo.tar.gz"
LOCAL_STATE_KEY = "local_state_file"


def _check_client_compatibility(client: substra.Client):
    """Checks wether the client backend is compatible with the download model files feature.

    Args:
        client (substra.Client): Substra client used to download the model from.
    """

    if client.backend_mode != substra.BackendType.DEPLOYED:
        logger.warning(
            "`download_algo` function is not fully supported yet for local backend. This could "
            "lead to unexpected behaviors and errors."
        )


def _check_environment_compatibility(metadata: dict):
    """Checks whether or not the local environment is compatible with the execution environment of the
    compute plan.

    Args:
        metadata (dict): Compute plan's metadata to download the model from.
    """

    if REQUIRED_KEYS.intersection(set(metadata.keys())) != REQUIRED_KEYS:
        raise NotImplementedError(
            "The given metadata doesn't seems to belong to a connectlib compute plan. "
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
    :func:`~connectlib.model_loading.load_algo` function. It can be generated with the
    :func:`~connectlib.model_loading.download_algo_files` function.

    Args:
        folder (Path): Folder generated with the :func:`~connectlib.model_loading.download_algo_files` function.

    Returns:
        dict: execution environment metadata of the model stored in the given folder
    """
    algo_path = folder / ALGO_FILE
    metadata_path = folder / METADATA_FILE
    local_state_file = None

    missing = []
    end_of_msg = (
        "This folder should be the result of the `connectlib.download_algo_files` function "
        f"and contain the following files: `{ALGO_FILE}`, `{METADATA_FILE}` and file staring with `model_` "
        "being the local state of the model to load within memory."
    )

    if not algo_path.exists():
        missing.append(ALGO_FILE)

    if not metadata_path.exists():
        missing.append(METADATA_FILE)
    else:
        metadata = json.loads(metadata_path.read_text())

        # if metadata file exist we check that the LOCAL_STATE_KEY key is provided
        local_state_file = metadata.get(LOCAL_STATE_KEY)

        if local_state_file is None:
            raise LoadAlgoMetadataError(
                "The metadata.json file from the specified folder should contain a `local_state_file` key"
                "pointing the the downloaded local state of the model to load within memory."
            )

        # And that the pointed file exists
        elif not ((folder / local_state_file).exists()):
            missing.append(local_state_file)
            end_of_msg += f", `{local_state_file}`"

    if len(missing) > 0:
        raise LoadAlgoFileNotFoundError(
            ", ".join(missing) + f" not found within the provided input folder `{folder}`.\n" + end_of_msg
        )

    return metadata


def _get_composite_from_round(
    client: substra.Client, compute_plan_key: str, round_idx: int
) -> substra.models.CompositeTraintuple:
    """Return the composite train tuple:

        - hosted on the given client organization
        - belonging to the given compute plan
        - of the given round_idx

    Args:
        client (substra.Client): Substra client where to fetch the composite train tuple from.
        compute_plan_key (str): Compute plan key to fetch the composite from.
        round_idx (int): Round of the strategy to fetch the composite from.

    Returns:
        substra.models.CompositeTraintuple: The composite matching the given requirements.
    """

    filters = {"compute_plan_key": [compute_plan_key]}
    org_id = client.organization_info().get("organization_id")

    # remote mode, in local mode the org_id is None
    if org_id is not None:
        filters["worker"] = [org_id]

    filters["metadata"] = [{"key": "round_idx", "type": "is", "value": str(round_idx)}]

    composite_traintuples = client.list_composite_traintuple(filters=filters)

    if len(composite_traintuples) == 0:
        raise TrainTaskNotFoundError(
            f"The given compute plan `{compute_plan_key}` has no composite train tuple of round {round_idx} "
            f"hosted on the organization {org_id}"
        )

    elif len(composite_traintuples) > 1:
        if client.backend_mode != substra.BackendType.DEPLOYED:
            logger.warning(
                "The given compute plan has {} composite train tuples for the round {}. The one with the "
                "highest rank is used to get the model".format(
                    str(len(composite_traintuples)),
                    str(round_idx),
                )
            )

            composite_traintuples = sorted(composite_traintuples, key=lambda x: x.rank, reverse=True)

        else:
            raise MultipleTrainTaskError(
                "The given compute plan has {} composite train tuples of round_idx {}. Downloading a model "
                "from an experiment containing multiple TrainDataNodes hosted on the same organization is "
                "not supported yet in local mode.".format(
                    str(len(composite_traintuples)),
                    str(round_idx),
                )
            )
    composite_traintuple = composite_traintuples[0]

    return composite_traintuple


def _load_algo(algo_path: Path, extraction_folder: Path) -> Any:
    """Load into memory a serialized (and compressed (.tar.gz)) connectlib algo within the given algo_path.
    This kind of file is usually the result of the ``substra.Client.download_algo`` function applied to
    a composite train tuple being part of a Connectlib experiment.

    Args:
        algo_path (Path): A file being the tar.gz compression of a connectlib RemoteStruct algorithm
        extraction_folder (Path): Where to unpack the folder.

    Returns:
        Any: The loaded connectlib object into memory.
    """

    with tarfile.open(algo_path, "r:gz") as tar:
        tar.extractall(path=extraction_folder)

    sys.path.append(str(extraction_folder))  # for local dependencies

    remote_struct = RemoteStruct.load(src=extraction_folder / CONNECTLIB_FOLDER)

    my_algo = remote_struct.get_instance()

    return my_algo


def download_algo_files(
    client: substra.Client,
    compute_plan_key: str,
    dest_folder: os.PathLike,
    round_idx: Optional[int] = None,
):
    """Download all the files needed to load the model:

        - hosted on the client organization
        - being part of the given compute plan
        - being the result of the associated strategy ofter `round_idx` steps

    into memory.

    Those files are:

        - the algorithm used for this task
        - the output local state of the task
        - a metadata.json

    Important:
        This function supports only strategies with one composite traintuple for a given organization and round.

    Args:
        client (substra.Client): Substra client where to fetch the model from.
        compute_plan_key (str): Compute plan key to fetch the model from.
        dest_folder (os.PathLike): Folder where to download the files.
        round_idx (Optional[int], None): Round of the strategy to fetch the model from. If set to ``None``,
            the last round will be used. (Default to None).

    Raises:
        NotImplementedError: The given compute plan must have been submitted to connect through the
            :func:`~connectlib.experiment.execute_experiment` function.
        TrainTaskNotFoundError: If no composite matches the given requirements.
        MultipleTrainTaskError: The experiment to get the model from can't have multiple
            TrainDataNodes hosted on the same organization. In practice this means the presence of multiple
            composite train tuples with the same round number on the same rank.
        UnfinishedTrainTaskError: The task from which the files are trying to be downloaded is not done.
    """
    _check_client_compatibility(client=client)

    compute_plan = client.get_compute_plan(compute_plan_key)

    _check_environment_compatibility(metadata=compute_plan.metadata)

    folder = Path(dest_folder)
    folder.mkdir(exist_ok=True, parents=True)

    if round_idx is None:
        round_idx = compute_plan.metadata["num_rounds"]

    # Get the composite associated to user inputs
    composite_traintuple = _get_composite_from_round(
        client=client, compute_plan_key=compute_plan_key, round_idx=round_idx
    )

    if composite_traintuple.status is not Status.done:
        raise UnfinishedTrainTaskError(
            f"Can't download algo files form task {composite_traintuple.key} as it is "
            f"in status {composite_traintuple.status}"
        )

    client.download_algo(composite_traintuple.algo.key, destination_folder=folder)

    # Get the associated head model (local state)
    client.download_head_model_from_composite_traintuple(composite_traintuple.key, folder=folder)
    head_model_key = [
        model.key for model in composite_traintuple.composite.models if model.category == substra.models.ModelType.head
    ]
    head_model_key = head_model_key[0]
    local_state_file = folder / f"model_{head_model_key}"

    # Environment requirements and local state path
    metadata = {k: v for k, v in compute_plan.metadata.items() if k in REQUIRED_KEYS}
    metadata[LOCAL_STATE_KEY] = str(local_state_file)
    metadata_path = folder / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata))


def load_algo(input_folder: os.PathLike) -> Any:
    """Loads an algo from a specified folder. This folder should contains:

        - algo.tar.gz
        - metadata.json
        - the file specified in metadata.local_state_file

    This kind of folder can be generated with the :func:`~connectlib.model_loading.download_algo_files`
    function.

    Args:
        input_folder (os.PathLike): Path to folder containing the required files.

    Raises:
        LoadAlgoMetadataError: The metadata file must contains the local_state_file key
        LoadAlgoFileNotFoundError: At least one of the required file to load the model is not found
        LoadAlgoLocalDependencyError: One of the dependency used by the algo is not installed within the the used
            environnement

    Returns:
        Any: The serialized algo within the input_folder
    """

    folder = Path(input_folder)

    algo_path = folder / ALGO_FILE

    metadata = _validate_load_algo_inputs(folder=folder)

    _check_environment_compatibility(metadata=metadata)

    try:
        algo = _load_algo(algo_path=algo_path, extraction_folder=folder)
        local_state = algo.load(Path(metadata[LOCAL_STATE_KEY]))

    except ModuleNotFoundError as e:
        raise LoadAlgoLocalDependencyError(
            "The algo from the given input folder requires the installation of "
            "additional dependencies. Those can be found in "
            f"{str(folder / 'connectlib_internal' / 'installable_library')}"
            f"\nFull trace of the error: {e}"
        )

    return local_state
