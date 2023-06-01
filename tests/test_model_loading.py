import enum
import json
import os
import subprocess
import sys
import uuid
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from platform import python_version
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
import substra
import substratools

import substrafl
from substrafl import exceptions
from substrafl.dependency import Dependency
from substrafl.model_loading import FUNCTION_DICT_KEY
from substrafl.model_loading import METADATA_FILE
from substrafl.model_loading import MODEL_DICT_KEY
from substrafl.model_loading import REQUIRED_KEYS
from substrafl.model_loading import _download_task_output_files
from substrafl.model_loading import _load_from_files
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.register.register import _create_substra_function_files

FILE_PATH = Path(__file__).resolve().parent


class AssetKeys(str, enum.Enum):
    compute_plan = "you"
    function = "were"
    valid_head_model = "the"
    trunk_model = "chosen"
    local_train_task = "one"
    invalid_head_model = "Anakin"
    aggregate_task = "PS: don't forget your ventolin"


@pytest.fixture(
    params=(
        ("train", OutputIdentifiers.local),
        ("train", OutputIdentifiers.shared),
        ("aggregate", OutputIdentifiers.model),
    )
)
def output_parameters(request):
    return request.param


@pytest.fixture
def fake_compute_plan():
    compute_plan = Mock(spec=substra.models.ComputePlan)
    compute_plan.key = AssetKeys.compute_plan
    compute_plan.metadata = {
        "substrafl_version": substrafl.__version__,
        "substra_version": substra.__version__,
        "substratools_version": substratools.__version__,
        "python_version": python_version(),
        "num_rounds": "4",
    }

    return compute_plan


@pytest.fixture
def trunk_model():
    model = Mock(spec=substra.models.OutModel)
    model.key = AssetKeys.trunk_model

    return model


@pytest.fixture
def fake_local_train_task(trunk_model):
    function = Mock(spec=substra.models.Function)
    function.key = AssetKeys.function

    head_model = Mock(spec=substra.models.OutModel)
    head_model.key = AssetKeys.valid_head_model

    local_train_task = Mock(spec=substra.models.Task)
    local_train_task.rank = 2
    local_train_task.key = AssetKeys.local_train_task
    local_train_task.function = function
    local_train_task.tag = "train"
    local_train_task.outputs = {
        "local": substra.models.ComputeTaskOutput(
            permissions=substra.models.Permissions(process={"public": True, "authorized_ids": []}), value=head_model
        ),
        "shared": substra.models.ComputeTaskOutput(
            permissions=substra.models.Permissions(process={"public": True, "authorized_ids": []}), value=trunk_model
        ),
    }
    local_train_task.status = substra.models.Status.done

    return local_train_task


@pytest.fixture
def fake_aggregate_task(trunk_model):
    function = Mock(spec=substra.models.Function)
    function.key = AssetKeys.function

    model = Mock(spec=substra.models.OutModel)
    model.key = AssetKeys.valid_head_model

    aggregate_task = Mock(spec=substra.models.Task)
    aggregate_task.rank = 3
    aggregate_task.key = AssetKeys.aggregate_task
    aggregate_task.function = function
    aggregate_task.tag = "aggregate"
    aggregate_task.outputs = {
        "model": substra.models.ComputeTaskOutput(
            permissions=substra.models.Permissions(process={"public": True, "authorized_ids": []}), value=model
        ),
    }
    aggregate_task.status = substra.models.Status.done

    return aggregate_task


@pytest.fixture
def fake_client(fake_compute_plan, fake_local_train_task, fake_aggregate_task):
    def download_model_from_task(task_key, identifier, folder):
        path = Path(folder) / f"model_{AssetKeys.valid_head_model}"
        path.write_text("General Kenobi ...")
        return path

    def download_function(key, destination_folder):
        path = Path(destination_folder) / "function.tar.gz"
        path.write_text("Hello there !")
        return path

    client = Mock(spec=substra.Client)
    client.backend_mode = substra.BackendType.REMOTE
    client.get_compute_plan = MagicMock(return_value=fake_compute_plan)
    client.organization_info = MagicMock(
        return_value=substra.models.OrganizationInfo(
            host="http://example.com",
            organization_id="Org1",
            organization_name="Org1",
            config=substra.models.OrganizationInfoConfig(model_export_enabled=True),
            channel="",
            version="",
            orchestrator_version="",
        )
    )
    client.list_task = MagicMock(return_value=[fake_local_train_task, fake_aggregate_task])
    client.download_function = MagicMock(
        side_effect=lambda key, destination_folder: download_function(key, destination_folder)
    )
    client.download_model_from_task = MagicMock(
        side_effect=lambda task_key, folder, identifier: download_model_from_task(
            task_key, folder=folder, identifier=identifier
        )
    )

    return client


@pytest.fixture(params=[True, False])
def is_dependency_uninstalled(request):
    return request.param


@pytest.fixture
def algo_files_with_local_dependency(session_dir, fake_compute_plan, dummy_algo_class, is_dependency_uninstalled):
    """Check that function _load_from_files raises a custom error in case of non-installed dependency and that it works with
    installed dependencies."""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata
    metadata.update({MODEL_DICT_KEY: "model"})
    metadata.update({FUNCTION_DICT_KEY: "function.tar.gz"})

    subprocess.check_output([sys.executable, "-m", "pip", "install", "."], cwd=str(FILE_PATH / "installable_library"))

    class MyAlgo(dummy_algo_class):
        def load_local_state(self, path):
            import substrafltestlibrary

            return substrafltestlibrary.dummy_string_function("hello ", "world")

    my_algo = MyAlgo()

    _create_files(input_folder, my_algo, metadata)

    if is_dependency_uninstalled:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "substrafltestlibrary"], check=True)

    yield input_folder
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "substrafltestlibrary"])


def test_download_utils_files(fake_client, fake_compute_plan, session_dir, caplog, output_parameters):
    """No warning and expected files matching the given names in the metadata.json"""
    dest_folder = session_dir / str(uuid.uuid4())

    expected_metadata = fake_compute_plan.metadata
    expected_metadata.update({MODEL_DICT_KEY: f"model_{AssetKeys.valid_head_model}"})
    expected_metadata.update({FUNCTION_DICT_KEY: "function.tar.gz"})

    caplog.clear()
    task_type, identifier = output_parameters
    _download_task_output_files(
        client=fake_client,
        compute_plan_key=fake_compute_plan.key,
        task_type=task_type,
        identifier=identifier,
        dest_folder=dest_folder,
    )
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0

    metadata = json.loads((dest_folder / METADATA_FILE).read_text())

    assert expected_metadata == metadata
    assert (dest_folder / metadata.get(FUNCTION_DICT_KEY)).exists()
    assert (dest_folder / metadata.get(MODEL_DICT_KEY)).exists()


@pytest.mark.parametrize(
    "round_idx, rank_idx, expectation",
    (
        [None, None, does_not_raise()],
        [1, 1, pytest.raises(exceptions.ArgumentConflictError)],
        [1, None, does_not_raise()],
        [None, 1, does_not_raise()],
    ),
)
def test_round_idx_vs_rank_idx(
    fake_client, fake_compute_plan, session_dir, output_parameters, round_idx, rank_idx, expectation
):
    dest_folder = session_dir / str(uuid.uuid4())
    task_type, identifier = output_parameters

    with expectation:
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            task_type=task_type,
            identifier=identifier,
            dest_folder=dest_folder,
            round_idx=round_idx,
            rank_idx=rank_idx,
        )


@pytest.mark.parametrize("to_remove", list(REQUIRED_KEYS))
def test_environment_compatibility_error(fake_client, fake_compute_plan, to_remove, output_parameters, session_dir):
    """Error if one of the required key is not in the metadata."""
    dest_folder = session_dir / str(uuid.uuid4())
    task_type, identifier = output_parameters

    del fake_compute_plan.metadata[to_remove]
    with pytest.raises(NotImplementedError):
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            task_type=task_type,
            identifier=identifier,
            dest_folder=dest_folder,
        )


def test_retro_compatibility_warning(fake_client, fake_compute_plan, session_dir, output_parameters, caplog):
    """Warning if there is a difference of version between the running env and the one specified in the metadata."""
    dest_folder = session_dir / str(uuid.uuid4())
    pkg_versions = list(filter(lambda x: x.endswith("_version"), REQUIRED_KEYS))
    task_type, identifier = output_parameters

    for pkg_version in pkg_versions:
        fake_compute_plan.metadata[pkg_version] = "error"
        name = pkg_version.split("_")[0]

        caplog.clear()
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            task_type=task_type,
            identifier=identifier,
            dest_folder=dest_folder,
        )
        assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 1
        assert name in caplog.records[0].msg


def test_task_not_found(fake_client, fake_compute_plan, session_dir, output_parameters):
    """Error if no train task are found."""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.list_task = MagicMock(return_value=[])
    task_type, identifier = output_parameters

    with pytest.raises(exceptions.TaskNotFoundError):
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            dest_folder=dest_folder,
            task_type=task_type,
            identifier=identifier,
            round_idx=fake_compute_plan.metadata["num_rounds"],
        )


def test_multiple_task_error(
    fake_client, fake_compute_plan, session_dir, fake_local_train_task, fake_aggregate_task, output_parameters
):
    """Error if multiple train tasks are found."""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.list_task = MagicMock(
        return_value=[fake_local_train_task, fake_local_train_task, fake_aggregate_task, fake_aggregate_task]
    )
    task_type, identifier = output_parameters

    with pytest.raises(exceptions.MultipleTaskError):
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            dest_folder=dest_folder,
            task_type=task_type,
            identifier=identifier,
            round_idx=fake_compute_plan.metadata["num_rounds"],
        )


def _create_files(input_folder, algo, metadata):
    # model file
    if metadata.get(MODEL_DICT_KEY):
        (input_folder / metadata.get(MODEL_DICT_KEY)).write_text("True")

    # algo file
    if metadata.get(FUNCTION_DICT_KEY):
        (input_folder / metadata.get(FUNCTION_DICT_KEY)).write_text("True")

    # metadata.json file
    (input_folder / METADATA_FILE).write_text(json.dumps(metadata))

    data_operation = algo.train(data_samples=[])

    _create_substra_function_files(
        remote_struct=data_operation.remote_struct,
        install_libraries=True,
        dependencies=Dependency(local_dependencies=[str(FILE_PATH / "installable_library")], editable_mode=True),
        operation_dir=input_folder,
    )


def test_load_algo(session_dir, fake_compute_plan, dummy_algo_class, caplog):
    """Checks that the _load_from_files method can load the file given by substrafl to substra
    and that the state of the algo is properly updated"""

    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    class MyAlgo(dummy_algo_class):
        def __init__(self) -> None:
            super().__init__()
            self._updated = False

        def load_local_state(self, path):
            self._updated = eval(Path(path).read_text())

            return self

    my_algo = MyAlgo()

    metadata = fake_compute_plan.metadata
    metadata.update({MODEL_DICT_KEY: "model"})
    metadata.update({FUNCTION_DICT_KEY: "function.tar.gz"})

    _create_files(input_folder, my_algo, metadata)

    caplog.clear()
    my_loaded_algo = _load_from_files(input_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0
    assert my_loaded_algo._updated


@pytest.mark.parametrize("to_remove", ["function.tar.gz", METADATA_FILE, "model"])
def test_missing_file_error(session_dir, fake_compute_plan, dummy_algo_class, to_remove):
    """Checks that the _load_from_files method raises an error if one of the needed file is not found."""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata
    metadata.update({MODEL_DICT_KEY: "model"})
    metadata.update({FUNCTION_DICT_KEY: "function.tar.gz"})

    _create_files(input_folder, dummy_algo_class(), metadata)

    os.remove(input_folder / to_remove)
    with pytest.raises(exceptions.LoadFileNotFoundError):
        _load_from_files(input_folder)


def test_missing_local_state_key_error(session_dir, fake_compute_plan, dummy_algo_class):
    """Error if the local state key is not provided"""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata

    _create_files(input_folder, dummy_algo_class(), metadata)

    with pytest.raises(exceptions.LoadMetadataError):
        _load_from_files(input_folder)


def test_load_model_dependency(algo_files_with_local_dependency, is_dependency_uninstalled):
    """Check that function _load_from_files raises a custom error in case of the use of an un installed dependency
    and that it works with installed dependencies."""

    input_folder = algo_files_with_local_dependency

    if is_dependency_uninstalled:
        with pytest.raises(exceptions.LoadLocalDependencyError):
            _load_from_files(input_folder)

    else:
        res = _load_from_files(input_folder)
        assert res == "hello world"


@pytest.mark.parametrize("status", [e.value for e in substra.models.Status if e.value != substra.models.Status.done])
def test_unfinished_task_error(
    fake_client,
    fake_compute_plan,
    fake_local_train_task,
    fake_aggregate_task,
    status,
    session_dir,
    output_parameters,
):
    """Raise error if the task status is not done"""

    task_type, identifier = output_parameters

    with pytest.raises(exceptions.UnfinishedTaskError):
        fake_local_train_task.status = status
        fake_aggregate_task.status = status
        _download_task_output_files(
            client=fake_client,
            compute_plan_key=fake_compute_plan.key,
            dest_folder=session_dir,
            task_type=task_type,
            identifier=identifier,
        )
