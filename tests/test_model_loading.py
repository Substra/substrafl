import enum
import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from platform import python_version
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
import substra
import substratools
from substra.sdk.models import Algo
from substra.sdk.models import CompositeTraintuple
from substra.sdk.models import ComputePlan
from substra.sdk.models import OutModel
from substra.sdk.models import Status
from substra.sdk.models import _Composite

import connectlib
from connectlib.dependency import Dependency
from connectlib.exceptions import LoadAlgoFileNotFoundError
from connectlib.exceptions import LoadAlgoLocalDependencyError
from connectlib.exceptions import LoadAlgoMetadataError
from connectlib.exceptions import MultipleTrainTaskError
from connectlib.exceptions import TrainTaskNotFoundError
from connectlib.exceptions import UnfinishedTrainTaskError
from connectlib.model_loading import ALGO_FILE
from connectlib.model_loading import LOCAL_STATE_KEY
from connectlib.model_loading import METADATA_FILE
from connectlib.model_loading import REQUIRED_KEYS
from connectlib.model_loading import download_algo_files
from connectlib.model_loading import load_algo
from connectlib.remote.register.register import _create_substra_algo_files

FILE_PATH = Path(__file__).resolve().parent


class AssetKeys(str, enum.Enum):
    compute_plan = "you"
    algo = "were"
    valid_head_model = "the"
    trunk_model = "chosen"
    composite_traintuple = "one"
    invalid_head_model = "Anakin"


@pytest.fixture
def fake_compute_plan():
    compute_plan = Mock(spec=ComputePlan)
    compute_plan.key = AssetKeys.compute_plan
    compute_plan.metadata = {
        "connectlib_version": connectlib.__version__,
        "substra_version": substra.__version__,
        "substratools_version": substratools.__version__,
        "python_version": python_version(),
        "num_rounds": "4",
    }

    return compute_plan


@pytest.fixture
def trunk_model():
    model = Mock(spec=OutModel)
    model.category = substra.models.ModelType.simple
    model.key = AssetKeys.trunk_model

    return model


@pytest.fixture
def fake_composite_traintuple(trunk_model):
    algo = Mock(spec=Algo)
    algo.key = AssetKeys.algo

    head_model = Mock(spec=OutModel)
    head_model.category = substra.models.ModelType.head
    head_model.key = AssetKeys.valid_head_model

    composite = Mock(spec=_Composite)
    composite.models = [head_model, trunk_model]

    composite_traintuple = Mock(spec=CompositeTraintuple)
    composite_traintuple.rank = 2
    composite_traintuple.key = AssetKeys.composite_traintuple
    composite_traintuple.algo = algo
    composite_traintuple.composite = composite
    composite_traintuple.status = Status.done

    return composite_traintuple


@pytest.fixture
def fake_client(fake_compute_plan, fake_composite_traintuple):
    client = Mock(spec=substra.Client)
    client.backend_mode = substra.BackendType.DEPLOYED
    client.get_compute_plan = MagicMock(return_value=fake_compute_plan)
    client.organization_info = MagicMock(return_value={"organization_id": "Org1"})
    client.list_composite_traintuple = MagicMock(return_value=[fake_composite_traintuple])
    client.download_algo = MagicMock(
        side_effect=lambda key, destination_folder: (Path(destination_folder) / ALGO_FILE).write_text("Hello there !")
    )
    client.download_head_model_from_composite_traintuple = MagicMock(
        side_effect=lambda tuple_key, folder: (Path(folder) / f"model_{AssetKeys.valid_head_model}").write_text(
            "General Kenobi ..."
        )
    )

    return client


@pytest.fixture(params=[True, False])
def is_dependency_uninstalled(request):
    return request.param


@pytest.fixture
def algo_files_with_local_dependency(session_dir, fake_compute_plan, dummy_algo_class, is_dependency_uninstalled):
    """Check that function load_algo raises a custom error in case of non-installed dependency and that it works with
    installed dependencies."""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata
    metadata.update({LOCAL_STATE_KEY: str(input_folder / "model")})

    subprocess.check_output([sys.executable, "-m", "pip", "install", "."], cwd=str(FILE_PATH / "installable_library"))

    class MyAlgo(dummy_algo_class):
        def load(self, path):
            import connectlibtestlibrary

            return connectlibtestlibrary.dummy_string_function("hello ", "world")

    my_algo = MyAlgo()

    _create_algo_files(input_folder, my_algo, metadata)

    if is_dependency_uninstalled:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "connectlibtestlibrary"], check=True)

    yield input_folder
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "connectlibtestlibrary"])


def test_download_algo_files(fake_client, fake_compute_plan, session_dir, caplog):
    """No warning and expected files matching the given names in the metadata.json"""
    dest_folder = session_dir / str(uuid.uuid4())

    expected_metadata = fake_compute_plan.metadata
    expected_metadata.update({LOCAL_STATE_KEY: str(dest_folder / f"model_{AssetKeys.valid_head_model}")})

    caplog.clear()
    download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0

    metadata = json.loads((dest_folder / METADATA_FILE).read_text())

    assert expected_metadata == metadata
    assert (dest_folder / ALGO_FILE).exists()
    assert (dest_folder / metadata.get(LOCAL_STATE_KEY)).exists()


@pytest.mark.parametrize("backend_mode", [substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER])
def test_local_client_warning(fake_client, fake_compute_plan, session_dir, caplog, backend_mode):
    """Warning for local clients"""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.backend_mode = backend_mode
    caplog.clear()
    download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 1


@pytest.mark.parametrize("to_remove", list(REQUIRED_KEYS))
def test_environment_compatibility_error(fake_client, fake_compute_plan, to_remove, session_dir):
    """Error if one of the required key is not in the metadata."""
    dest_folder = session_dir / str(uuid.uuid4())

    del fake_compute_plan.metadata[to_remove]
    with pytest.raises(NotImplementedError):
        download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)


def test_retro_compatibility_warning(fake_client, fake_compute_plan, session_dir, caplog):
    """Warning if there is a difference of version between the running env and the one specified in the metadata."""
    dest_folder = session_dir / str(uuid.uuid4())
    pkg_versions = list(filter(lambda x: x.endswith("_version"), REQUIRED_KEYS))
    for pkg_version in pkg_versions:
        fake_compute_plan.metadata[pkg_version] = "error"
        name = pkg_version.split("_")[0]

        caplog.clear()
        download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)
        assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 1
        assert name in caplog.records[0].msg


def test_train_task_not_found(fake_client, fake_compute_plan, session_dir):
    """Error if no train task are found."""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.list_composite_traintuple = MagicMock(return_value=[])
    with pytest.raises(TrainTaskNotFoundError):
        download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)


def test_multiple_train_task_error(fake_client, fake_compute_plan, session_dir, fake_composite_traintuple):
    """With a deployed backend, error if multiple train tasks are found."""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.list_composite_traintuple = MagicMock(
        return_value=[fake_composite_traintuple, fake_composite_traintuple]
    )
    with pytest.raises(MultipleTrainTaskError):
        download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)


@pytest.mark.parametrize("backend_mode", [substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER])
def test_multiple_train_task_local(
    fake_client, fake_compute_plan, session_dir, fake_composite_traintuple, trunk_model, backend_mode, caplog
):
    """With a local backend, if multiple train tasks are found, the one with the highest rank is used and a warning is
    thrown"""
    dest_folder = session_dir / str(uuid.uuid4())

    fake_client.list_composite_traintuple = MagicMock(
        return_value=[fake_composite_traintuple, fake_composite_traintuple]
    )
    fake_client.backend_mode = backend_mode
    # This test needs to be removed when substra client will simulates multiple clients

    # Let's define an other composite of smaller rank and let's check that the downloaded model
    # is the right one. We can check this in the metadata as the name of the model is
    # computed from the head_model key.
    # TODO: simplify the test once substra returns the file path with the get methods
    smaller_fake_composite_traintuple = deepcopy(fake_composite_traintuple)
    smaller_fake_composite_traintuple.rank = 1
    head_model = Mock(spec=OutModel)
    head_model.category = substra.models.ModelType.head
    head_model.key = AssetKeys.invalid_head_model
    composite = Mock(spec=_Composite)
    composite.models = [head_model, trunk_model]

    caplog.clear()
    fake_client.list_composite_traintuple = MagicMock(
        return_value=[
            smaller_fake_composite_traintuple,
            smaller_fake_composite_traintuple,
            fake_composite_traintuple,
            smaller_fake_composite_traintuple,
        ]
    )
    download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)
    assert (
        len(
            list(filter(lambda x: x.levelname == "WARNING" and "has 4 composite train tuples" in x.msg, caplog.records))
        )
        == 1
    )
    metadata = json.loads((dest_folder / METADATA_FILE).read_text())

    assert metadata.get(LOCAL_STATE_KEY) == str(dest_folder / f"model_{AssetKeys.valid_head_model}")


def _create_algo_files(input_folder, algo, metadata):

    # model file
    if metadata.get(LOCAL_STATE_KEY):
        Path(metadata.get(LOCAL_STATE_KEY)).write_text("True")

    # metadata.json file
    metadata_file = input_folder / METADATA_FILE
    metadata_file.write_text(json.dumps(metadata))

    data_operation = algo.train(data_samples=[])

    _create_substra_algo_files(
        remote_struct=data_operation.remote_struct,
        install_libraries=True,
        dependencies=Dependency(local_dependencies=[str(FILE_PATH / "installable_library")]),
        operation_dir=input_folder,
    )


def test_load_algo(session_dir, fake_compute_plan, dummy_algo_class, caplog):
    """Checks that the load_algo method can load the file given by connectlib to substra
    and that the state of the algo is properly updated"""

    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    class MyAlgo(dummy_algo_class):
        def __init__(self) -> None:
            super().__init__()
            self._updated = False

        def load(self, path):
            self._updated = eval(Path(path).read_text())

            return self

    my_algo = MyAlgo()

    metadata = fake_compute_plan.metadata
    metadata.update({LOCAL_STATE_KEY: str(input_folder / "model")})

    _create_algo_files(input_folder, my_algo, metadata)

    caplog.clear()
    my_loaded_algo = load_algo(input_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0
    assert my_loaded_algo._updated


@pytest.mark.parametrize("to_remove", [ALGO_FILE, METADATA_FILE, "model"])
def test_missing_file_error(session_dir, fake_compute_plan, dummy_algo_class, to_remove):
    """Checks that the load_algo method raises an error if one of the needed file is not found."""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata
    metadata.update({LOCAL_STATE_KEY: str(input_folder / "model")})

    _create_algo_files(input_folder, dummy_algo_class(), metadata)

    os.remove(input_folder / to_remove)
    with pytest.raises(LoadAlgoFileNotFoundError):
        load_algo(input_folder)


def test_missing_local_state_key_error(session_dir, fake_compute_plan, dummy_algo_class):
    """Error if the local state key is not provided"""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata

    _create_algo_files(input_folder, dummy_algo_class(), metadata)

    with pytest.raises(LoadAlgoMetadataError):
        load_algo(input_folder)


def test_load_model_dependency(algo_files_with_local_dependency, is_dependency_uninstalled):
    """Check that function load_algo raises a custom error in case of the use of an un installed dependency
    and that it works with installed dependencies."""

    input_folder = algo_files_with_local_dependency

    if is_dependency_uninstalled:
        with pytest.raises(LoadAlgoLocalDependencyError):
            load_algo(input_folder)

    else:
        res = load_algo(input_folder)
        assert res == "hello world"


@pytest.mark.parametrize("status", [e.value for e in Status if e.value != Status.done])
def test_unfinished_task_error(fake_client, fake_compute_plan, fake_composite_traintuple, status, session_dir):
    """Raise error if the task status is not done"""
    with pytest.raises(UnfinishedTrainTaskError):
        fake_composite_traintuple.status = status
        download_algo_files(fake_client, fake_compute_plan.key, session_dir, round_idx=None)
