import enum
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from platform import python_version
from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pytest
import substra
import substratools
from substra.sdk.models import Algo
from substra.sdk.models import CompositeTraintuple
from substra.sdk.models import ComputePlan
from substra.sdk.models import OutModel
from substra.sdk.models import Status
from substra.sdk.models import _Composite

<<<<<<< HEAD
import substrafl
from substrafl.dependency import Dependency
from substrafl.exceptions import LoadAlgoFileNotFoundError
from substrafl.exceptions import LoadAlgoLocalDependencyError
from substrafl.exceptions import LoadAlgoMetadataError
from substrafl.exceptions import MultipleTrainTaskError
from substrafl.exceptions import TrainTaskNotFoundError
from substrafl.exceptions import UnfinishedTrainTaskError
from substrafl.model_loading import ALGO_DICT_KEY
from substrafl.model_loading import LOCAL_STATE_DICT_KEY
from substrafl.model_loading import METADATA_FILE
from substrafl.model_loading import REQUIRED_KEYS
from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo
from substrafl.remote.register.register import _create_substra_algo_files
=======
import connectlib
from connectlib.dependency import Dependency
from connectlib.exceptions import LoadAlgoFileNotFoundError
from connectlib.exceptions import LoadAlgoLocalDependencyError
from connectlib.exceptions import LoadAlgoMetadataError
from connectlib.exceptions import MultipleTrainTaskError
from connectlib.exceptions import TrainTaskNotFoundError
from connectlib.exceptions import UnfinishedTrainTaskError
from connectlib.experiment import execute_experiment
from connectlib.model_loading import ALGO_FILE
from connectlib.model_loading import LOCAL_STATE_KEY
from connectlib.model_loading import METADATA_FILE
from connectlib.model_loading import REQUIRED_KEYS
from connectlib.model_loading import download_algo_files
from connectlib.model_loading import load_algo
from connectlib.remote.decorators import remote_data
from connectlib.remote.register.register import _create_substra_algo_files
>>>>>>> c1fe4347 (tests: add independent e2e test for download model.)

from . import utils

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
        "substrafl_version": substrafl.__version__,
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
    def download_head_model_from_composite_traintuple(tuple_key, folder):
        path = Path(folder) / f"model_{AssetKeys.valid_head_model}"
        path.write_text("General Kenobi ...")
        return path

    def download_algo(key, destination_folder):
        path = Path(destination_folder) / "algo.tar.gz"
        path.write_text("Hello there !")
        return path

    client = Mock(spec=substra.Client)
    client.backend_mode = substra.BackendType.DEPLOYED
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
    client.list_composite_traintuple = MagicMock(return_value=[fake_composite_traintuple])
    client.download_algo = MagicMock(side_effect=lambda key, destination_folder: download_algo(key, destination_folder))
    client.download_head_model_from_composite_traintuple = MagicMock(
        side_effect=lambda tuple_key, folder: download_head_model_from_composite_traintuple(tuple_key, folder)
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
    metadata.update({LOCAL_STATE_DICT_KEY: "model"})
    metadata.update({ALGO_DICT_KEY: "algo.tar.gz"})

    subprocess.check_output([sys.executable, "-m", "pip", "install", "."], cwd=str(FILE_PATH / "installable_library"))

    class MyAlgo(dummy_algo_class):
        def load(self, path):
            import substrafltestlibrary

            return substrafltestlibrary.dummy_string_function("hello ", "world")

    my_algo = MyAlgo()

    _create_algo_files(input_folder, my_algo, metadata)

    if is_dependency_uninstalled:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "substrafltestlibrary"], check=True)

    yield input_folder
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "substrafltestlibrary"])


def test_download_algo_files(fake_client, fake_compute_plan, session_dir, caplog):
    """No warning and expected files matching the given names in the metadata.json"""
    dest_folder = session_dir / str(uuid.uuid4())

    expected_metadata = fake_compute_plan.metadata
    expected_metadata.update({LOCAL_STATE_DICT_KEY: f"model_{AssetKeys.valid_head_model}"})
    expected_metadata.update({ALGO_DICT_KEY: "algo.tar.gz"})

    caplog.clear()
    download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0

    metadata = json.loads((dest_folder / METADATA_FILE).read_text())

    assert expected_metadata == metadata
    assert (dest_folder / metadata.get(ALGO_DICT_KEY)).exists()
    assert (dest_folder / metadata.get(LOCAL_STATE_DICT_KEY)).exists()


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
    """Error if multiple train tasks are found."""
    dest_folder = session_dir / str(uuid.uuid4())
    fake_client.list_composite_traintuple = MagicMock(
        return_value=[fake_composite_traintuple, fake_composite_traintuple]
    )
    with pytest.raises(MultipleTrainTaskError):
        download_algo_files(client=fake_client, compute_plan_key=fake_compute_plan.key, dest_folder=dest_folder)


def _create_algo_files(input_folder, algo, metadata):

    # model file
    if metadata.get(LOCAL_STATE_DICT_KEY):
        (input_folder / metadata.get(LOCAL_STATE_DICT_KEY)).write_text("True")

    # algo file
    if metadata.get(ALGO_DICT_KEY):
        (input_folder / metadata.get(ALGO_DICT_KEY)).write_text("True")

    # metadata.json file
    (input_folder / METADATA_FILE).write_text(json.dumps(metadata))

    data_operation = algo.train(data_samples=[])

    _create_substra_algo_files(
        remote_struct=data_operation.remote_struct,
        install_libraries=True,
        dependencies=Dependency(local_dependencies=[str(FILE_PATH / "installable_library")], editable_mode=True),
        operation_dir=input_folder,
    )


def test_load_algo(session_dir, fake_compute_plan, dummy_algo_class, caplog):
    """Checks that the load_algo method can load the file given by substrafl to substra
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
    metadata.update({LOCAL_STATE_DICT_KEY: "model"})
    metadata.update({ALGO_DICT_KEY: "algo.tar.gz"})

    _create_algo_files(input_folder, my_algo, metadata)

    caplog.clear()
    my_loaded_algo = load_algo(input_folder)
    assert len(list(filter(lambda x: x.levelname == "WARNING", caplog.records))) == 0
    assert my_loaded_algo._updated


@pytest.mark.parametrize("to_remove", ["algo.tar.gz", METADATA_FILE, "model"])
def test_missing_file_error(session_dir, fake_compute_plan, dummy_algo_class, to_remove):
    """Checks that the load_algo method raises an error if one of the needed file is not found."""
    input_folder = session_dir / str(uuid.uuid4())
    input_folder.mkdir()

    metadata = fake_compute_plan.metadata
    metadata.update({LOCAL_STATE_DICT_KEY: "model"})
    metadata.update({ALGO_DICT_KEY: "algo.tar.gz"})

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


@pytest.fixture
def cyclic_strategy(dummy_strategy_class):
    class Cyclic(dummy_strategy_class):
        def __init__(self) -> None:
            super().__init__()
            self._previous_shared_state = None
            self._previous_local_states = {}

        def perform_round(
            self,
            algo,
            train_data_nodes,
            aggregation_node,
            round_idx,
        ):

            for k, organization in enumerate(train_data_nodes):

                # Local updates from the latest aggregation

                if round_idx == 1 and self._previous_shared_state is not None:
                    useless_local, _ = organization.update_states(
                        algo.train(  # type: ignore
                            organization.data_sample_keys,
                            shared_state=None,
                            _algo_name=f"Training with {algo.__class__.__name__}",
                        ),
                        round_idx=0,
                    )

                    self._previous_local_states[k] = useless_local

                local_state, shared_state = organization.update_states(
                    algo.train(  # type: ignore
                        organization.data_sample_keys,
                        shared_state=self._previous_shared_state,
                        _algo_name=f"Training with {algo.__class__.__name__}",
                    ),
                    local_state=self._previous_local_states.get(k),
                    round_idx=round_idx,
                )
                self._previous_local_states[k] = local_state
                self._previous_shared_state = shared_state

    return Cyclic


@pytest.fixture
def incremental_algo(dummy_algo_class):
    class Incrementalizer(dummy_algo_class):
        def __init__(self) -> None:
            super().__init__()
            self._counter = 0

        @remote_data
        def train(self, x, y, shared_state):

            if shared_state is None:
                shared_state = {"shared": 0}

            self._counter = shared_state["shared"] + 1

            return {"shared": self._counter}

        def load(self, path):
            self._counter = int(np.load(path))
            return self

        def save(self, path):
            np.save(path, self._counter)
            shutil.move(str(path) + ".npy", path)

    return Incrementalizer


@pytest.fixture
def compute_plan(network, cyclic_strategy, incremental_algo, train_linear_nodes, session_dir):
    """For a cyclic like strategy, we use an algo that increment the the `_counter` argument each time it's used
    In deploy mode, for each output algo, this counter is supposed to be:
        number of nodes used * (num rounds - 1) plus the index of the client within the network.clients list
        as the train nodes and the clients are index wised associated
    In local mode, as there is no notion of organization, for a round, the algo with the highest rank is used. Hence
    counter arg will be number of nodes used * num rounds
    """

    num_rounds = 3

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=incremental_algo(),
        strategy=cyclic_strategy(),
        dependencies=Dependency(editable_mode=True),
        train_data_nodes=train_linear_nodes,
        experiment_folder=session_dir,
        num_rounds=num_rounds,
        clean_models=False,
    )

    return compute_plan


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.substra
def test_download_load_model(network, compute_plan, train_linear_nodes, session_dir):
    """Checks that expected local state of downloaded algos."""

    utils.wait(network.clients[0], compute_plan)

    for round in range(1, int(compute_plan.metadata["num_rounds"]) + 1):

        for k, client in enumerate(network.clients):
            model_folder = session_dir / f"model_client_{k}_round_{round}"

            download_algo_files(client, compute_plan.key, model_folder, round_idx=round)
            algo = load_algo(model_folder)

            if client.backend_mode == substra.BackendType.DEPLOYED:
                # We check that we always get the right model based on its _counter arg
                assert algo._counter == len(train_linear_nodes) * (round - 1) + k

            else:
                # in local it should always be the model with the highest rank
                # (cf https://github.com/owkin/connectlib/blob/5d784d7680abf479f1e398183a7eecdcd08fee1b/connectlib/model_loading.py#L170) # noqa: E501
                assert algo._counter == len(train_linear_nodes) * round
