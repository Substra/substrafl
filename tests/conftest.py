import multiprocessing
import shutil
import sys
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pytest
import substra
import torch
from substra.sdk.schemas import Permissions

import docker
from substrafl.algorithms.algo import Algo
from substrafl.dependency import Dependency
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote.decorators import remote_data
from substrafl.strategies.schemas import StrategyName
from substrafl.strategies.strategy import Strategy

from . import assets_factory
from . import settings

LINEAR_N_COL = 2
LINEAR_N_TARGET = 1


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote."""
    parser.addoption(
        "--mode",
        choices=["subprocess", "docker", "remote"],
        default="remote",
        help="Choose the mode on which to run the tests",
    )
    parser.addoption(
        "--prune-docker",
        action="store_true",
        help="Prune ALL docker images if set. Will be considered only if mode set to docker",
    )
    parser.addoption(
        "--ci",
        action="store_true",
        help="Run the tests on the backend deployed by substra-test nightly (remote mode). "
        "Otherwise run the tests only on the default remote backend.",
    )


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_variable():
    # https://github.com/pytest-dev/pytest-flask/issues/104
    # necessary on OS X to run multiprocessing
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn")
    else:
        multiprocessing.set_start_method("fork")


def get_docker_client():
    try:
        docker_client = docker.from_env()
        return docker_client
    except docker.errors.DockerException as e:
        raise ConnectionError(
            "Couldn't get the Docker client from environment variables. "
            "Is your Docker server running ?\n"
            "Docker error : {0}".format(e)
        )


@pytest.fixture(scope="module", autouse=True)
def prune_docker_image(request):
    yield
    backend_type = substra.BackendType(request.config.getoption("--mode"))
    prune_docker = request.config.getoption("--prune-docker")

    if backend_type == substra.BackendType.LOCAL_DOCKER and prune_docker:
        docker_client = get_docker_client()
        docker_client.containers.prune()
        docker_client.images.prune(filters={"dangling": False})


@pytest.fixture(scope="session")
def session_dir():
    temp_dir = Path.cwd() / "local-assets-cl"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def network(request):
    """Network fixture. Create network instance from the configuration files and the options
    passed as arguments to pytest.

    Network must be started outside of the tests environment and the network is kept
    alive while running all tests.

    if mode is subprocess or docker, the session is be started in local mode and all the clients will be
        duplicated.

    Args:
        network_cfg: Network configuration.

    Returns:
        Network: All the elements needed to interact with the :term:`Substra` platform.
    """
    backend_type = substra.BackendType(request.config.getoption("--mode"))
    is_ci = request.config.getoption("--ci")

    if backend_type != substra.BackendType.REMOTE and is_ci:
        raise pytest.UsageError("--ci can only be used with a remote backend")

    network = settings.network(backend_type=backend_type, is_ci=is_ci)
    return network


@pytest.fixture(scope="session")
def default_permissions() -> Permissions:
    """Default permissions fixture. Those are needed to add any asset to substra.

    Returns:
        Permissions: Public permissions.
    """
    return Permissions(public=True, authorized_ids=[])


@pytest.fixture(scope="session")
def mae():
    return lambda y_pred, y_true: abs(y_pred - y_true).mean()


@pytest.fixture(scope="session")
def mae_metric(mae):
    def mae_score(data_from_opener, predictions):
        y_pred = np.array(predictions)
        y_true = data_from_opener[1]
        return mae(y_pred, y_true)

    return mae_score


@pytest.fixture(scope="session")
def numpy_datasets(network, session_dir, default_permissions):
    """Create and add to the first organization of the network an opener that will
    load and save data, load and save numpy predictions.

    Args:
        network (Network): The defined substra network by the config files.
        session_dir (Path): A temp file created for the pytest session.
        default_permissions (Permissions): Default permissions for all of the assets
        of the session.

    Return:
        str: The dataset key returned by substra.
    """

    dataset_key = assets_factory.add_numpy_datasets(
        datasets_permissions=[default_permissions] * network.n_organizations,
        clients=network.clients,
        tmp_folder=session_dir,
    )

    return dataset_key


@pytest.fixture(scope="session")
def constant_samples(network, numpy_datasets, session_dir):
    """0s and 1s data samples for clients 0 and 1.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        session_dir (Path): A temp file created for the pytest session.
    """

    key = assets_factory.add_numpy_samples(
        contents=[np.zeros((1, 2)), np.ones((1, 2))],
        dataset_keys=numpy_datasets,
        tmp_folder=session_dir,
        clients=network.clients,
    )

    return key


@pytest.fixture(scope="session")
def train_linear_data_samples(network):
    """Generates linear linked data for training purposes. The train_linear_data_samples data and
    test_linear_data_samples data are linked with the same weights as they fixed per the same seed.

    Args:
        network (Network): Substra network from the configuration file.

    Returns:
        List[np.ndarray]: A list of linear data for each organization of the network.
    """
    return [
        assets_factory.linear_data(
            n_col=LINEAR_N_COL + LINEAR_N_TARGET,
            n_samples=1024,
            weights_seed=42,
            noise_seed=i,
        )
        for i in range(network.n_organizations)
    ]


@pytest.fixture(scope="session")
def train_linear_nodes(network, numpy_datasets, train_linear_data_samples, session_dir):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        train_linear_data_samples (List[np.ndarray]): A List of linear linked data.
        session_dir (Path): A temp file created for the pytest session.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every organization.
        contents=train_linear_data_samples,
        dataset_keys=numpy_datasets,
        clients=network.clients,
        tmp_folder=session_dir,
    )

    train_data_nodes = [
        TrainDataNode(
            network.msp_ids[k],
            numpy_datasets[k],
            [linear_samples[k]],
        )
        for k in range(network.n_organizations)
    ]

    return train_data_nodes


@pytest.fixture(scope="session")
def test_linear_data_samples():
    """Generates linear linked data for testing purposes. The train_linear_data_samples data and
    test_linear_data_samples data are linked with the same weights as they fixed per the same seed.

    Returns:
        List[np.ndarray]: A one element list containing linear linked data.
    """
    return [
        assets_factory.linear_data(
            n_col=LINEAR_N_COL + LINEAR_N_TARGET,
            n_samples=64,
            weights_seed=42,
            noise_seed=42,
        )
    ]


@pytest.fixture(scope="session")
def test_linear_nodes(
    network,
    numpy_datasets,
    test_linear_data_samples,
    session_dir,
):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        test_linear_data_samples (np.array): linear data.
        session_dir (Path): A temp file created for the pytest session.

    Returns:
        List[TestDataNode]: A one element list containing a substrafl TestDataNode for linear data with
        a mae metric.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every organization.
        contents=test_linear_data_samples,
        dataset_keys=[numpy_datasets[0]],
        clients=[network.clients[0]],
        tmp_folder=session_dir,
    )

    test_data_nodes = [TestDataNode(network.msp_ids[0], numpy_datasets[0], linear_samples)]

    return test_data_nodes


@pytest.fixture(scope="session")
def aggregation_node(network):
    """The central organization to use.

    Args:
        network (Network): Substra network from the configuration file.

    Returns:
        AggregationNode: Substrafl aggregation Node.
    """
    return AggregationNode(network.msp_ids[0])


@pytest.fixture(scope="session")
def torch_cpu_dependency():
    return Dependency(
        pypi_dependencies=[
            "torch==2.4.1",
            "numpy==2.1.1",
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        ],
        editable_mode=True,
    )


@pytest.fixture(scope="session")
def torch_linear_model():
    """Generates a basic torch model (Perceptron). This model can be trained on the linear data fixtures
    as its number of input organizations is set per the LINEAR_N_COL variable which is also used to generates the linear
    train and test data samples.

    Returns:
        torch.nn.Module: A torch perceptron trainable on the linear data.
    """

    class Perceptron(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(LINEAR_N_COL, LINEAR_N_TARGET)

        def forward(self, x):
            out = self.linear1(x)
            return out

    return Perceptron


@pytest.fixture(scope="session")
def rtol():
    """
    relative tolerance for pytest.approx()

    Returns:
        float: rtol
    """
    return 1e-5


@pytest.fixture(scope="session")
def seed():
    """
    Seed to apply.

    Returns:
        int: seed
    """
    return 42


@pytest.fixture
def dummy_strategy_class():
    class DummyStrategy(Strategy):
        @property
        def name(self) -> StrategyName:
            return "dummy"

        def perform_round(
            self,
            train_data_nodes: List[TrainDataNode],
            aggregation_node: Optional[AggregationNode],
            round_idx: int,
            clean_models: bool,
            additional_orgs_permissions: Optional[set] = None,
        ):
            pass

        def perform_evaluation(
            self,
            test_data_nodes: List[TestDataNode],
            train_data_nodes: List[TrainDataNode],
            round_idx: int,
        ):
            pass

    return DummyStrategy


@pytest.fixture
def dummy_algo_class():
    class DummyAlgo(Algo):
        @property
        def strategies(self) -> List[StrategyName]:
            # compatible with all strategies and the dummy one
            return list(StrategyName) + ["dummy"]

        @property
        def model(self):
            return "model"

        @remote_data
        def train(self, data_from_opener, shared_state):
            return dict(test=np.array([4]), data_from_opener=data_from_opener, shared_state=shared_state)

        def predict(self, data_from_opener: np.array, shared_state):
            return dict(data_from_opener=data_from_opener, shared_state=shared_state)

        def load_local_state(self, path: Path):
            return self

        def save_local_state(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    return DummyAlgo


@pytest.fixture(scope="session")
def numpy_torch_dataset():
    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, data_from_opener, is_inference=False):
            self.x = data_from_opener[0]
            self.y = data_from_opener[1]
            self.is_inference = is_inference

        def __getitem__(self, index):
            x = torch.from_numpy(self.x[index]).float()
            if not self.is_inference:
                y = torch.from_numpy(self.y[index]).float()
                return x, y
            else:
                return x

        def __len__(self):
            return len(self.x)

    return TorchDataset


SETUP_CONTENT = """from setuptools import setup, find_packages

setup(
    name='mymodule',
    version='1.0.2',
    author='Author Name',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 3.5.1'],
)"""


@pytest.fixture
def local_installable_module():
    def _local_installable_module(root_dir):
        module_root = root_dir / "my_module"
        module_root.mkdir()
        setup_file = module_root / "setup.py"
        setup_file.write_text(SETUP_CONTENT)
        return module_root

    return _local_installable_module
