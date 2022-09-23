import multiprocessing
import shutil
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pytest
import substra
import torch
import torch.nn.functional as functional
from substra.sdk.schemas import Permissions

from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote.decorators import remote_data
from substrafl.schemas import StrategyName
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
        "--ci",
        action="store_true",
        help="Run the tests on the backend deployed by substra-test nightly (remote mode). "
        "Otherwise run the tests only on the default remote backend.",
    )


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_variable():
    # https://github.com/pytest-dev/pytest-flask/issues/104
    # necessary on OS X to run multiprocessing
    multiprocessing.set_start_method("fork")


@pytest.fixture(scope="session")
def session_dir():
    temp_dir = Path.cwd() / "local-assets-cl"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def predictions_path(session_dir):
    return session_dir / "predictions"


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
def mae(network, default_permissions, session_dir):
    class CustomMetric:
        def __init__(self, python_formula, name) -> None:
            self.name = name
            self.python_formula = python_formula
            self.key = None

        def compute(self, y_true, y_pred):
            return eval(self.python_formula)

        def add_to_substra(self, permissions, client, tmp_folder):
            key = assets_factory.add_python_metric(
                python_formula=self.python_formula,
                name=self.name,
                permissions=permissions,
                client=client,
                tmp_folder=tmp_folder,
            )
            self.key = key

    mae = CustomMetric(python_formula="abs(y_pred-y_true).mean()", name="MAE")
    mae.add_to_substra(permissions=default_permissions, client=network.clients[0], tmp_folder=session_dir)

    return mae


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
    mae,
    test_linear_data_samples,
    session_dir,
):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        mae (str): Mean absolute error metric for the TestDataNode
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

    test_data_nodes = [TestDataNode(network.msp_ids[0], numpy_datasets[0], linear_samples, metric_keys=[mae.key])]

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
def batch_norm_cnn():
    """Generates a CNN model with 1d an 2d batch normalization layers

    Returns:
        torch.nn.Module: A torch CNN
    """

    class BatchNormCnn(torch.nn.Module):
        def __init__(self):
            super(BatchNormCnn, self).__init__()
            torch.manual_seed(42)
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_bn = torch.nn.BatchNorm2d(20)
            self.dense1 = torch.nn.Linear(in_features=320, out_features=50)
            self.dense1_bn = torch.nn.BatchNorm1d(50)
            self.dense2 = torch.nn.Linear(50, 1)

        def forward(self, x):
            x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
            x = functional.relu(functional.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
            x = x.view(-1, 320)  # reshape
            x = functional.relu(self.dense1_bn(self.dense1(x)))
            x = functional.relu(self.dense2(x))
            return functional.sigmoid(x)

    return BatchNormCnn


@pytest.fixture(scope="session")
def rtol():
    """
    relative tolerance for pytest.approx()

    Returns:
        float: rtol
    """
    return 10e-6


@pytest.fixture
def dummy_strategy_class():
    class DummyStrategy(Strategy):
        @property
        def name(self) -> StrategyName:
            return "dummy"

        def perform_round(
            self,
            algo: Algo,
            train_data_nodes: List[TrainDataNode],
            aggregation_node: Optional[AggregationNode],
            round_idx: int,
            clean_models: bool,
        ):
            pass

        def predict(
            self,
            algo: Algo,
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
        def train(self, datasamples, shared_state):
            return dict(test=np.array([4]), datasamples=datasamples, shared_state=shared_state)

        @remote_data
        def predict(self, datasamples: np.array, shared_state):
            return dict(datasamples=datasamples, shared_state=shared_state)

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    return DummyAlgo


@pytest.fixture(scope="session")
def numpy_torch_dataset():
    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, datasamples, is_inference=False):
            self.x = datasamples[0]
            self.y = datasamples[1]
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
