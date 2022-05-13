# Copyright 2018 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import shutil
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from substra.sdk.schemas import Permissions

from connectlib.algorithms.algo import Algo
from connectlib.nodes.aggregation_node import AggregationNode
from connectlib.nodes.test_data_node import TestDataNode
from connectlib.nodes.train_data_node import TrainDataNode
from connectlib.remote.decorators import remote_data
from connectlib.schemas import StrategyName
from connectlib.strategies.strategy import Strategy

from . import assets_factory
from . import settings

LINEAR_N_COL = 2
LINEAR_N_TARGET = 1


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote."""
    parser.addoption(
        "--local",
        action="store_true",
        help="Run the tests on the local backend only (debug mode). "
        "Otherwise run the tests only on the remote backend.",
    )
    parser.addoption(
        "--ci",
        action="store_true",
        help="Run the tests on the backend deployed by connect-test nightly (remote mode). "
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
def network(request):
    """Network fixture. Create network instance from the configuration files and the options
    passed as arguments to pytest.

    Network must be started outside of the tests environment and the network is kept
    alive while running all tests.

    if --local is passed, the session will be started in debug mode and all the clients will be duplicated.

    Args:
        network_cfg: Network configuration.

    Returns:
        Network: All the elements needed to interact with the :term:`Connect` platform.
    """
    is_local = request.config.getoption("--local")
    is_ci = request.config.getoption("--ci")

    network = settings.local_network() if is_local else settings.remote_network(is_ci=is_ci)
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
    key = assets_factory.add_python_metric(
        python_formula="abs(y_pred-y_true).mean()",
        name="MAE",
        permissions=default_permissions,
        client=network.clients[0],
        tmp_folder=session_dir,
    )

    return key


@pytest.fixture(scope="session")
def numpy_datasets(network, session_dir, default_permissions):
    """Create and add to the first node of the network an opener that will
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
        datasets_permissions=[default_permissions] * network.n_nodes,
        clients=network.clients,
        msp_ids=network.msp_ids,
        tmp_folder=session_dir,
    )

    return dataset_key


@pytest.fixture(scope="session")
def constant_samples(network, numpy_datasets, session_dir):
    """0s and 1s data samples for clients 0 and 1.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each node.
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
        List[np.ndarray]: A list of linear data for each node of the network.
    """
    return [
        assets_factory.linear_data(
            n_col=LINEAR_N_COL + LINEAR_N_TARGET,
            n_samples=1024,
            weights_seed=42,
            noise_seed=i,
        )
        for i in range(network.n_nodes)
    ]


@pytest.fixture(scope="session")
def train_linear_nodes(network, numpy_datasets, train_linear_data_samples, session_dir):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each node.
        train_linear_data_samples (List[np.ndarray]): A List of linear linked data.
        session_dir (Path): A temp file created for the pytest session.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every node.
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
        for k in range(network.n_nodes)
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
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each node.
        mae (str): Mean absolute error metric for the TestDataNode
        session_dir (Path): A temp file created for the pytest session.

    Returns:
        List[TestDataNode]: A one element list containing a connectlib TestDataNode for linear data with a mae metric.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every node.
        contents=test_linear_data_samples,
        dataset_keys=[numpy_datasets[0]],
        clients=[network.clients[0]],
        tmp_folder=session_dir,
    )

    test_data_nodes = [TestDataNode(network.msp_ids[0], numpy_datasets[0], linear_samples, metric_keys=[mae])]

    return test_data_nodes


@pytest.fixture(scope="session")
def aggregation_node(network):
    """The central node to use.

    Args:
        network (Network): Substra network from the configuration file.

    Returns:
        AggregationNode: Connectlib aggregation Node.
    """
    return AggregationNode(network.msp_ids[0])


@pytest.fixture(scope="session")
def torch_linear_model():
    """Generates a basic torch model (Perceptron). This model can be trained on the linear data fixtures
    as its number of input nodes is set per the LINEAR_N_COL variable which is also used to generates the linear
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
        def train(self, x, y, shared_state):
            return dict(test=np.array([4]), x=x, y=y, shared_state=shared_state)

        @remote_data
        def predict(self, x: np.array, shared_state):
            return dict(x=x, shared_state=shared_state)

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    return DummyAlgo
