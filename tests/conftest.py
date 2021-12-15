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
import shutil
from pathlib import Path

import numpy as np
import pytest
from substra.sdk.schemas import Permissions

from . import assets_factory, settings


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote."""
    parser.addoption(
        "--local",
        action="store_true",
        help="Run the tests on the local backend only (debug mode). "
        "Otherwise run the tests only on the remote backend.",
    )


@pytest.fixture(scope="session")
def session_dir():
    temp_dir = Path.cwd() / "local-assets-cl"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def network_cfg(request):
    """Network configuration fixture.
    Loads the appropriate backend configuration based on the options passed to pytest.

    Args:
        request: Pytest cli request.

    Returns:
        settings.Settings: The entire :term:`Connect` network configuration.
    """
    is_local = settings.is_local_mode(request)

    return settings.load_backend_config(debug=is_local)


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
    is_local = settings.is_local_mode(request)

    network = settings.local_network() if is_local else settings.remote_network()
    return network


@pytest.fixture(scope="session")
def default_permissions() -> Permissions:
    """Default permissions fixture. Those are needed to add any asset to substra.

    Returns:
        Permissions: Public permissions.
    """
    return Permissions(public=True, authorized_ids=[])


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
