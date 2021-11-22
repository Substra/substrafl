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


import pytest
import settings
from sdk import data_factory


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote."""
    parser.addoption(
        "--local",
        action="store_true",
        help="Run the tests on the local backend only (debug mode). "
        "Otherwise run the tests only on the remote backend.",
    )


@pytest.fixture(scope="session")
def network_cfg(request):
    """Network configuration fixture.
    Loads the appropriate backend configuration based on the options passed to pytest.

    Args:
        request: Pytest cli request.

    Returns:
        settings.Settings: The entire :term:`Connect` network configuration.
    """
    local = request.config.getoption("--local")

    return settings.load_backend_config(debug=bool(local))


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
    local = request.config.getoption("--local")

    network = settings.local_network() if local else settings.remote_network()
    return network


# TODO : the entire way of creating and using assets for the test needs to be redefine
# This will be done in an other PR
@pytest.fixture
def dataset_query(tmpdir):
    opener_path = tmpdir / "opener.py"
    opener_path.write_text("raise ValueError()", encoding="utf-8")

    desc_path = tmpdir / "description.md"
    desc_path.write_text("#Hello world", encoding="utf-8")

    return {
        "name": "dataset_name",
        "data_opener": str(opener_path),
        "type": "images",
        "description": str(desc_path),
        "metric_key": "",
        "permissions": {
            "public": True,
            "authorized_ids": [],
        },
    }


@pytest.fixture
def metric_query(tmpdir):
    metrics_path = tmpdir / "metrics.zip"
    metrics_path.write_text("foo archive", encoding="utf-8")

    desc_path = tmpdir / "description.md"
    desc_path.write_text("#Hello world", encoding="utf-8")

    return {
        "name": "metrics_name",
        "metrics": str(metrics_path),
        "metrics_name": "name of the metrics",
        "description": str(desc_path),
        "test_data_manager_key": None,
        "test_data_sample_keys": [],
        "permissions": {
            "public": True,
            "authorized_ids": [],
        },
    }


@pytest.fixture
def data_sample_query(tmpdir):
    data_sample_dir_path = tmpdir / "data_sample_0"
    data_sample_file_path = data_sample_dir_path / "data.txt"
    data_sample_file_path.write_text("Hello world 0", encoding="utf-8", ensure=True)

    return {
        "path": str(data_sample_dir_path),
        "data_manager_keys": ["42"],
        "test_only": False,
    }


@pytest.fixture
def data_samples_query(tmpdir):
    nb = 3
    paths = []
    for i in range(nb):
        data_sample_dir_path = tmpdir / f"data_sample_{i}"
        data_sample_file_path = data_sample_dir_path / "data.txt"
        data_sample_file_path.write_text(
            f"Hello world {i}", encoding="utf-8", ensure=True
        )

        paths.append(str(data_sample_dir_path))

    return {
        "paths": paths,
        "data_manager_keys": ["42"],
        "test_only": False,
    }


@pytest.fixture(scope="session")
def asset_factory():
    return data_factory.AssetsFactory("test_debug")


@pytest.fixture()
def data_sample(asset_factory):
    return asset_factory.create_data_sample()
