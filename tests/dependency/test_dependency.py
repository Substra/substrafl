import uuid
from pathlib import Path

import numpy as np
import pytest
import substra
import utils
from local_code_file import combine_strings
from local_code_subfolder.local_code import add_strings
from pydantic import ValidationError

from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.exceptions import InvalidPathException
from connectlib.remote import remote_data
from connectlib.remote.register import create_substra_algo_files

current_file = Path(__file__)

ASSETS_DIR = current_file.parents[1] / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(
    public=True, authorized_ids=list()
)
LOCAL_WORKER_PATH = Path.cwd() / "local-worker"


def test_dependency_validators_file_not_exist():
    with pytest.raises(InvalidPathException):
        # Can't find file.
        Dependency(local_code=[str(uuid.uuid4())])


def test_dependency_validators_not_valid_path():
    with pytest.raises(ValidationError):
        # Can't pass non parsable object.
        Dependency(local_dependencies=[{"a_random_test": 3}])


def test_dependency_validators_no_setup_file():
    with pytest.raises(InvalidPathException):
        # :arg:local_dependencies folders must contain a setup.py.
        Dependency(local_dependencies=[current_file.parent])


@pytest.mark.slow
@pytest.mark.substra
class TestLocalDependency:
    @pytest.fixture(scope="class")
    def dataset_key(self, asset_factory, network):
        """Register a dataset"""
        dataset_query = asset_factory.create_dataset()
        dataset_key = network.clients[0].add_dataset(dataset_query)
        return dataset_key

    @pytest.fixture(scope="class")
    def data_sample_key(self, asset_factory, network, dataset_key):
        """Register a data sample"""
        data_sample = asset_factory.create_data_sample(
            datasets=[dataset_key], test_only=False, content="0,0"
        )
        sample_key = network.clients[0].add_data_sample(data_sample)
        return sample_key

    def _register_algo(self, my_algo, algo_deps, client):
        """Register a composite algo"""
        data_op = my_algo.train(data_samples=list(), shared_state=None, num_updates=4)
        archive_path, description_path = create_substra_algo_files(
            data_op.remote_struct, dependencies=algo_deps
        )
        algo_query = substra.sdk.schemas.AlgoSpec(
            name="algo_test_deps",
            category=substra.sdk.schemas.AlgoCategory.composite,
            description=description_path,
            file=archive_path,
            permissions=substra.sdk.schemas.Permissions(
                public=True, authorized_ids=list()
            ),
        )
        algo_key = client.add_algo(algo_query)
        return algo_key

    def _register_composite(self, algo_key, dataset_key, data_sample_key, client):
        """Register a composite traintuple"""
        composite_traintuple_query = substra.sdk.schemas.CompositeTraintupleSpec(
            algo_key=algo_key,
            data_manager_key=dataset_key,
            train_data_sample_keys=[data_sample_key],
            out_trunk_model_permissions=substra.sdk.schemas.Permissions(
                public=True, authorized_ids=list()
            ),
        )
        composite_key = client.add_composite_traintuple(composite_traintuple_query)
        composite_traintuple = client.get_composite_traintuple(composite_key)
        return composite_traintuple

    def test_pypi_dependency(self, network, dataset_key, data_sample_key):
        """Test that dependencies from PyPi are installed."""

        class MyAlgo(Algo):
            # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
            # pytest calls it)
            def delayed_init(self, seed: int, *args, **kwargs):
                pass

            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                num_updates: int,
                shared_state,
            ):
                x = [4]
                return dict(test=np.array(x), n_samples=len(x))

            @remote_data
            def predict(self, x: np.array, shared_state):
                return shared_state["test"]

            def load(self, path: Path):
                return self

            def save(self, path: Path):
                assert path.parent.exists()
                with path.open("w") as f:
                    f.write("test")

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
        )
        algo_key = self._register_algo(my_algo, algo_deps, client)

        composite_traintuple = self._register_composite(
            algo_key, dataset_key, data_sample_key, client
        )
        utils.wait(client, composite_traintuple)

    def test_local_dependencies_directory(self, network, dataset_key, data_sample_key):
        """Test that you can import a directory"""

        class MyAlgo(Algo):
            # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
            # pytest calls it)
            def delayed_init(self, seed: int, *args, **kwargs):
                pass

            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                num_updates: int,
                shared_state,
            ):

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

            @remote_data
            def predict(self, x: np.array, shared_state):
                return shared_state["test"]

            def load(self, path: Path):
                return self

            def save(self, path: Path):
                assert path.parent.exists()
                with path.open("w") as f:
                    f.write("test")

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[current_file.parent / "local_code_subfolder"],
        )
        algo_key = self._register_algo(my_algo, algo_deps, client)

        composite_traintuple = self._register_composite(
            algo_key, dataset_key, data_sample_key, client
        )
        utils.wait(client, composite_traintuple)

    def test_local_dependencies_file(self, network, dataset_key, data_sample_key):
        """Test that you can import a file"""

        class MyAlgo(Algo):
            # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
            # pytest calls it)
            def delayed_init(self, seed: int, *args, **kwargs):
                pass

            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                num_updates: int,
                shared_state,
            ):

                some_strings = combine_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

            @remote_data
            def predict(self, x: np.array, shared_state):
                return shared_state["test"]

            def load(self, path: Path):
                return self

            def save(self, path: Path):
                assert path.parent.exists()
                with path.open("w") as f:
                    f.write("test")

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[current_file.parent / "local_code_file.py"],
        )
        algo_key = self._register_algo(my_algo, algo_deps, client)

        composite_traintuple = self._register_composite(
            algo_key, dataset_key, data_sample_key, client
        )
        utils.wait(client, composite_traintuple)

    @pytest.mark.docker_only
    def test_local_dependencies_installable_library(
        self, network, dataset_key, data_sample_key
    ):
        """Test that you can install a local library
        Automatically done in docker but need to be manually done if force in subprocess mode
        """

        class MyAlgo(Algo):
            # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
            # pytest calls it)
            def delayed_init(self, seed: int, *args, **kwargs):
                pass

            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                num_updates: int,
                shared_state,
            ):
                # the import is here so that we can run the test without
                # installing it locally
                from connectlibtestlibrary import dummy_string_function

                some_strings = dummy_string_function("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

            @remote_data
            def predict(self, x: np.array, shared_state):
                return shared_state["test"]

            def load(self, path: Path):
                return self

            def save(self, path: Path):
                assert path.parent.exists()
                with path.open("w") as f:
                    f.write("test")

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_dependencies=[Path(__file__).parent / "installable_library"],
        )
        algo_key = self._register_algo(my_algo, algo_deps, client)

        composite_traintuple = self._register_composite(
            algo_key, dataset_key, data_sample_key, client
        )
        utils.wait(client, composite_traintuple)
