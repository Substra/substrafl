import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from substra import BackendType
from substra.sdk.schemas import AlgoCategory
from substra.sdk.schemas import AlgoSpec
from substra.sdk.schemas import CompositeTraintupleSpec
from substra.sdk.schemas import ComputeTaskOutput
from substra.sdk.schemas import InputRef
from substra.sdk.schemas import Permissions

from substrafl.dependency import Dependency
from substrafl.exceptions import InvalidPathError
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote import remote_data
from substrafl.remote.register.register import _create_substra_algo_files

CURRENT_FILE = Path(__file__)

# workaround to work with tests/dependency/test_local_dependencies_file_notebook.ipynb
# because we can't import tests in the CI (interfere with substra/tests), and we can't do relative import with nbmake
sys.path.append(str(CURRENT_FILE.parents[1]))
import utils  # noqa: E402

ASSETS_DIR = CURRENT_FILE.parents[1] / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = Permissions(public=True, authorized_ids=list())
LOCAL_WORKER_PATH = Path.cwd() / "local-worker"


def test_dependency_validators_file_not_exist():
    with pytest.raises(InvalidPathError):
        # Can't find file.
        Dependency(local_code=[str(uuid.uuid4())], editable_mode=True)


def test_dependency_validators_not_valid_path():
    with pytest.raises(ValidationError):
        # Can't pass non parsable object.
        Dependency(local_dependencies=[{"a_random_test": 3}], editable_mode=True)


def test_dependency_validators_no_setup_file():
    with pytest.raises(InvalidPathError):
        # :arg:local_dependencies folders must contain a setup.py.
        Dependency(local_dependencies=[CURRENT_FILE.parent], editable_mode=True)


@pytest.mark.slow
@pytest.mark.substra
class TestLocalDependency:
    def _register_algo(self, my_algo, algo_deps, client, session_dir):
        """Register a composite algo"""
        data_op = my_algo.train(data_samples=list(), shared_state=None)
        operation_dir = Path(tempfile.mkdtemp(dir=session_dir))
        archive_path, description_path = _create_substra_algo_files(
            data_op.remote_struct,
            dependencies=algo_deps,
            install_libraries=client.backend_mode != BackendType.LOCAL_SUBPROCESS,
            operation_dir=operation_dir,
        )
        algo_query = AlgoSpec(
            name="algo_test_deps",
            category=AlgoCategory.composite,
            description=description_path,
            file=archive_path,
            permissions=Permissions(public=True, authorized_ids=list()),
        )
        algo_key = client.add_algo(algo_query)
        return algo_key

    def _register_composite(self, algo_key, dataset_key, data_sample_key, client):
        """Register a composite traintuple"""
        composite_traintuple_query = CompositeTraintupleSpec(
            algo_key=algo_key,
            data_manager_key=dataset_key,
            train_data_sample_keys=[data_sample_key],
            inputs=[
                InputRef(identifier=InputIdentifiers.OPENER, asset_key=dataset_key),
                InputRef(identifier=InputIdentifiers.DATASAMPLES, asset_key=data_sample_key),
            ],
            outputs={
                OutputIdentifiers.SHARED: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
                OutputIdentifiers.LOCAL: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
            },
        )
        composite_key = client.add_composite_traintuple(composite_traintuple_query)
        composite_traintuple = client.get_composite_traintuple(composite_key)
        return composite_traintuple

    def test_pypi_dependency(
        self,
        network,
        numpy_datasets,
        constant_samples,
        session_dir,
        dummy_algo_class,
    ):
        """Test that dependencies from PyPi are installed."""

        client = network.clients[0]
        algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
        algo_key = self._register_algo(dummy_algo_class(), algo_deps, client, session_dir)

        composite_traintuple = self._register_composite(algo_key, numpy_datasets[0], constant_samples[0], client)
        utils.wait(client, composite_traintuple)

    def test_local_dependencies_directory(
        self,
        network,
        numpy_datasets,
        constant_samples,
        session_dir,
        dummy_algo_class,
    ):
        """Test that you can import a directory"""

        class MyAlgo(dummy_algo_class):
            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                shared_state,
            ):
                from local_code_subfolder.local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder"],
            editable_mode=True,
        )
        algo_key = self._register_algo(my_algo, algo_deps, client, session_dir)

        composite_traintuple = self._register_composite(algo_key, numpy_datasets[0], constant_samples[0], client)
        utils.wait(client, composite_traintuple)

    def test_local_dependencies_file_in_directory(
        self,
        network,
        numpy_datasets,
        constant_samples,
        session_dir,
        dummy_algo_class,
    ):
        """Test that you can import a file that is in a subdirectory"""

        class MyAlgo(dummy_algo_class):
            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                shared_state,
            ):
                from local_code_subfolder.local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder" / "local_code.py"],
            editable_mode=True,
        )
        algo_key = self._register_algo(my_algo, algo_deps, client, session_dir)

        composite_traintuple = self._register_composite(algo_key, numpy_datasets[0], constant_samples[0], client)
        utils.wait(client, composite_traintuple)

    def test_local_dependencies_file(
        self,
        network,
        numpy_datasets,
        constant_samples,
        session_dir,
        dummy_algo_class,
    ):
        """Test that you can import a file"""

        class MyAlgo(dummy_algo_class):
            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                shared_state,
            ):
                from local_code_file import combine_strings

                some_strings = combine_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_file.py"],
            editable_mode=True,
        )
        algo_key = self._register_algo(my_algo, algo_deps, client, session_dir)

        composite_traintuple = self._register_composite(algo_key, numpy_datasets[0], constant_samples[0], client)
        utils.wait(client, composite_traintuple)

    @pytest.mark.docker_only
    @pytest.mark.parametrize(
        "pkg_paths",
        [["installable_library"], ["poetry_installable_library"], ["installable_library", "installable_library2"]],
    )
    def test_local_dependencies_installable_library(
        self,
        network,
        numpy_datasets,
        constant_samples,
        pkg_paths,
        session_dir,
        dummy_algo_class,
    ):
        """Test that you can install a local library
        Automatically done in docker but need to be manually done if force in subprocess mode
        """

        class MyAlgo(dummy_algo_class):
            @remote_data
            def train(
                self,
                x: np.ndarray,
                y: np.ndarray,
                shared_state,
            ):
                # the import is here so that we can run the test without
                # installing it locally
                from substrafltestlibrary import dummy_string_function

                some_strings = dummy_string_function("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(x), n_samples=len(x))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_dependencies=[CURRENT_FILE.parent / pkg_path for pkg_path in pkg_paths],
            editable_mode=True,
        )
        algo_key = self._register_algo(my_algo, algo_deps, client, session_dir)

        composite_traintuple = self._register_composite(algo_key, numpy_datasets[0], constant_samples[0], client)
        utils.wait(client, composite_traintuple)
