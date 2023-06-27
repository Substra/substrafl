import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import substra
from pydantic import ValidationError

import substrafl
from substrafl.dependency import Dependency
from substrafl.exceptions import InvalidPathError
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote import remote_data
from substrafl.remote.register.register import _create_substra_function_files

CURRENT_FILE = Path(__file__)

# workaround to work with tests/dependency/test_local_dependencies_file_notebook.ipynb
# because we can't import tests in the CI (interfere with substra/tests), and we can't do relative import with nbmake
sys.path.append(str(CURRENT_FILE.parents[1]))

ASSETS_DIR = CURRENT_FILE.parents[1] / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = substra.schemas.Permissions(public=True, authorized_ids=list())


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
    def _register_function(self, my_algo, algo_deps, client, session_dir):
        """Register a train function"""
        data_op = my_algo.train(data_samples=list(), shared_state=None)
        operation_dir = Path(tempfile.mkdtemp(dir=session_dir))
        archive_path, description_path = _create_substra_function_files(
            data_op.remote_struct,
            dependencies=algo_deps,
            install_libraries=client.backend_mode != substra.BackendType.LOCAL_SUBPROCESS,
            operation_dir=operation_dir,
        )
        algo_query = substra.schemas.FunctionSpec(
            name="algo_test_deps",
            inputs=[
                substra.schemas.FunctionInputSpec(
                    identifier=InputIdentifiers.datasamples,
                    kind=substra.schemas.AssetKind.data_sample.value,
                    optional=False,
                    multiple=True,
                ),
                substra.schemas.FunctionInputSpec(
                    identifier=InputIdentifiers.opener,
                    kind=substra.schemas.AssetKind.data_manager.value,
                    optional=False,
                    multiple=False,
                ),
                substra.schemas.FunctionInputSpec(
                    identifier=InputIdentifiers.local,
                    kind=substra.schemas.AssetKind.model.value,
                    optional=True,
                    multiple=False,
                ),
                substra.schemas.FunctionInputSpec(
                    identifier=InputIdentifiers.shared,
                    kind=substra.schemas.AssetKind.model.value,
                    optional=True,
                    multiple=False,
                ),
            ],
            outputs=[
                substra.schemas.FunctionOutputSpec(
                    identifier=OutputIdentifiers.local, kind=substra.schemas.AssetKind.model.value, multiple=False
                ),
                substra.schemas.FunctionOutputSpec(
                    identifier=OutputIdentifiers.shared, kind=substra.schemas.AssetKind.model.value, multiple=False
                ),
            ],
            description=description_path,
            file=archive_path,
            permissions=substra.schemas.Permissions(public=True, authorized_ids=list()),
        )
        function_key = client.add_function(algo_query)
        return function_key

    def _register_train_task(self, function_key, dataset_key, data_sample_key, client):
        """Register a traintask"""
        train_task_query = substra.schemas.TaskSpec(
            function_key=function_key,
            data_manager_key=dataset_key,
            train_data_sample_keys=[data_sample_key],
            inputs=[
                substra.schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=dataset_key),
                substra.schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample_key),
            ],
            outputs={
                OutputIdentifiers.local: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=True, authorized_ids=[])
                ),
                OutputIdentifiers.shared: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=True, authorized_ids=[])
                ),
            },
            worker=client.organization_info().organization_id,
        )
        train_task_key = client.add_task(train_task_query)
        train_task = client.get_task(train_task_key)
        return train_task

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
        function_key = self._register_function(dummy_algo_class(), algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raises=True)

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
                datasamples: np.ndarray,
                shared_state,
            ):
                from local_code_subfolder.local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(datasamples), n_samples=len(datasamples))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raises=True)

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
                datasamples: np.ndarray,
                shared_state,
            ):
                from local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(datasamples), n_samples=len(datasamples))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder" / "local_code.py"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raises=True)

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
                datasamples: np.ndarray,
                shared_state,
            ):
                from local_code_file import combine_strings

                some_strings = combine_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(datasamples), n_samples=len(datasamples))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_file.py"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raises=True)

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
                datasamples: np.ndarray,
                shared_state,
            ):
                # the import is here so that we can run the test without
                # installing it locally
                from substrafltestlibrary import dummy_string_function

                some_strings = dummy_string_function("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(datasamples), n_samples=len(datasamples))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_dependencies=[CURRENT_FILE.parent / pkg_path for pkg_path in pkg_paths],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raises=True)

    @pytest.mark.docker_only
    @patch("substrafl.remote.register.register.local_lib_wheels", MagicMock(return_value="INSTALL IN EDITABLE MODE"))
    def test_force_editable_mode(
        self,
        monkeypatch,
        network,
        session_dir,
        dummy_algo_class,
    ):
        client = network.clients[0]
        algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=False)

        monkeypatch.setenv("SUBSTRA_FORCE_EDITABLE_MODE", str(True))
        self._register_function(dummy_algo_class(), algo_deps, client, session_dir)
        assert substrafl.remote.register.register.local_lib_wheels.call_count == 1

        substrafl.remote.register.register.local_lib_wheels.reset_mock()

        monkeypatch.setenv("SUBSTRA_FORCE_EDITABLE_MODE", str(False))
        self._register_function(dummy_algo_class(), algo_deps, client, session_dir)
        assert substrafl.remote.register.register.local_lib_wheels.call_count == 0
