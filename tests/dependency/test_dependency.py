import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import substra
from pydantic import ValidationError

import substrafl
from substrafl.constants import SUBSTRAFL_FOLDER
from substrafl.dependency import Dependency
from substrafl.exceptions import InvalidPathError
from substrafl.nodes.schemas import InputIdentifiers
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.remote import remote_data
from substrafl.remote.register.register import register_function

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
        Dependency(local_installable_dependencies=[{"a_random_test": 3}], editable_mode=True)


def test_dependency_validators_no_setup_file():
    with pytest.raises(InvalidPathError):
        # :arg:local_installable_dependencies folders must contain a setup.py.
        Dependency(local_installable_dependencies=[CURRENT_FILE.parent], editable_mode=True)


@pytest.mark.slow
@pytest.mark.substra
class TestLocalDependency:
    def _register_function(self, my_algo, algo_deps, client, session_dir):
        """Register a train function"""
        data_op = my_algo.train(data_samples=list(), shared_state=None)

        inputs = [
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
        ]

        outputs = [
            substra.schemas.FunctionOutputSpec(
                identifier=OutputIdentifiers.local, kind=substra.schemas.AssetKind.model.value, multiple=False
            ),
            substra.schemas.FunctionOutputSpec(
                identifier=OutputIdentifiers.shared, kind=substra.schemas.AssetKind.model.value, multiple=False
            ),
        ]
        permissions = substra.schemas.Permissions(public=True, authorized_ids=list())

        function_key = register_function(
            client=client,
            remote_struct=data_op.remote_struct,
            permissions=permissions,
            inputs=inputs,
            outputs=outputs,
            dependencies=algo_deps,
        )

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
        client.wait_task(train_task.key, raise_on_failure=True)

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
                data_from_opener: np.ndarray,
                shared_state,
            ):
                from local_code_subfolder.local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(data_from_opener), n_samples=len(data_from_opener))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raise_on_failure=True)

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
                data_from_opener: np.ndarray,
                shared_state,
            ):
                from local_code import add_strings

                some_strings = add_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(data_from_opener), n_samples=len(data_from_opener))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_subfolder" / "local_code.py"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raise_on_failure=True)

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
                data_from_opener: np.ndarray,
                shared_state,
            ):
                from local_code_file import combine_strings

                some_strings = combine_strings("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(data_from_opener), n_samples=len(data_from_opener))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_code=[CURRENT_FILE.parent / "local_code_file.py"],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raise_on_failure=True)

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
                data_from_opener: np.ndarray,
                shared_state,
            ):
                # the import is here so that we can run the test without
                # installing it locally
                from substrafltestlibrary import dummy_string_function

                some_strings = dummy_string_function("Foo", "Bar")
                assert some_strings == "FooBar"  # For flake8 purposes

                return dict(test=np.array(data_from_opener), n_samples=len(data_from_opener))

        client = network.clients[0]
        my_algo = MyAlgo()
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            local_installable_dependencies=[CURRENT_FILE.parent / pkg_path for pkg_path in pkg_paths],
            editable_mode=True,
        )
        function_key = self._register_function(my_algo, algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raise_on_failure=True)

    @pytest.mark.docker_only
    def test_binary_dependencies(
        self,
        network,
        numpy_datasets,
        constant_samples,
        session_dir,
        dummy_algo_class,
    ):
        """Test that you can install binary dependencies"""

        client = network.clients[0]
        algo_deps = Dependency(
            pypi_dependencies=["pytest"],
            binary_dependencies=["python3-dev"],
            editable_mode=True,
        )
        function_key = self._register_function(dummy_algo_class(), algo_deps, client, session_dir)

        train_task = self._register_train_task(function_key, numpy_datasets[0], constant_samples[0], client)
        client.wait_task(train_task.key, raise_on_failure=True)

    @pytest.mark.docker_only
    def test_force_editable_mode(
        self,
        mocker,
        monkeypatch,
        network,
        session_dir,
        dummy_algo_class,
    ):
        mocker.patch(
            "substrafl.dependency.manage_dependencies.local_lib_wheels", return_value=["INSTALL IN EDITABLE MODE"]
        )
        mocker.patch("substrafl.dependency.manage_dependencies.compile_requirements")

        client = network.clients[0]
        monkeypatch.setenv("SUBSTRA_FORCE_EDITABLE_MODE", str(True))

        algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=False)

        self._register_function(dummy_algo_class(), algo_deps, client, session_dir)
        assert substrafl.dependency.manage_dependencies.local_lib_wheels.call_count == 1

        substrafl.dependency.manage_dependencies.local_lib_wheels.reset_mock()

        monkeypatch.setenv("SUBSTRA_FORCE_EDITABLE_MODE", str(False))
        self._register_function(dummy_algo_class(), algo_deps, client, session_dir)
        assert substrafl.dependency.manage_dependencies.local_lib_wheels.call_count == 0


def test_get_compute():
    dependency = Dependency(
        pypi_dependencies=["pytest"],
        local_installable_dependencies=[CURRENT_FILE.parent / "installable_library"],
        local_code=[CURRENT_FILE.parent / "local_code_file.py"],
        editable_mode=True,
    )

    cache_dir = dependency.cache_directory

    assert (cache_dir / "local_code_file.py").is_file()
    assert (cache_dir / "requirements.txt").is_file()
    assert (cache_dir / SUBSTRAFL_FOLDER).is_dir()
    assert (cache_dir / SUBSTRAFL_FOLDER / "dist").is_dir()
    assert (cache_dir / SUBSTRAFL_FOLDER / "local_dependencies").is_dir()


def test_get_compute_with_compile():
    dependency = Dependency(
        pypi_dependencies=["pytest"],
        local_installable_dependencies=[CURRENT_FILE.parent / "installable_library"],
        editable_mode=True,
        compile=True,
    )

    cache_dir = dependency.cache_directory

    assert (cache_dir / "requirements.in").is_file()
    assert (cache_dir / "requirements.txt").is_file()


def test_dependency_deletion():
    dependency = Dependency(
        pypi_dependencies=["pytest"],
    )
    cache_dir = dependency._cache_directory
    assert cache_dir.exists()

    del dependency
    assert not cache_dir.exists()
