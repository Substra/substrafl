import sys
import tarfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import substra
import substratools

import substrafl
from substrafl.dependency import Dependency
from substrafl.exceptions import UnsupportedPythonVersionError
from substrafl.nodes import TestDataNode
from substrafl.remote.decorators import remote_data
from substrafl.remote.register import register
from substrafl.remote.register import register_metrics
from substrafl.remote.register.register import _create_dockerfile


class RemoteClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote_data
    def foo(self):
        pass


class DummyClient:
    backend_mode = substra.BackendType.LOCAL_SUBPROCESS

    def add_function(*args):
        pass


@pytest.mark.parametrize("version", ["2.7", "3.7", "3.18"])
def test_check_python_version(version):
    with pytest.raises(UnsupportedPythonVersionError):
        register._check_python_version(version)


@pytest.mark.parametrize("version", ["3.8", "3.9", "3.10"])
def test_check_python_version_valid(version):
    """Does not raise for supported versions"""
    register._check_python_version(version)


@pytest.mark.parametrize("use_latest", [True, False])
def test_latest_substratools_image_selection(use_latest, monkeypatch, default_permissions):
    monkeypatch.setenv("USE_LATEST_SUBSTRATOOLS", str(use_latest))

    client = substra.Client(backend_type=substra.BackendType.LOCAL_SUBPROCESS)

    my_class = RemoteClass()

    data_op = my_class.foo(data_samples=["fake_path"], shared_state=None)

    remote_struct = data_op.remote_struct

    function_deps = Dependency()

    function_key = register.register_function(
        client=client,
        remote_struct=remote_struct,
        permissions=default_permissions,
        dependencies=function_deps,
        inputs=None,  # No need to register inputs and outputs as this algo is not actually used
        outputs=None,
    )

    function = client.get_function(function_key)

    with tarfile.open(function.function.storage_address, "r:gz") as tar:
        dockerfile = tar.extractfile("Dockerfile")
        lines = dockerfile.readlines()

    if use_latest:
        assert "latest" in str(lines[1])
    else:
        assert "latest" not in str(lines[1])


def test_create_dockerfile(tmp_path, mocker, local_installable_module):
    mocker.patch("substrafl.remote.register.register._get_base_docker_image", return_value="substratools-mocked")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    substrafl_wheel = f"substrafl_internal/dist/substrafl-{substrafl.__version__}-py3-none-any.whl"
    substra_wheel = f"substrafl_internal/dist/substra-{substra.__version__}-py3-none-any.whl"
    substratools_wheel = f"substrafl_internal/dist/substratools-{substratools.__version__}-py3-none-any.whl"
    local_installable_dependencies = local_installable_module(tmp_path)
    local_installable_wheel = "substrafl_internal/local_dependencies/mymodule-1.0.2-py3-none-any.whl"
    local_code_folder = tmp_path / "local"
    local_code_folder.mkdir()
    local_code = local_code_folder / "foo.py"
    local_code.touch()

    dependencies = Dependency(
        editable_mode=True,
        pypi_dependencies=[],
        local_installable_dependencies=[local_installable_dependencies],
        local_code=[local_code_folder],
    )
    dependencies._compute_in_cache_directory()

    expected_dockerfile = f"""
FROM substratools-mocked

# install dependencies
RUN python{python_version} -m pip install -U pip

# Copy local wheels
COPY {substrafl_wheel} {substrafl_wheel}
COPY {substra_wheel} {substra_wheel}
COPY {substratools_wheel} {substratools_wheel}
COPY {local_installable_wheel} {local_installable_wheel}

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Install requirements
RUN python{python_version} -m pip install --no-cache-dir -r requirements.txt

# Copy all other files
COPY function.py .
COPY substrafl_internal/cls_cloudpickle substrafl_internal/
COPY substrafl_internal/description.md substrafl_internal/
COPY local local

ENTRYPOINT ["python{python_version}", "function.py", "--function-name", "foo_bar"]
"""
    dockerfile = _create_dockerfile(True, dependencies, tmp_path, "foo_bar")
    assert dockerfile == expected_dockerfile


@pytest.mark.parametrize("algo_name, result", [("Dummy Algo Name", "Dummy Algo Name"), (None, "foo_RemoteClass")])
def test_algo_name(algo_name, result):
    my_class = RemoteClass()

    data_op = my_class.foo(
        data_samples=["fake_path"],
        shared_state=None,
        _algo_name=algo_name,
    )

    remote_struct = data_op.remote_struct

    assert remote_struct.algo_name == result


@patch("substra.sdk.schemas.FunctionSpec", MagicMock(return_value=None))
@pytest.mark.parametrize("algo_name, result", [("Dummy Algo Name", "Dummy Algo Name"), (None, "foo_RemoteClass")])
def test_register_function_name(algo_name, result, default_permissions):
    client = DummyClient()

    my_class = RemoteClass()

    data_op = my_class.foo(
        data_samples=["fake_path"],
        shared_state=None,
        _algo_name=algo_name,
    )

    remote_struct = data_op.remote_struct

    algo_deps = Dependency()

    _ = register.register_function(
        client=client,
        remote_struct=remote_struct,
        permissions=default_permissions,
        dependencies=algo_deps,
        inputs=None,  # No need to register inputs and outputs as this algo is not actually used
        outputs=None,
    )

    assert substra.sdk.schemas.FunctionSpec.call_args[1]["name"] == result


@patch("substra.sdk.schemas.FunctionSpec", MagicMock(return_value=None))
def test_register_metrics(default_permissions):
    client = DummyClient()
    algo_deps = Dependency()

    def f(datasamples, predictions_path):
        return

    def g(datasamples, predictions_path):
        return

    def h(datasamples, predictions_path):
        return

    expected_identifier = ["f", "g", "h"]

    test_data_node = TestDataNode(
        organization_id="fake_id",
        data_manager_key="fake_id",
        test_data_sample_keys=["fake_id"],
        metric_functions=[f, g, h],
    )

    _ = register_metrics(
        client=client,
        dependencies=algo_deps,
        permissions=default_permissions,
        metric_functions=test_data_node.metric_functions,
    )

    list_identifiers = []

    for output in substra.sdk.schemas.FunctionSpec.call_args[1]["outputs"]:
        assert output.kind == substra.sdk.schemas.AssetKind.performance
        list_identifiers.append(output.identifier)
    assert list_identifiers == expected_identifier
