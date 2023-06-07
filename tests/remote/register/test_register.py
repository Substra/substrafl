import tarfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import substra

from substrafl.dependency import Dependency
from substrafl.exceptions import UnsupportedPythonVersionError
from substrafl.nodes import TestDataNode
from substrafl.remote.decorators import remote_data
from substrafl.remote.register import register
from substrafl.remote.register import register_metrics


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
