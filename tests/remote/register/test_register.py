import tarfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import substra

from connectlib.dependency import Dependency
from connectlib.remote.decorators import remote_data
from connectlib.remote.register.register import register_algo


class RemoteClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote_data
    def foo(self):
        pass


class DummyClient:
    backend_mode = substra.sdk.config.BackendType.LOCAL_SUBPROCESS

    def add_algo(*args):
        pass


@pytest.mark.parametrize("use_latest", [True, False])
def test_latest_connecttools_image_selection(use_latest, monkeypatch, default_permissions):
    monkeypatch.setenv("USE_LATEST_CONNECT_TOOLS", str(use_latest))

    client = substra.Client(debug=True)

    my_class = RemoteClass()

    data_op = my_class.foo(data_samples=["fake_path"], shared_state=None)

    remote_struct = data_op.remote_struct

    algo_deps = Dependency()

    algo_key = register_algo(
        client=client,
        remote_struct=remote_struct,
        is_composite=False,
        permissions=default_permissions,
        dependencies=algo_deps,
    )

    algo = client.get_algo(algo_key)

    with tarfile.open(algo.algorithm.storage_address, "r:gz") as tar:
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


@patch("substra.sdk.schemas.AlgoSpec", MagicMock(return_value=None))
@pytest.mark.parametrize("algo_name, result", [("Dummy Algo Name", "Dummy Algo Name"), (None, "foo_RemoteClass")])
def test_register_algo_name(algo_name, result, default_permissions):

    client = DummyClient()

    my_class = RemoteClass()

    data_op = my_class.foo(
        data_samples=["fake_path"],
        shared_state=None,
        _algo_name=algo_name,
    )

    remote_struct = data_op.remote_struct

    algo_deps = Dependency()

    _ = register_algo(
        client=client,
        remote_struct=remote_struct,
        is_composite=False,
        permissions=default_permissions,
        dependencies=algo_deps,
    )

    assert substra.sdk.schemas.AlgoSpec.call_args[1]["name"] == result