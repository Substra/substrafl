import tarfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import substra

from substrafl import exceptions
from substrafl.dependency import Dependency
from substrafl.remote.decorators import remote_data
from substrafl.remote.register import register


class RemoteClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote_data
    def foo(self):
        pass


class DummyClient:
    backend_mode = substra.BackendType.LOCAL_SUBPROCESS

    def add_algo(*args):
        pass


@pytest.mark.parametrize("use_latest", [True, False])
def test_latest_substratools_image_selection(use_latest, monkeypatch, default_permissions):
    monkeypatch.setenv("USE_LATEST_SUBSTRATOOLS", str(use_latest))

    client = substra.Client(backend_type=substra.BackendType.LOCAL_SUBPROCESS)

    my_class = RemoteClass()

    data_op = my_class.foo(data_samples=["fake_path"], shared_state=None)

    remote_struct = data_op.remote_struct

    algo_deps = Dependency()

    algo_key = register.register_algo(
        client=client,
        remote_struct=remote_struct,
        permissions=default_permissions,
        dependencies=algo_deps,
        inputs=None,  # No need to register inputs and outputs as this algo is not actually used
        outputs=None,
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


@patch("substra.sdk.schemas.FunctionSpec", MagicMock(return_value=None))
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

    _ = register.register_algo(
        client=client,
        remote_struct=remote_struct,
        permissions=default_permissions,
        dependencies=algo_deps,
        inputs=None,  # No need to register inputs and outputs as this algo is not actually used
        outputs=None,
    )

    assert substra.sdk.schemas.FunctionSpec.call_args[1]["name"] == result


@pytest.mark.parametrize(
    "metric_function, error",
    [
        (lambda wrong_arg: "any_str", exceptions.MetricFunctionSignatureError),
        (lambda datasamples: "any_str", exceptions.MetricFunctionSignatureError),
        (lambda predictions_path: "any_str", exceptions.MetricFunctionSignatureError),
        (lambda datasamples, predictions_path, wrong_arg: "any_str", exceptions.MetricFunctionSignatureError),
        ("not a function", exceptions.MetricFunctionTypeError),
        (lambda datasamples, predictions_path: "any_str", None),
    ],
)
def test_add_metric_wrong_metric_function(metric_function, error, default_permissions):
    client = DummyClient()

    metric_deps = Dependency()

    if error is not None:
        with pytest.raises(error):
            register.add_metric(
                client,
                default_permissions,
                metric_deps,
                metric_function,
            )
    else:
        register.add_metric(
            client,
            default_permissions,
            metric_deps,
            metric_function,
        )
