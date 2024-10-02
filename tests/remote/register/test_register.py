import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import substra
import substratools

import substrafl
from substrafl.dependency import Dependency
from substrafl.exceptions import UnsupportedPythonVersionError
from substrafl.remote.decorators import remote_data
from substrafl.remote.register import register
from substrafl.remote.register.register import _create_dockerfile
from substrafl.remote.register.register import _get_base_docker_image


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


@pytest.mark.parametrize("version", ["2.7", "3.7", "3.8", "3.9", "3.18"])
def test_check_python_version(version):
    with pytest.raises(UnsupportedPythonVersionError):
        register._check_python_version(version)


@pytest.mark.parametrize("version", ["3.10", "3.11", "3.12"])
def test_check_python_version_valid(version):
    """Does not raise for supported versions"""
    register._check_python_version(version)


def test_get_base_docker_image_cpu():
    expected_dockerfile = """
FROM python:3.12-slim

# update image
RUN apt-get update -y && pip uninstall -y setuptools
"""
    assert expected_dockerfile == _get_base_docker_image("3.12", use_gpu=False)


def test_get_base_docker_image_gpu():
    expected_dockerfile = """
FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

# update image & install Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y\
    && apt-get install -y software-properties-common\
    && add-apt-repository -y ppa:deadsnakes/ppa\
    && apt-get -y upgrade\
    && apt-get install -y python3.11 python3.11-venv python3-pip \
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/*

"""
    assert expected_dockerfile == _get_base_docker_image("3.11", use_gpu=True)


def test_create_dockerfile(tmp_path, local_installable_module):
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
        use_gpu=False,
    )
    dependencies._compute_in_cache_directory

    expected_dockerfile = f"""
FROM python:{python_version}-slim

# update image
RUN apt-get update -y && pip uninstall -y setuptools

# create a non-root user
RUN addgroup --gid 1001 group
RUN adduser --disabled-password --gecos "" --uid 1001 --gid 1001 --home /home/user user
WORKDIR /home/user
USER user

RUN python{python_version} -m venv /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH" VIRTUAL_ENV="/home/user/venv"

# install dependencies
RUN python{python_version} -m pip install -U pip && pip install -U setuptools>=70.0.0

# Copy local wheels
COPY {substrafl_wheel} {substrafl_wheel}
COPY {substra_wheel} {substra_wheel}
COPY {substratools_wheel} {substratools_wheel}
COPY {local_installable_wheel} {local_installable_wheel}

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Install requirements
RUN python{python_version} -m pip install --no-cache-dir -r requirements.txt

USER root
RUN apt-get purge -y --auto-remove build-essential *-dev
USER user

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
