"""
Create the Substra function assets and register them to the platform.
"""

import logging
import shutil
import tarfile
import tempfile
import typing
from pathlib import Path
from pathlib import PurePosixPath
from platform import python_version

import substra

from substrafl.constants import SUBSTRAFL_FOLDER
from substrafl.constants import TMP_SUBSTRAFL_PREFIX
from substrafl.dependency import Dependency
from substrafl.exceptions import UnsupportedPythonVersionError
from substrafl.remote.remote_struct import RemoteStruct

logger = logging.getLogger(__name__)

# Substra tools version for which the image naming scheme changed
MINIMAL_DOCKER_SUBSTRATOOLS_VERSION = "0.16.0"

# minimal and maximal values of Python 3 minor versions supported
# we need to store this as integer, else "3.11" < "3.9" (string comparison)
MINIMAL_PYTHON_VERSION = 10  # 3.10
MAXIMAL_PYTHON_VERSION = 12  # 3.12

_CPU_BASE_IMAGE = """
FROM python:{python_version}-slim

# update image
RUN apt-get update -y && pip uninstall -y setuptools
"""

_CPU_BASE_IMAGE_WITH_DEPENDENCIES = """
FROM python:{python_version}-slim

# update image
RUN apt-get update -y\
    && pip uninstall -y setuptools\
    && apt-get install -y {binary_dependencies}\
    && apt-get clean
"""

_GPU_BASE_IMAGE = """
FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

# update image & install Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y\
    && apt-get install -y software-properties-common\
    && add-apt-repository -y ppa:deadsnakes/ppa\
    && apt-get -y upgrade\
    && apt-get install -y python{python_version} python{python_version}-venv python3-pip {binary_dependencies}\
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/*

"""

DOCKERFILE_TEMPLATE = """{base_docker_image}
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
{copy_wheels}

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Install requirements
RUN python{python_version} -m pip install --no-cache-dir -r requirements.txt

USER root
RUN apt-get update -y && apt-get purge -y --auto-remove build-essential *-dev
USER user

# Copy all other files
COPY function.py .
COPY {internal_dir}/cls_cloudpickle {internal_dir}/
COPY {internal_dir}/description.md {internal_dir}/
{copy_local_code}

ENTRYPOINT ["python{python_version}", "function.py", "--function-name", "{method_name}"]
"""

FUNCTION = """
import json
import cloudpickle

import substratools as tools

from substrafl.remote.remote_struct import RemoteStruct

from pathlib import Path

if __name__ == "__main__":
    # Load the wrapped user code
    remote_struct = RemoteStruct.load(src=Path(__file__).parent / '{substrafl_folder}')

    # Create a Substra function from the wrapped user code
    remote_instance = remote_struct.get_remote_instance()

    # Register the functions to substra-tools
    remote_instance.register_substratools_function()

    # Execute the function using substra-tools
    tools.execute()
"""


def _create_archive(archive_path: Path, src_path: Path):
    """Create a tar archive from a folder"""
    with tarfile.open(archive_path, "w:gz") as tar:
        for filepath in src_path.glob("*"):
            if not filepath.name.endswith(".tar.gz"):
                tar.add(filepath, arcname=filepath.name, recursive=True)


def _check_python_version(python_major_minor: str) -> None:
    """Raise UnsupportedPythonVersionError exception if the Python version is not supported"""
    major, minor = python_major_minor.split(".")
    if major != "3":
        raise UnsupportedPythonVersionError("Only Python 3 is supported")
    if int(minor) < MINIMAL_PYTHON_VERSION or int(minor) > MAXIMAL_PYTHON_VERSION:
        raise UnsupportedPythonVersionError(
            f"The current Python version is {python_major_minor}, which is unsupported;"
            f"supported versions are 3.{MINIMAL_PYTHON_VERSION} to 3.{MAXIMAL_PYTHON_VERSION}"
        )


def _get_base_docker_image(
    python_major_minor: str, use_gpu: bool, custom_binary_dependencies: typing.Optional[list] = None
) -> str:
    """Get the base Docker image for the Dockerfile"""

    if use_gpu:
        base_docker_image = _GPU_BASE_IMAGE.format(
            python_version=python_major_minor, binary_dependencies=" ".join(custom_binary_dependencies or [])
        )
    elif custom_binary_dependencies:
        base_docker_image = _CPU_BASE_IMAGE_WITH_DEPENDENCIES.format(
            python_version=python_major_minor, binary_dependencies=" ".join(custom_binary_dependencies)
        )
    else:
        base_docker_image = _CPU_BASE_IMAGE.format(
            python_version=python_major_minor,
        )

    return base_docker_image


def _generate_copy_local_files(local_files: typing.List[Path]) -> str:
    # In Dockerfiles, we need to always have '/'. PurePosixPath resolves that.
    return "\n".join([f"COPY {PurePosixPath(file)} {PurePosixPath(file)}" for file in local_files])


def _create_dockerfile(install_libraries: bool, dependencies: Dependency, operation_dir: Path, method_name: str) -> str:
    # get Python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # check that the Python version is supported
    _check_python_version(python_major_minor)

    # Get the base Docker image
    base_docker_image = _get_base_docker_image(
        python_major_minor=python_major_minor,
        use_gpu=dependencies.use_gpu,
        custom_binary_dependencies=dependencies.binary_dependencies,
    )
    # Build Substrafl, Substra and Substratools, and local dependencies wheels if necessary
    if install_libraries:
        # generate the copy wheel command
        copy_wheels_cmd = _generate_copy_local_files(dependencies._wheels)

    else:
        copy_wheels_cmd = ""

    # user-defined local dependencies
    copy_local_code_cmd = _generate_copy_local_files(dependencies._local_paths)

    return DOCKERFILE_TEMPLATE.format(
        base_docker_image=base_docker_image,
        python_version=python_major_minor,
        copy_wheels=copy_wheels_cmd,
        copy_local_code=copy_local_code_cmd,
        method_name=method_name,
        internal_dir=SUBSTRAFL_FOLDER,
    )


def _create_substra_function_files(
    remote_struct: RemoteStruct,
    install_libraries: bool,
    dependencies: Dependency,
    operation_dir: Path,
) -> typing.Tuple[Path, Path]:
    """Creates the necessary files from the remote struct to register the associated function to substra, zip them into
        an archive (.tar.gz).

        Necessary files:

            - the RemoteStruct (wrapped code) dump file
            - the wheel of the current version of Substrafl and Substra
            - the Dockerfile
            - the description.md
            - the function.py entrypoint

        Target tree structure:
            ├── Dockerfile
            ├── function.py
            ├── function.tar.gz
            ├── local_code.py
            └── substrafl_internal
                ├── cls_cloudpickle
                ├── description.md
                ├── dist
                │   ├── substra-0.44.0-py3-none-any.whl
                │   ├── substrafl-0.36.0-py3-none-any.whl
                │   └── substratools-0.20.0-py3-none-any.whl
                ├── local_dependencies
                │   └── local-module-1.6.1-py3-none-any.whl
                ├── requirements.in
                └── requirements.txt

    Args:
        remote_struct (RemoteStruct): A representation of a substra algorithm.
        install_libraries (bool): whether we need to build the wheels and copy the files to install the libraries
        dependencies (Dependency): Function dependencies.
        operation_dir (pathlib.Path): path to the operation directory

        Returns:
            Tuple[Path, Path]: The archive path and the description file path.
    """
    substrafl_internal = operation_dir / SUBSTRAFL_FOLDER
    substrafl_internal.mkdir()

    dependency_cache_folder = dependencies.cache_directory
    shutil.copytree(dependency_cache_folder, operation_dir, dirs_exist_ok=True)

    remote_struct.save(dest=substrafl_internal)

    # Write dockerfile based on template
    dockerfile_path = operation_dir / "Dockerfile"
    dockerfile_path.write_text(
        _create_dockerfile(
            install_libraries=install_libraries,
            dependencies=dependencies,
            operation_dir=operation_dir,
            method_name=remote_struct._method_name,
        )
    )

    # Write template to function.py
    function_path = operation_dir / "function.py"
    function_path.write_text(
        FUNCTION.format(
            substrafl_folder=SUBSTRAFL_FOLDER,
        )
    )

    # Write description
    description_path = substrafl_internal / "description.md"
    description_path.write_text("# Substrafl Operation")

    # Create necessary archive to register the operation on substra
    archive_path = operation_dir / "function.tar.gz"
    _create_archive(archive_path=archive_path, src_path=operation_dir)

    return archive_path, description_path


def register_function(
    *,
    client: substra.Client,
    remote_struct: RemoteStruct,
    permissions: substra.sdk.schemas.Permissions,
    inputs: typing.List[substra.sdk.schemas.FunctionInputSpec],
    outputs: typing.List[substra.sdk.schemas.FunctionOutputSpec],
    dependencies: Dependency,
) -> str:
    """Automatically creates the needed files to register the function associated to the remote_struct.

    Args:
        client (substra.Client): The substra client.
        remote_struct (RemoteStruct): The substra submittable function representation.
        permissions (substra.sdk.schemas.Permissions): Permissions for the function.
        inputs (typing.List[substra.sdk.schemas.FunctionInputSpec]): List of function inputs to be used.
        outputs (typing.List[substra.sdk.schemas.FunctionOutputSpec]): List of function outputs to be used.
        dependencies (Dependency): Function dependencies.

    Returns:
        str: Substra function key.
    """
    with tempfile.TemporaryDirectory(dir=str(Path.cwd().resolve()), prefix=TMP_SUBSTRAFL_PREFIX) as operation_dir:
        archive_path, description_path = _create_substra_function_files(
            remote_struct,
            dependencies=dependencies,
            install_libraries=client.backend_mode != substra.BackendType.LOCAL_SUBPROCESS,
            operation_dir=Path(operation_dir),
        )
        key = client.add_function(
            substra.sdk.schemas.FunctionSpec(
                name=remote_struct.algo_name,
                description=description_path,
                file=archive_path,
                inputs=inputs,
                outputs=outputs,
                permissions=permissions,
                metadata=dict(),
            )
        )
        return key
