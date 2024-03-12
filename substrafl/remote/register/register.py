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
MINIMAL_PYTHON_VERSION = 9  # 3.9
MAXIMAL_PYTHON_VERSION = 11  # 3.11

_DEFAULT_BASE_DOCKER_IMAGE = "python:{python_version}-slim"

DOCKERFILE_TEMPLATE = """
FROM {docker_image}

# install dependencies
RUN python{python_version} -m pip install -U pip

# Copy local wheels
{copy_wheels}

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Install requirements
RUN python{python_version} -m pip install --no-cache-dir -r requirements.txt

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


def _get_base_docker_image(python_major_minor: str, editable_mode: bool) -> str:
    """Get the base Docker image for the Dockerfile"""
    _check_python_version(python_major_minor)

    substratools_image = _DEFAULT_BASE_DOCKER_IMAGE.format(
        python_version=python_major_minor,
    )

    return substratools_image


def _generate_copy_local_files(local_files: typing.List[Path]) -> str:
    # In Dockerfiles, we need to always have '/'. PurePosixPath resolves that.
    return "\n".join([f"COPY {PurePosixPath(file)} {PurePosixPath(file)}" for file in local_files])


def _create_dockerfile(install_libraries: bool, dependencies: Dependency, operation_dir: Path, method_name: str) -> str:
    # get Python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Get the base Docker image
    substratools_image = _get_base_docker_image(
        python_major_minor=python_major_minor, editable_mode=dependencies.editable_mode
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
        docker_image=substratools_image,
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
