"""
Create the Substra algo assets and register them to the platform.
"""
import logging
import os
import shutil
import tarfile
import tempfile
import typing
import warnings
from distutils import util
from pathlib import Path
from platform import python_version

import substra
import substratools
from packaging import version

import substrafl
from substrafl.dependency import Dependency
from substrafl.exceptions import SubstraToolsDeprecationWarning
from substrafl.remote.register.generate_wheel import local_lib_wheels
from substrafl.remote.register.generate_wheel import pypi_lib_wheels
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.remote.substratools_methods import RemoteDataMethod

logger = logging.getLogger(__name__)

# Substra tools version for which the image naming scheme changed
MINIMAL_DOCKER_SUBSTRATOOLS_VERSION = "0.16.0"

_DEFAULT_SUBSTRATOOLS_IMAGE = "ghcr.io/substra/substra-tools:\
{substratools_version}-nvidiacuda11.6.0-base-ubuntu20.04-python{python_version}"

SUBSTRAFL_FOLDER = "substrafl_internal"

DOCKERFILE_TEMPLATE = """
FROM {docker_image}

COPY . .

# install dependencies
RUN python{python_version} -m pip install -U pip

# Install substrafl, substra (and substratools if editable mode)
{cl_deps}

# PyPi dependencies
{pypi_dependencies}

# Install local dependencies
{local_dependencies}

ENTRYPOINT ["python{python_version}", "algo.py", "--method-name", "{method_name}"]
"""

ALGO = """
import json
import cloudpickle

import substratools as tools

from substrafl.remote.remote_struct import RemoteStruct

from pathlib import Path

if __name__ == "__main__":
    # Load the wrapped user code
    remote_struct = RemoteStruct.load(src=Path(__file__).parent / '{substrafl_folder}')

    # Create a Substra algo from the wrapped user code
    remote_instance = remote_struct.get_remote_instance()

    # Execute the algo using substra-tools
    tools.algo.execute(remote_instance)
"""


def _copy_local_packages(
    path: Path, local_dependencies: typing.List[Path], python_major_minor: str, operation_dir: Path
):
    """Copy the local libraries given by the user and generate the installation command."""
    local_dependencies_cmd = list()
    path.mkdir(exist_ok=True)
    for dependency_path in local_dependencies:
        dest_path = path / dependency_path.name
        if dependency_path.is_dir():
            shutil.copytree(
                dependency_path, dest_path, ignore=shutil.ignore_patterns("local-worker*", "tmp_substrafl*")
            )
        elif dependency_path.is_file():
            shutil.copy(dependency_path, dest_path)
        else:
            raise ValueError(f"Does not exist {dependency_path}")

        local_dependencies_cmd.append(
            f"RUN python{python_major_minor} -m pip install " f"--no-cache-dir {dest_path.relative_to(operation_dir)}"
        )
    return "\n".join(local_dependencies_cmd)


def _copy_local_code(path: Path, algo_file_path: Path, operation_dir: Path):
    """Copy the local code given by the user to the operation directory."""
    relative_path = path.relative_to(algo_file_path)
    (operation_dir / relative_path.parent).mkdir(exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, operation_dir / relative_path)
    elif path.is_file():
        shutil.copy(path, operation_dir / relative_path)
    else:
        raise ValueError(f"Does not exist {path}")


def _create_archive(archive_path: Path, src_path: Path):
    """Create a tar archive from a folder"""
    with tarfile.open(archive_path, "w:gz") as tar:
        for filepath in src_path.glob("*"):
            if not filepath.name.endswith(".tar.gz"):
                tar.add(filepath, arcname=filepath.name, recursive=True)


def _get_base_docker_image(python_major_minor: str, editable_mode: bool):
    """Get the base Docker image for the Dockerfile"""

    substratools_image_version = substratools.__version__
    if util.strtobool(os.environ.get("USE_LATEST_SUBSTRATOOLS", "False")):
        substratools_image_version = "latest"
    elif version.parse(substratools_image_version) < version.parse(MINIMAL_DOCKER_SUBSTRATOOLS_VERSION):
        if not editable_mode:
            warnings.warn(
                f"Your environment uses substra-tools={substratools_image_version}. Version \
                {MINIMAL_DOCKER_SUBSTRATOOLS_VERSION} will be used on Docker.",
                SubstraToolsDeprecationWarning,
            )
        substratools_image_version = MINIMAL_DOCKER_SUBSTRATOOLS_VERSION
    substratools_image = _DEFAULT_SUBSTRATOOLS_IMAGE.format(
        substratools_version=substratools_image_version,
        python_version=python_major_minor,
    )

    return substratools_image


def _create_substra_algo_files(
    remote_struct: RemoteStruct,
    install_libraries: bool,
    dependencies: Dependency,
    operation_dir: Path,
) -> typing.Tuple[Path, Path]:
    """Creates the necessary files from the remote struct to register the associated algorithm to substra, zip them into
        an archive (.tar.gz).

        Necessary files:

            - the RemoteStruct (wrapped code) dump file
            - the wheel of the current version of Substrafl and Substra
            - the Dockerfile
            - the description.md
            - the algo.py entrypoint

    Args:
        remote_struct (RemoteStruct): A representation of a substra algorithm.
        install_libraries (bool): whether we need to build the wheels and copy the files to install the libraries
        dependencies (Dependency): Algorithm dependencies.
        operation_dir (pathlib.Path): path to the operation directory

        Returns:
            Tuple[Path, Path]: The archive path and the description file path.
    """
    substrafl_internal = operation_dir / SUBSTRAFL_FOLDER
    substrafl_internal.mkdir()

    remote_struct.save(dest=substrafl_internal)

    # get Python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Build Substrafl, Substra and Substratools wheel if needed
    install_cmd = ""

    if install_libraries:
        # Install either from pypi wheel or repo in editable mode
        if dependencies.editable_mode:
            install_cmd = local_lib_wheels(
                lib_modules=[
                    substratools,
                    substra,
                    substrafl,
                ],  # We reinstall substratools in editable mode to overwrite the installed version
                operation_dir=operation_dir,
                python_major_minor=python_major_minor,
                dest_dir=f"{SUBSTRAFL_FOLDER}/dist",
            )
        else:
            install_cmd = pypi_lib_wheels(
                lib_modules=[substra, substrafl],
                operation_dir=operation_dir,
                python_major_minor=python_major_minor,
                dest_dir=f"{SUBSTRAFL_FOLDER}/dist",
            )

    # Pypi dependencies docker command if specified by the user
    pypi_dependencies_cmd = (
        f"RUN python{python_major_minor} -m pip install --no-cache-dir {' '.join(dependencies.pypi_dependencies)}"
        if dependencies is not None and len(dependencies.pypi_dependencies) > 0
        else ""
    )

    # The files to copy to the container must be in the same folder as the Dockerfile
    local_dependencies_cmd = ""
    if dependencies is not None:
        algo_file_path = remote_struct.get_cls_file_path()
        for path in dependencies.local_code:
            _copy_local_code(path=path, algo_file_path=algo_file_path, operation_dir=operation_dir)

        if install_libraries:
            local_dep_dir = substrafl_internal / "local_dependencies"
            local_dependencies_cmd = _copy_local_packages(
                path=local_dep_dir,
                local_dependencies=dependencies.local_dependencies,
                python_major_minor=python_major_minor,
                operation_dir=operation_dir,
            )

    # Write template to algo.py
    algo_path = operation_dir / "algo.py"
    algo_path.write_text(
        ALGO.format(
            substrafl_folder=SUBSTRAFL_FOLDER,
        )
    )

    # Write description
    description_path = substrafl_internal / "description.md"
    description_path.write_text("# Substrafl Operation")

    # Get the base Docker image
    substratools_image = _get_base_docker_image(
        python_major_minor=python_major_minor, editable_mode=dependencies.editable_mode
    )

    # Write dockerfile based on template
    dockerfile_path = operation_dir / "Dockerfile"
    dockerfile_path.write_text(
        DOCKERFILE_TEMPLATE.format(
            docker_image=substratools_image,
            python_version=python_major_minor,
            cl_deps=install_cmd,
            pypi_dependencies=pypi_dependencies_cmd,
            local_dependencies=local_dependencies_cmd,
            # INFO: this is a temporary solution
            # At the moment the method name is the one from the former Melody
            # schemas, this will be variabilized soon
            method_name=remote_struct._method_name
            if remote_struct._remote_cls.__name__ == RemoteDataMethod.__name__
            else "aggregate",
        )
    )

    # Create necessary archive to register the operation on substra
    archive_path = operation_dir / "algo.tar.gz"
    _create_archive(archive_path=archive_path, src_path=operation_dir)

    return archive_path, description_path


def register_algo(
    client: substra.Client,
    remote_struct: RemoteStruct,
    category: substra.sdk.schemas.AlgoCategory,
    permissions: substra.sdk.schemas.Permissions,
    inputs: typing.List[substra.sdk.schemas.AlgoInputSpec],
    outputs: typing.List[substra.sdk.schemas.AlgoOutputSpec],
    dependencies: Dependency,
) -> str:
    """Automatically creates the needed files to register the composite algorithm associated to the remote_struct.

    Args:
        client (substra.Client): The substra client.
        remote_struct (RemoteStruct): The substra submittable algorithm representation.
        category (substra.sdk.schemas.AlgoCategory): Register the algorithm to the platform for the composite, predict
            or aggregate categories.
        permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
        inputs (typing.List[substra.sdk.schemas.AlgoInputSpec]): List of algo inputs to be used.
        outputs (typing.List[substra.sdk.schemas.AlgoOutputSpec]): List of algo outputs to be used.
        dependencies (Dependency): Algorithm dependencies.

    Returns:
        str: Substra algorithm key.
    """
    with tempfile.TemporaryDirectory(dir=str(Path.cwd().resolve()), prefix="tmp_substrafl_") as operation_dir:
        archive_path, description_path = _create_substra_algo_files(
            remote_struct,
            dependencies=dependencies,
            install_libraries=client.backend_mode != substra.BackendType.LOCAL_SUBPROCESS,
            operation_dir=Path(operation_dir),
        )
        key = client.add_algo(
            substra.sdk.schemas.AlgoSpec(
                name=remote_struct.algo_name,
                description=description_path,
                file=archive_path,
                inputs=inputs,
                outputs=outputs,
                permissions=permissions,
                metadata=dict(),
                category=category,
            )
        )
        return key
