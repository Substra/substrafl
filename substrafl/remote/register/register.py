"""
Create the Substra function assets and register them to the platform.
"""
import logging
import os
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
from substrafl import exceptions
from substrafl.constants import TMP_SUBSTRAFL_PREFIX
from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.remote.register.generate_wheel import local_lib_wheels
from substrafl.remote.register.generate_wheel import pypi_lib_wheels
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.remote.substratools_methods import RemoteMethod

logger = logging.getLogger(__name__)

# Substra tools version for which the image naming scheme changed
MINIMAL_DOCKER_SUBSTRATOOLS_VERSION = "0.16.0"

_DEFAULT_SUBSTRATOOLS_IMAGE = "ghcr.io/substra/substra-tools:\
{substratools_version}-nvidiacuda11.8.0-base-ubuntu22.04-python{python_version}"

SUBSTRAFL_FOLDER = "substrafl_internal"

DOCKERFILE_TEMPLATE = """
FROM {docker_image}

# install dependencies
RUN python{python_version} -m pip install -U pip

# Install substrafl, substra (and substratools if editable mode)
{cl_deps}

# PyPi dependencies
{pypi_dependencies}

# Install local dependencies
{local_dependencies}

ENTRYPOINT ["python{python_version}", "function.py", "--function-name", "{method_name}"]

COPY . .

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


def iter_dockerfile_install_command(paths: typing.List[Path], python_major_minor: str) -> str:
    for path in paths:
        # Does not work with path.is_dir(), surely a race condition. Should be removed in  #123
        formatted_path = (str(path) + "/") if not path.is_file() else path
        yield f"COPY {path}  {path} \nRUN python{python_major_minor} -m pip install --no-cache-dir {formatted_path} \n"


def _copy_local_packages(path: Path, python_major_minor: str, operation_dir: Path, dependencies: Dependency):
    """Copy the local libraries given by the user and generate the installation command."""
    path.mkdir(exist_ok=True)
    dependencies_paths = dependencies.copy_dependencies_local_package(dest_dir=operation_dir)

    local_dependencies_cmd = "\n".join(iter_dockerfile_install_command(dependencies_paths, python_major_minor))

    return local_dependencies_cmd


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
                exceptions.SubstraToolsDeprecationWarning,
                stacklevel=2,
            )
        substratools_image_version = MINIMAL_DOCKER_SUBSTRATOOLS_VERSION
    substratools_image = _DEFAULT_SUBSTRATOOLS_IMAGE.format(
        substratools_version=substratools_image_version,
        python_version=python_major_minor,
    )

    return substratools_image


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

    remote_struct.save(dest=substrafl_internal)

    # get Python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Build Substrafl, Substra and Substratools wheel if needed
    install_cmd = ""
    if install_libraries:
        # Install either from pypi wheel or repo in editable mode
        if dependencies.editable_mode or util.strtobool(os.environ.get("SUBSTRA_FORCE_EDITABLE_MODE", "False")):
            install_cmd = local_lib_wheels(
                lib_modules=[
                    substrafl,
                    substra,
                    substratools,
                ],  # We reinstall substratools in editable mode to overwrite the installed version
                operation_dir=operation_dir,
                python_major_minor=python_major_minor,
                dest_dir=f"{SUBSTRAFL_FOLDER}/dist",
            )
        else:
            install_cmd = pypi_lib_wheels(
                lib_modules=[substrafl, substra],
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
        dependencies.copy_dependencies_local_code(dest_dir=operation_dir)
        if install_libraries:
            local_dep_dir = substrafl_internal / "local_dependencies"
            local_dependencies_cmd = _copy_local_packages(
                path=local_dep_dir,
                dependencies=dependencies,
                python_major_minor=python_major_minor,
                operation_dir=operation_dir,
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
            method_name=remote_struct._method_name,
        )
    )

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


def register_metrics(
    *,
    client: substra.Client,
    dependencies: Dependency,
    permissions: substra.sdk.schemas.Permissions,
    metric_functions: typing.Dict[str, typing.Callable],
):
    """Adds a function to the Substra platform using the given metric functions as the
    function to register.
    Each metric function must be of type function, and their signature must ONLY contains
    `datasamples` and `predictions_path` as parameters. An error is raised otherwise.

    Args:
        client (substra.Client): The substra client.
        permissions (substra.sdk.schemas.Permissions): Permissions for the metric function.
        dependencies (Dependency): Metric function dependencies.
        metric_functions (typing.Dict[str, typing.Callable]): functions to compute the score from the datasamples and
            the predictions. These functions are registered in substra as one function.

    Returns:
        str: Substra function containing all the given metric functions.
    """

    inputs_metrics = [
        substra.sdk.schemas.FunctionInputSpec(
            identifier=InputIdentifiers.datasamples,
            kind=substra.sdk.schemas.AssetKind.data_sample,
            optional=False,
            multiple=True,
        ),
        substra.sdk.schemas.FunctionInputSpec(
            identifier=InputIdentifiers.opener,
            kind=substra.sdk.schemas.AssetKind.data_manager,
            optional=False,
            multiple=False,
        ),
        substra.sdk.schemas.FunctionInputSpec(
            identifier=InputIdentifiers.predictions,
            kind=substra.sdk.schemas.AssetKind.model,
            optional=False,
            multiple=False,
        ),
    ]

    outputs_metrics = [
        substra.sdk.schemas.FunctionOutputSpec(
            identifier=metric_function_id,
            kind=substra.sdk.schemas.AssetKind.performance,
            multiple=False,
        )
        for metric_function_id in metric_functions
    ]

    class Metric:
        def score(self, datasamples, predictions_path, _skip=True):
            # The _skip argument is needed to match the default signature of methods executed
            # on substratools_methods.py.
            return {
                metric_function_id: metric_function(datasamples=datasamples, predictions_path=predictions_path)
                for metric_function_id, metric_function in metric_functions.items()
            }

    remote_struct = RemoteStruct(
        cls=Metric,
        cls_args=[],
        cls_kwargs={},
        remote_cls=RemoteMethod,
        method_name="score",
        method_parameters={},
        algo_name="Evaluating",
    )

    key = register_function(
        client=client,
        remote_struct=remote_struct,
        permissions=permissions,
        inputs=inputs_metrics,
        outputs=outputs_metrics,
        dependencies=dependencies,
    )

    return key
