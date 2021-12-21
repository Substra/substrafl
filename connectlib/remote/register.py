import inspect
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path
from platform import python_version
from typing import List, Optional, Tuple

import cloudpickle
import substra
import substratools

import connectlib
from connectlib.dependency import Dependency
from connectlib.remote.methods import RemoteStruct

CONNECTLIB_FOLDER = "connectlib_internal"

# TODO: need to have the GPU drivers in the Docker image
DOCKERFILE_TEMPLATE = """
FROM python:{python_version}
WORKDIR /sandbox
ENV PYTHONPATH /sandbox

COPY . .

# install dependencies
RUN python{python_version} -m pip install -U pip

# Install connectlib, substra and substratools
{cl_deps}

# PyPi dependencies
{pypi_dependencies}

# Copy local code
{local_code}

# Install local dependencies
{local_dependencies}

ENTRYPOINT ["python{python_version}", "algo.py"]
"""

ALGO = """
import json
import cloudpickle

import substratools as tools

from connectlib.remote.methods import {csl_name}

from pathlib import Path

if __name__ == "__main__":
    cls_parameters_path = Path(__file__).parent / "{connectlib_folder}" / "cls_parameters.json"

    with cls_parameters_path.open("r") as f:
        cls_parameters = json.load(f)
    print(cls_parameters)

    cls_cloudpickle_path = Path(__file__).parent / "{connectlib_folder}" / "cls_cloudpickle"

    with cls_cloudpickle_path.open("rb") as f:
        cls = cloudpickle.load(f)

    instance = cls(*cls_parameters["args"], **cls_parameters["kwargs"])

    remote_cls_parameters_path = Path(__file__).parent / "{connectlib_folder}" / "remote_cls_parameters.json"

    with remote_cls_parameters_path.open("r") as f:
        remote_cls_parameters = json.load(f)

    tools.algo.execute({csl_name}(instance, *remote_cls_parameters["args"], **remote_cls_parameters["kwargs"]))
"""


def get_local_lib(lib_modules: List, operation_dir: Path, python_major_minor) -> str:
    """Prepares the private libraries from lib_modules list
    to be installed in the Docker and makes the command for dockerfile.
    It first creates the wheel for each library. Each of the libraries must be already installed in the correct version
    locally. Use command: `pip install -e library-name` in the directory of each library.

    Args:
        lib_modules (`list`): list of modules to be installed.
        operation_dir (Path): PosixPath to the operation directory
        python_major_minor (str): version which is to be used in the dockerfile. Eg: '3.8'

    Returns:
        str: dockerfile command for installing the given modules
    """
    install_cmds = []
    wheels_dir = operation_dir / CONNECTLIB_FOLDER / "dist"
    wheels_dir.mkdir(exist_ok=True)

    for lib_module in lib_modules:
        if not (Path(lib_module.__file__).parents[1] / "setup.py").exists():
            # TODO: add private pypi (eg user needs to pass the credentials)
            raise NotImplementedError(
                "You must have connectlib, substra and substratools in editable mode.\n"
                "eg `pip install -e substra` in the substra directory"
            )
        lib_name = lib_module.__name__
        lib_path = Path(lib_module.__file__).parents[1]
        wheel_name = f"{lib_name}-{lib_module.__version__}-py3-none-any.whl"

        # Recreate the wheel only if it exists
        # TODO: only for dev, see what we do in another PR
        if not (lib_path / "dist" / wheel_name).exists():
            # if the right version of substra or substratools is not found, it will search if they are already
            # installed in 'dist' and take them from there.
            # sys.executable takes the Python interpreter run by the code and not the default one on the computer
            extra_args: list = list()
            if lib_name == "connectlib":
                extra_args = [
                    "--find-links",
                    operation_dir / "dist/substra",
                    "--find-links",
                    operation_dir / "dist/substratools",
                ]
            try:
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "wheel",
                        ".",
                        "-w",
                        "dist",
                        "--no-deps",
                    ]
                    + extra_args,
                    cwd=str(lib_path),
                )
            except subprocess.CalledProcessError as e:
                print(e.output)

        # Get wheel name based on current version
        shutil.copy(lib_path / "dist" / wheel_name, wheels_dir / wheel_name)
        # Necessary command to install the wheel in the docker image
        install_cmd = f"RUN cd {CONNECTLIB_FOLDER}/dist && python{python_major_minor} -m pip install {wheel_name}\n"
        install_cmds.append(install_cmd)
    return "\n".join(install_cmds)


def create_substra_algo_files(
    remote_struct: RemoteStruct,
    dependencies: Optional[Dependency] = None,
) -> Tuple[Path, Path]:
    """Creates the necessary files from the remote struct to register the associated algorithm to substra, zip them into
    an archive (.tar.gz).

    Necessary files :
        - the class Cloudpickle
        - the instance parameters captured by Blueprint
        - the wheel of the current version of Connectlib if in editable mode
        - the Dockerfile
        - the description.md
        - the algo.py entrypoint

    Args:
        remote_struct (RemoteStruct): A representation of a substra algorithm.
        dependencies (Optional[List[str]], optional): The list of public dependencies of the algorithm. Defaults to None.

    Returns:
        Tuple[Path, Path]: The archive path and the description file path.
    """

    operation_dir = Path(tempfile.mkdtemp())
    connectlib_internal = operation_dir / CONNECTLIB_FOLDER
    connectlib_internal.mkdir()

    # serialize cls
    cloudpickle_path = connectlib_internal / "cls_cloudpickle"
    with cloudpickle_path.open("wb") as f:
        cloudpickle.dump(remote_struct.cls, f)

    # serialize cls parameters
    cls_parameters_path = connectlib_internal / "cls_parameters.json"
    cls_parameters_path.write_text(remote_struct.cls_parameters)

    # serialize remote cls parameters
    remote_cls_parameters_path = connectlib_internal / "remote_cls_parameters.json"
    remote_cls_parameters_path.write_text(remote_struct.remote_cls_parameters)

    # get Python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Build Connectlib, Substra and Substratools wheel if needed
    lib_modules = [substratools, substra, connectlib]  # owkin private dependencies
    install_cmd = get_local_lib(lib_modules, operation_dir, python_major_minor)

    # Pypi dependencies docker command if specified by the user
    pypi_dependencies_cmd = (
        f"RUN python{python_major_minor} -m pip install --no-cache-dir {' '.join(dependencies.pypi_dependencies)}"
        if dependencies is not None and len(dependencies.pypi_dependencies) > 0
        else ""
    )

    # The files to copy to the container must be in the same folder as the Dockerfile
    local_code_cmd = ""
    local_dependencies_cmd = ""
    if dependencies is not None:
        algo_file_path = Path(inspect.getfile(remote_struct.cls)).resolve().parent
        for path in dependencies.local_code:
            relative_path = path.relative_to(algo_file_path)
            (operation_dir / relative_path.parent).mkdir(exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, operation_dir / relative_path)
            elif path.is_file():
                shutil.copy(path, operation_dir / relative_path)
            else:
                raise ValueError(f"Does not exist {path}")

        for path in dependencies.local_dependencies:
            dest_path = connectlib_internal / "local_dependencies" / path.name
            if path.is_dir():
                shutil.copytree(path, dest_path)
            elif path.is_file():
                shutil.copy(path, dest_path)
            else:
                raise ValueError(f"Does not exist {path}")

            local_dependencies_cmd += f"RUN python{python_major_minor} -m pip install --no-cache-dir -e {dest_path.relative_to(operation_dir)}"

    # Write template to algo.py
    algo_path = operation_dir / "algo.py"
    algo_path.write_text(
        ALGO.format(
            csl_name=remote_struct.remote_cls_name, connectlib_folder=CONNECTLIB_FOLDER
        )
    )

    # Write description
    description_path = connectlib_internal / "description.md"
    description_path.write_text("# ConnnectLib Operation")

    # Write dockerfile based on template
    dockerfile_path = operation_dir / "Dockerfile"
    dockerfile_path.write_text(
        DOCKERFILE_TEMPLATE.format(
            python_version=python_major_minor,
            cl_deps=install_cmd,
            pypi_dependencies=pypi_dependencies_cmd,
            local_dependencies=local_dependencies_cmd,
            local_code=local_code_cmd,
            cloudpickle_path=cloudpickle_path.name,
            cls_parameters_path=cls_parameters_path.name,
            remote_cls_parameters_path=remote_cls_parameters_path.name,
        )
    )

    # Create necessary archive to register the operation on substra
    archive_path = operation_dir / "algo.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for filepath in operation_dir.glob("*"):
            if not filepath.name.endswith(".tar.gz"):
                tar.add(filepath, arcname=os.path.basename(filepath), recursive=True)
    return archive_path, description_path


def register_algo(
    client: substra.Client,
    remote_struct: RemoteStruct,
    is_composite: bool,
    permissions: substra.sdk.schemas.Permissions,
    dependencies: Optional[Dependency] = None,
) -> str:
    """Automatically creates the needed files to register the composite algorithm associated to the remote_struct.

    Args:
        client (substra.Client): The substra client.
        remote_struct (RemoteStruct): The substra submitable algorithm representation.
        is_composite (bool): Either to register a composite or an aggregate algorithm.
        permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
        dependencies (Optional[List[str]], optional): Public algorithm dependencies. Defaults to None.
    Returns:
        str: Substra algorithm key.
    """
    archive_path, description_path = create_substra_algo_files(
        remote_struct, dependencies=dependencies
    )
    if is_composite:
        category = substra.sdk.schemas.AlgoCategory.composite
    else:
        category = substra.sdk.schemas.AlgoCategory.aggregate

    key = client.add_algo(
        substra.sdk.schemas.AlgoSpec(
            name=uuid.uuid4().hex,
            description=description_path,
            file=archive_path,
            permissions=permissions,
            metadata=dict(),
            category=category,
        )
    )
    return key
