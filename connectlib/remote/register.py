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
from connectlib.remote.methods import RemoteStruct

# TODO: need to have the GPU drivers in the Docker image
DOCKERFILE_TEMPLATE = """
FROM python:{0}

WORKDIR /sandbox
ENV PYTHONPATH /sandbox

RUN mkdir /wheels

# install dependencies
RUN python{0} -m pip install -U pip

# Install connectlib, substra and substratools
{1}

# Install dependencies
{2}

COPY ./algo.py /algo/algo.py
COPY ./{3} /algo/cls_cloudpickle
COPY ./{4} /algo/cls_parameters.json
COPY ./{5} /algo/remote_cls_parameters.json

ENTRYPOINT ["python{0}", "/algo/algo.py"]
"""

ALGO = """
import json
import cloudpickle

import substratools as tools

from connectlib.remote.methods import {0}

from pathlib import Path

if __name__ == "__main__":
    cls_parameters_path = Path(__file__).parent / "cls_parameters.json"

    with cls_parameters_path.open("r") as f:
        cls_parameters = json.load(f)
    print(cls_parameters)

    cls_cloudpickle_path = Path(__file__).parent / "cls_cloudpickle"

    print(__file__)
    with cls_cloudpickle_path.open("rb") as f:
        cls = cloudpickle.load(f)

    instance = cls(*cls_parameters["args"], **cls_parameters["kwargs"])

    remote_cls_parameters_path = Path(__file__).parent / "remote_cls_parameters.json"

    with remote_cls_parameters_path.open("r") as f:
        remote_cls_parameters = json.load(f)

    tools.algo.execute({0}(instance, *remote_cls_parameters["args"], **remote_cls_parameters["kwargs"]))
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
    for lib_module in lib_modules:
        if not (Path(lib_module.__file__).parents[1] / "setup.py").exists():
            # TODO: add private pypi (eg user needs to pass the credentials)
            raise NotImplementedError(
                "You must have connectlib, substra and substratools in editable mode.\n"
                "eg `pip install -e substra` in the substra directory"
            )
        else:
            lib_name = lib_module.__name__
            lib_path = Path(lib_module.__file__).parents[1]

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
            wheel_name = f"{lib_name}-{lib_module.__version__}-py3-none-any.whl"
            (operation_dir / "dist").mkdir(exist_ok=True)
            shutil.copy(
                lib_path / "dist" / wheel_name, operation_dir / "dist" / wheel_name
            )

            # Necessary command to install the wheel in the docker Image
            install_cmd = (
                f"COPY ./dist/{wheel_name} /wheels/{wheel_name}\n"
                f"RUN cd /wheels && python{python_major_minor} -m pip install {wheel_name}\n"
            )
            install_cmds.append(install_cmd)
    return "\n".join(install_cmds)


def prepare_substra_algo(
    remote_struct: RemoteStruct,
    dependencies: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    # Create temporary directory where we will serialize:
    # - the class Cloudpickle
    # - the instance parameters captured by Blueprint
    # - the wheel of the current version of Connectlib if in editable mode
    # - the Dockerfile
    # - the description.md
    # - the algo.py entrypoint
    operation_dir = Path(tempfile.mkdtemp())

    # serialize cls
    cloudpickle_path = operation_dir / "cls_cloudpickle"
    with cloudpickle_path.open("wb") as f:
        cloudpickle.dump(remote_struct.cls, f)

    # serialize cls parameters
    cls_parameters_path = operation_dir / "cls_parameters.json"
    cls_parameters_path.write_text(remote_struct.cls_parameters)

    # serialize remote cls parameters
    remote_cls_parameters_path = operation_dir / "remote_cls_parameters.json"
    remote_cls_parameters_path.write_text(remote_struct.remote_cls_parameters)

    # get python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Build Connectlib, Substra and Substratools wheel if needed
    lib_modules = [substratools, substra, connectlib]  # owkin private dependencies
    install_cmd = get_local_lib(lib_modules, operation_dir, python_major_minor)

    # Write template to algo.py
    algo_path = operation_dir / "algo.py"
    algo_path.write_text(ALGO.format(remote_struct.remote_cls_name))

    # Write description
    description_path = operation_dir / "description.md"
    description_path.write_text("# ConnnectLib Operation")

    # Write dockerfile based on template
    dockerfile_path = operation_dir / "Dockerfile"
    dockerfile_path.write_text(
        DOCKERFILE_TEMPLATE.format(
            python_major_minor,
            install_cmd,
            f"RUN python{python_major_minor} -m pip install " + " ".join(dependencies)
            if dependencies is not None
            else "",  # Dependencies
            cloudpickle_path.name,
            cls_parameters_path.name,
            remote_cls_parameters_path.name,
        )
    )

    # Create necessary archive to register the operation on substra
    archive_path = operation_dir / "algo.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for filepath in operation_dir.glob("*"):
            if not filepath.name.endswith(".tar.gz"):
                tar.add(filepath, arcname=os.path.basename(filepath), recursive=True)
    return archive_path, description_path


def register_aggregation_node_op(
    client: substra.Client,
    remote_struct: RemoteStruct,
    permissions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    archive_path, description_path = prepare_substra_algo(
        remote_struct, dependencies=dependencies
    )

    key = client.add_algo(
        substra.sdk.schemas.AlgoSpec(
            name=uuid.uuid4().hex,
            description=description_path,
            file=archive_path,
            permissions=permissions,
            metadata=dict(),
            category=substra.sdk.schemas.AlgoCategory.aggregate,
        )
    )
    return key


def register_data_node_op(
    client: substra.Client,
    remote_struct: RemoteStruct,
    permissions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    archive_path, description_path = prepare_substra_algo(
        remote_struct, dependencies=dependencies
    )

    key = client.add_algo(
        substra.sdk.schemas.AlgoSpec(
            name=uuid.uuid4().hex,
            description=description_path,
            file=archive_path,
            permissions=permissions,
            metadata=dict(),
            category=substra.sdk.schemas.AlgoCategory.composite,
        )
    )

    return key
