import os
import cloudpickle
import tempfile
import shutil
import substra
import subprocess
import zipfile
import connectlib

from typing import Optional, List, Tuple
from pathlib import Path
from platform import python_version

from connectlib.operations.blueprint import Blueprint
from connectlib.nodes.pointers import (
    AggregatePointer,
    AlgoPointer,
    RemoteTestPointer,
    RemoteTrainPointer,
)

# TODO: change the base Image to a python image
DOCKERFILE_TEMPLATE = """
FROM substrafoundation/substra-tools:0.7.0

RUN apt-get update && apt-get install -y python{0}-dev

# install dependencies
RUN python{0} -m pip install -U pip

# Install connectlib
{1}

# Install dependencies
{2}

COPY ./algo.py /algo/algo.py
COPY ./{3} /algo/cloudpickle
COPY ./{4} /algo/parameters.json

ENTRYPOINT ["python{0}", "/algo/algo.py"]
"""

# TODO: Let user use their own seed if they want
# TODO: the SEED needs to be read/write from disk
# TODO: the SEED needs to be incremented for every execution
ALGO = """
import json
import cloudpickle

import substratools as tools

from pathlib import Path

if __name__ == "__main__":
    parameters_path = Path(__file__).parent / "parameters.json"

    with parameters_path.open("r") as f:
        parameters = json.load(f)

    cloudpickle_path = Path(__file__).parent / "cloudpickle"

    with cloudpickle_path.open("rb") as f:
        cls = cloudpickle.load(f)

    tools.algo.execute(cls(*parameters["args"], **parameters["kwargs"]))
"""


def prepare_blueprint(
    blueprint: Blueprint,
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

    # serialize parameters
    parameters_path = operation_dir / "parameters.json"
    with parameters_path.open("w") as f:
        f.write(blueprint.parameters)

    # serialize cls
    cloudpickle_path = operation_dir / "cloudpickle"
    with cloudpickle_path.open("wb") as f:
        cloudpickle.dump(blueprint.cls, f)

    # get python version
    # Required to select the correct version of python inside the docker Image
    # Cloudpickle will crash if we don't deserialize with the same major.minor
    python_major_minor = ".".join(python_version().split(".")[:2])

    # Build Connectlib wheel if needed
    if (connectlib.LIB_PATH.parent / "pyproject.toml").exists():
        subprocess.call(["poetry", "build"])
        shutil.copytree(connectlib.LIB_PATH.parent / "dist", operation_dir / "dist")

        # Get wheel name based on current version
        wheel_name = f"connectlib-{connectlib.__version__}-py3-none-any.whl"

        # Necessary command to install the wheel in the docker Image
        connectlib_install_cmd = (
            f"COPY ./dist /connectlib\n"
            f"RUN cd /connectlib && python{python_major_minor} -m pip install {wheel_name}"
        )
    else:
        # TODO: add private pypi
        connectlib_install_cmd = f"RUN python{python_major_minor} -m pip install connectlib=={connectlib.__version__}"

    # Write template to algo.py
    algo_path = operation_dir / "algo.py"
    with algo_path.open("w") as f:
        f.write(ALGO)

    # Write description
    description_path = operation_dir / "description.md"
    with description_path.open("w") as f:
        f.write(f"# ConnnectLib Operation: {blueprint.cls.__name__}")

    # Write dockerfile based on template
    dockerfile_path = operation_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                python_major_minor,
                connectlib_install_cmd,
                f"RUN python{python_major_minor} -m pip install " + " ".join(dependencies)
                if dependencies is not None
                else "",  # Dependencies
                cloudpickle_path.name,
                parameters_path.name,
            )
        )

    # Create necessary archive to register the operation on substra
    archive_path = operation_dir / "algo.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for filepath in operation_dir.glob("*[!.zip]"):
            if filepath.name == "dist":
                for dist in filepath.glob("*"):
                    z.write(dist, arcname=os.path.join("dist", dist.name))
            else:
                z.write(filepath, arcname=os.path.basename(filepath))

    return archive_path, description_path


def register_aggregate_op(
    client: substra.Client,
    blueprint: Blueprint,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> AggregatePointer:
    archive_path, description_path = prepare_blueprint(blueprint, dependencies=dependencies)

    key = client.add_aggregate_algo(
        substra.sdk.schemas.AggregateAlgoSpec(
            name=blueprint.cls.__name__,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )

    return AggregatePointer(key)


def _register_remote_data_op(
    client: substra.Client,
    blueprint: Blueprint,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    archive_path, description_path = prepare_blueprint(blueprint, dependencies=dependencies)

    key = client.add_composite_algo(
        substra.sdk.schemas.CompositeAlgoSpec(
            name=blueprint.cls.__name__,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )

    return key


def register_remote_train_op(
    client: substra.Client,
    blueprint: Blueprint,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> RemoteTrainPointer:
    key = _register_remote_data_op(client, blueprint, permisions, dependencies)
    return RemoteTrainPointer(key)


def register_remote_test_op(
    client: substra.Client,
    blueprint: Blueprint,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> RemoteTestPointer:
    key = _register_remote_data_op(client, blueprint, permisions, dependencies)
    return RemoteTestPointer(key)


def register_algo(
    client: substra.Client,
    blueprint: Blueprint,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> AlgoPointer:
    key = _register_remote_data_op(client, blueprint, permisions, dependencies)
    return AlgoPointer(key)


# TODO: add code to clean archives and other files
