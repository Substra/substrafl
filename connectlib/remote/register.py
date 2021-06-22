import uuid
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

from connectlib.remote.methods import RemoteStruct

# TODO: change the base Image to a python image
DOCKERFILE_TEMPLATE = """
FROM substrafoundation/substra-tools:0.7.0

RUN apt update
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

RUN wget https://www.python.org/ftp/python/{6}/Python-{6}.tgz \
    && tar -xf Python-{6}.tgz \
    && cd Python-{6} \
    && ./configure --enable-optimizations \
    && make -j 12 \
    && make altinstall

# install dependencies
RUN python{0} -m pip install -U pip
RUN python{0} -m pip install six pytest
# Install connectlib
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
    with cls_parameters_path.open("w") as f:
        f.write(remote_struct.cls_parameters)

    # serialize remote cls parameters
    remote_cls_parameters_path = operation_dir / "remote_cls_parameters.json"
    with remote_cls_parameters_path.open("w") as f:
        f.write(remote_struct.remote_cls_parameters)

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
        f.write(ALGO.format(remote_struct.remote_cls_name))

    # Write description
    description_path = operation_dir / "description.md"
    with description_path.open("w") as f:
        f.write("# ConnnectLib Operation")

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
                cls_parameters_path.name,
                remote_cls_parameters_path.name,
                python_version(),
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


def register_aggregate_node_op(
    client: substra.Client,
    remote_struct: RemoteStruct,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    archive_path, description_path = prepare_substra_algo(
        remote_struct, dependencies=dependencies
    )

    key = client.add_aggregate_algo(
        substra.sdk.schemas.AggregateAlgoSpec(
            name=uuid.uuid4().hex,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )
    return key


def register_data_node_op(
    client: substra.Client,
    remote_struct: RemoteStruct,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    archive_path, description_path = prepare_substra_algo(
        remote_struct, dependencies=dependencies
    )

    key = client.add_composite_algo(
        substra.sdk.schemas.CompositeAlgoSpec(
            name=uuid.uuid4().hex,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )

    return key
