import os
import shutil
import substra
import subprocess
import zipfile
import connectlib

from typing import Optional, List
from platform import python_version

from .algo import RegisteredAlgo

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


def add_algo(
    client: substra.Client,
    registered_algo: RegisteredAlgo,
    permisions: substra.sdk.schemas.Permissions,
    dependencies: Optional[List[str]] = None,
) -> str:
    python_major_minor = ".".join(python_version().split(".")[:2])

    if (connectlib.LIB_PATH.parent / "pyproject.toml").exists():
        subprocess.call(["poetry", "build"])
        shutil.copytree(connectlib.LIB_PATH.parent / "dist", registered_algo.algo_dir / "dist")

        wheel_name = f"connectlib-{connectlib.__version__}-py3-none-any.whl"
        connectlib_install_cmd = (
            f"COPY ./dist /connectlib\n"
            f"RUN cd /connectlib && python{python_major_minor} -m pip install {wheel_name}"
        )
    else:
        connectlib_install_cmd = f"RUN python{python_major_minor} -m pip install connectlib=={connectlib.__version__}"

    algo_path = registered_algo.algo_dir / "algo.py"
    with algo_path.open("w") as f:
        f.write(ALGO)

    description_path = registered_algo.algo_dir / "description.md"
    with description_path.open("w") as f:
        f.write(f"# ConnnectLib Algo: {registered_algo.algo_cls.__name__}")

    dockerfile_path = registered_algo.algo_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                python_major_minor,
                connectlib_install_cmd,
                "RUN python{python_major_minor} -m pip install " + " ".join(dependencies)
                if dependencies is not None
                else "",  # Dependencies
                registered_algo.cloudpickle_path.name,
                registered_algo.parameters_path.name,
            )
        )

    archive_path = registered_algo.algo_dir / "algo.zip"

    with zipfile.ZipFile(archive_path, "w") as z:
        for filepath in registered_algo.algo_dir.glob("*[!.zip]"):
            if filepath.name == "dist":
                for dist in filepath.glob("*"):
                    z.write(dist, arcname=os.path.join("dist", dist.name))
            else:
                z.write(filepath, arcname=os.path.basename(filepath))

    print(registered_algo.algo_dir)

    algo_key = client.add_algo(
        substra.sdk.schemas.AlgoSpec(
            name=registered_algo.algo_cls.__name__,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )

    return algo_key
