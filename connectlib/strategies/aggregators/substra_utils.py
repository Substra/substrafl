import os
import shutil
import tempfile

import substra
import subprocess
import zipfile
import connectlib

from platform import python_version

from .aggregator import Aggregator

DOCKERFILE_TEMPLATE = """
FROM substrafoundation/substra-tools:0.7.0

RUN apt-get update && apt-get install -y python{0}-dev

# install dependencies
RUN python{0} -m pip install -U pip

# Install connectlib
{1}

COPY ./algo.py /algo/algo.py

ENTRYPOINT ["python{0}", "/algo/algo.py"]
"""

ALGO = """
import substratools as tools

from connectlib.strategies.aggregators import {}

if __name__ == "__main__":
    tools.algo.execute({}())
"""


def add_aggregator(
    client: substra.Client,
    aggregator_cls: type(Aggregator),
    permisions: substra.sdk.schemas.Permissions,
) -> str:
    algo_dir = tempfile.mkdtemp()

    python_major_minor = ".".join(python_version().split(".")[:2])

    if (connectlib.LIB_PATH.parent / "pyproject.toml").exists():
        subprocess.call(["poetry", "build"])
        shutil.copytree(connectlib.LIB_PATH.parent / "dist", algo_dir / "dist")

        wheel_name = f"connectlib-{connectlib.__version__}-py3-none-any.whl"
        connectlib_install_cmd = (
            f"COPY ./dist /connectlib\n"
            f"RUN cd /connectlib && python{python_major_minor} -m pip install {wheel_name}"
        )
    else:
        connectlib_install_cmd = f"RUN python{python_major_minor} -m pip install connectlib=={connectlib.__version__}"

    algo_path = algo_dir / "algo.py"
    with algo_path.open("w") as f:
        f.write(ALGO.format(aggregator_cls.__name__))

    description_path = algo_dir / "description.md"
    with description_path.open("w") as f:
        f.write(f"# ConnnectLib Aggregator Algo: {aggregator_cls.__name__}")

    dockerfile_path = algo_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                python_major_minor,
                connectlib_install_cmd,
            )
        )

    archive_path = algo_dir / "algo.zip"

    with zipfile.ZipFile(archive_path, "w") as z:
        for filepath in algo_dir.glob("*[!.zip]"):
            if filepath.name == "dist":
                for dist in filepath.glob("*"):
                    z.write(dist, arcname=os.path.join("dist", dist.name))
            else:
                z.write(filepath, arcname=os.path.basename(filepath))

    print(algo_dir)

    aggregator_key = client.add_algo(
        substra.sdk.schemas.AlgoSpec(
            name=aggregator_cls.__name__,
            description=description_path,
            file=archive_path,
            permissions=permisions,
            metadata=dict(),
        )
    )

    return aggregator_key
