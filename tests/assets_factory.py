import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from substra import Client
from substra.sdk.schemas import AlgoCategory
from substra.sdk.schemas import AlgoInputSpec
from substra.sdk.schemas import AlgoOutputSpec
from substra.sdk.schemas import AlgoSpec
from substra.sdk.schemas import AssetKind
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions

from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import OutputIdentifiers

DEFAULT_SUBSTRATOOLS_VERSION = (
    f"latest-nvidiacuda11.6.0-base-ubuntu20.04-python{sys.version_info.major}.{sys.version_info.minor}"
)

DEFAULT_SUBSTRATOOLS_DOCKER_IMAGE = f"gcr.io/connect-314908/connect-tools:{DEFAULT_SUBSTRATOOLS_VERSION}"

DEFAULT_METRICS_DOCKERFILE = f"""
FROM {DEFAULT_SUBSTRATOOLS_DOCKER_IMAGE}
COPY metrics.py .
ENTRYPOINT ["python3", "metrics.py"]
"""

DEFAULT_METRICS_FILE = f"""
import substratools as tools
import math
import numpy as np
class AccuracyMetric(tools.Metrics):
    def score(self, inputs, outputs):
        y_true = inputs['{InputIdentifiers.y}']
        y_pred = self.load_predictions(inputs['{InputIdentifiers.predictions}'])
        tools.save_performance({{}}, outputs['{OutputIdentifiers.performance}'])

    def load_predictions(self, path):
        return np.load(path)


if __name__ == "__main__":
    tools.metrics.execute(AccuracyMetric())
"""

DEFAULT_OPENER_FILE = """
import os
import shutil
import numpy as np
import substratools as tools

class NumpyOpener(tools.Opener):
    def get_X(self, folders):
        data = self._get_data(folders)
        return self._get_X(data)

    def get_y(self, folders):
        data = self._get_data(folders)
        return self._get_y(data)

    def fake_X(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_X(data)

    def fake_y(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_y(data)

    @classmethod
    def _get_X(cls, data):
        return data[:, :-1]

    @classmethod
    def _get_y(cls, data):
        return data[:, -1:]

    @classmethod
    def _fake_data(cls, n_samples=None, n_col=3):
        return np.random.uniform(0, 1, (n_samples, n_col))

    @classmethod
    def _get_data(cls, folders):
        paths = []
        for folder in folders:
            paths += [
                os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == ".npy"
            ]
        return np.concatenate([np.load(file, allow_pickle=True) for file in paths], axis=0)
"""


def generic_description(name: str, tmp_folder: Path) -> Path:
    """Creates a generic description.md file starting with the named
    passed as argument in the tmp_folder folder.

    Args:
        name (str): The name of you description.
        tmp_folder (Path): The folder where this file will be created.

    Returns:
        Path: The path of the created file (ends with /description.md)
    """
    description_path = tmp_folder / "description.md"
    description_content = f"# {name}"
    description_path.write_text(description_content)

    return description_path


def add_numpy_datasets(datasets_permissions: Permissions, clients: List[Client], tmp_folder: Path) -> List[str]:
    """Add a numpy opener with the corresponding datasets_permissions to the clients.
    Pairs are created based on their indexes.
    During the process a description.md and a opener.py files are created in the tmp_folder.

    Args:
        datasets_permissions (Permissions): The wanted permissions for each datasets.
        client (Client): A substra client.
        tmp_folder (Path): A folder where a description.md file sill be created.

    Asserts:
        len(clients)==len(datasets_permissions)

    Returns:
        str: The dataset keys returned by substra to clients.
    """
    assert len(clients) == len(datasets_permissions), (
        "clients and datasets_permissions must have the same length as they are associated in pairs based "
        "on their indexes."
    )

    tmp_opener_dir = Path(tempfile.mkdtemp(dir=tmp_folder))

    description_path = generic_description(name="Numpy", tmp_folder=tmp_opener_dir)
    opener_path = tmp_opener_dir / "opener.py"
    opener_path.write_text(DEFAULT_OPENER_FILE)

    keys = []
    for client, permissions in zip(clients, datasets_permissions):
        dataset_spec = DatasetSpec(
            name="Numpy Opener",
            type="npy",
            data_opener=opener_path,
            description=description_path,
            permissions=permissions,
            logs_permission=permissions,
        )

        keys.append(client.add_dataset(dataset_spec))

    return keys


def add_numpy_samples(
    contents: List[np.ndarray],
    dataset_keys: List[str],
    clients: List[Client],
    tmp_folder: Path,
) -> List[str]:
    """Each client will associated one element of the contents list (pairs are made according to their respective index)
    with the corresponding dataset key and submit it to substra. The content will be stored in the tmp_folder in the
    process. All the samples will be added with the argument `test_only=False`

    Args:
        contents (List[np.ndarray]): Numpy contents to add to each organization.
        dataset_keys (str): Substra dataset_keys accessible per each client.
        clients (List[Client]):  Substra clients used to add each content to the organization.
        tmp_folder (Path): The folder where the numpy data will be stored to be added to substra.

    Asserts:
        The number of clients, the number of organizations and the number of dataset keys must be the same.

    Returns:
        List[str]: A list of data_samples keys.
    """

    assert len(clients) == len(contents) == len(dataset_keys), (
        "The number of passed contents, clients and dataset_keys must be the same as each client will submit the "
        "content and dataset with the same index than his."
    )

    keys = []

    for client, content, dataset_key in zip(clients, contents, dataset_keys):
        data_sample_folder = Path(tempfile.mkdtemp(dir=tmp_folder))
        data_sample_file = data_sample_folder / "data.npy"
        np.save(data_sample_file, content)

        data_sample_spec = DataSampleSpec(
            data_manager_keys=[dataset_key],
            test_only=False,
            path=data_sample_folder,
        )

        keys.append(
            client.add_data_sample(
                data_sample_spec,
                local=True,
            )
        )

    return keys


def metric_archive(metric_path: Path, dockerfile_path: Path, tmp_folder: Path) -> Path:
    """Creates a tar.gz archive in the tmp_folder folder from the metric_path and the dockerfile_path.
    Whatever the named of the passed files, they will be named metric.py and Dockerfile in the archive.

    Args:
        metric_path (Path): The path to a python file where a substra compatible class is written.
        dockerfile_path (Path): The dockerfile associated to the metric.
        tmp_folder (Path): The folder where the archive will be written.

    Returns:
        Path: the archive path (ends with /metric.tar.gz)
    """

    archive_path = tmp_folder / "metric.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dockerfile_path, arcname="Dockerfile")
        tar.add(metric_path, arcname="metrics.py")
    return archive_path


def metric_dockerfile(tmp_folder: Path) -> Path:
    """Creates a generic python dockerfile from DEFAULT_SUBSTRATOOLS_VERSION docker image.
    The entry point runs a metric.py file.

    Args:
        tmp_folder (Path): The folder where the dockerfile will be written.

    Returns:
        Path: The path of the dockerfile (ends with /Dockerfile)
    """
    dockerfile_path = tmp_folder / "Dockerfile"
    dockerfile_path.write_text(DEFAULT_METRICS_DOCKERFILE)
    return dockerfile_path


def add_python_metric(
    python_formula: str,
    name: str,
    permissions: Permissions,
    client: Client,
    tmp_folder: Path,
):
    """Add the given numpy formula as a substra metric with the given name and permissions
    thanks to the specified client.
    All the necessary files will be created in the tmp_folder.

    Args:
        python_formula (str): A pure python formula passed in a string.
            This formula must be based on y_pred and y_true variable which are both numpy array.
            The formula must return a float python object.
            E.g.: The Accuracy formula would be: (y_pred==y_true).sum()/y_true.shape[0] as no numpy function
            is used there.
            The math module is also imported as is. So all math.cos, math.sqrt functions can be called.
        name (str): The name of your metric.
        permissions (Permissions): The substra permissions for your metric.
        client (Client): The substra client that will add the metric.
        tmp_folder (Path): The folder used to create all the needed intermediate files.

    Returns:
        str: The metric key returned by substra to the client.
    """
    metric_folder = Path(tempfile.mkdtemp(dir=tmp_folder))

    description = generic_description(name=name, tmp_folder=metric_folder)

    metric_file = metric_folder / "metric.py"
    metric_file.write_text(DEFAULT_METRICS_FILE.format(python_formula))

    dockerfile_file = metric_dockerfile(tmp_folder)
    archive = metric_archive(
        metric_path=metric_file,
        dockerfile_path=dockerfile_file,
        tmp_folder=metric_folder,
    )
    metric_spec = AlgoSpec(
        category=AlgoCategory.metric,
        name=name,
        inputs=[
            AlgoInputSpec(
                identifier=InputIdentifiers.datasamples,
                kind=AssetKind.data_sample.value,
                optional=False,
                multiple=True,
            ),
            AlgoInputSpec(
                identifier=InputIdentifiers.opener, kind=AssetKind.data_manager.value, optional=False, multiple=False
            ),
            AlgoInputSpec(
                identifier=InputIdentifiers.predictions, kind=AssetKind.model.value, optional=False, multiple=False
            ),
        ],
        outputs=[
            AlgoOutputSpec(identifier=OutputIdentifiers.performance, kind=AssetKind.performance.value, multiple=False)
        ],
        description=description,
        file=archive,
        permissions=permissions,
    )
    key = client.add_algo(metric_spec)
    return key


def linear_data(n_col: int = 3, n_samples: int = 11, weights_seed: int = 42, noise_seed: int = 12) -> np.ndarray:
    """Generate 2D dataset fo n_col and n_samples. The data are linearly linked with less than
    10% of noise.

    Args:
        n_col (int, Optional): The wished number of column in the dataset. Defaults to 3.
        n_samples (int, Optional): The wished number of samples in the dataset. Defaults to 11.
        weights_seed (int, Optional): Used to set the weights. This ensure the reproducibility of the relation
        between the features.
        noise_seed (int, Optional): Used to set the noise. This ensure the reproducibility of the noise added.

    Returns:
        np.ndarray: A 2D (n_samples, n_col) np.ndarray
    """
    np.random.seed(weights_seed)
    random_content = np.random.uniform(0, 1, (n_samples, n_col - 1))

    np.random.seed(noise_seed)
    noise = np.random.normal(0, 0.01, (n_samples, n_col - 1))

    target = (random_content + noise).sum(axis=1)

    dataset = np.c_[random_content, target]

    return dataset
