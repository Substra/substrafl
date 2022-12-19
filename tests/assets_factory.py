import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from substra import Client
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions

DEFAULT_SUBSTRATOOLS_VERSION = (
    f"latest-nvidiacuda11.8.0-base-ubuntu22.04-python{sys.version_info.major}.{sys.version_info.minor}"
)

DEFAULT_SUBSTRATOOLS_DOCKER_IMAGE = f"ghcr.io/substra/substra-tools:{DEFAULT_SUBSTRATOOLS_VERSION}"

DEFAULT_OPENER_FILE = """
import os
import numpy as np
import substratools as tools

class NumpyOpener(tools.Opener):
    def get_data(self, folders):
        paths = []
        for folder in folders:
            paths += [
                os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == ".npy"
            ]
        data = np.concatenate([np.load(file, allow_pickle=True) for file in paths], axis=0)
        return (data[:, :-1], data[:, -1:])

    def fake_data(self, n_samples=None):
        # SubstraFL is never tested in hybrid mode
        pass
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
