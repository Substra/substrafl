import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from .utils import parse_params

DEST_DIR = Path(__file__).parents[1]
DATA_DIR = DEST_DIR / "data"

IMG_DIR = DATA_DIR / "tiles_0.5mpp"

TRAIN_INDEX_FILE = DATA_DIR / "train_data_index.csv"
TEST_INDEX_FILE = DATA_DIR / "test_data_index.csv"


def fetch_camelyon():
    """Download the data if needed from :
    https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview # noqa: E501
    """
    if not IMG_DIR.exists():
        print("Downloading the data (this could take several minutes).")
        try:
            subprocess.check_call(
                [
                    "gsutil",
                    "-o",
                    "GSUtil:parallel_process_count=1",
                    "-m",
                    "cp",
                    "-r",
                    "gs://camelyon_0_5/data/",
                    DEST_DIR,
                ]
            )
        except subprocess.CalledProcessError:
            print(
                "Unable to download the data, please ensure that you have access to the following repo :\n"
                "https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview"  # noqa: E501
            )
            raise


def reset_data_folder():
    """Reset data folder to it's original state i.e. all the data in the IMG_DIR folder."""
    # Deleting old experiment folders
    old_folders = [x for x in os.listdir(DATA_DIR) if bool(re.search(r"(^train_\d$|^test_\d$|^test$)", x))]

    for folder in old_folders:
        shutil.rmtree(DATA_DIR / folder)


def creates_data_folder(dest_folder, index):
    """Creates the `dest_folder` and hard link the data passed in index to it.

    Args:
        dest_folder (Path): Folder to put the needed data.
        index (np.ndarray): A 2D array (file_name, target) of the data wanted in the dest
        folder.
    """

    dest_folder.mkdir(exist_ok=True)

    # Save our new index file
    index = index[np.argsort(index[:, 0])]

    for k, f in enumerate(index):
        file_name = f[0] + ".npy"
        out_name = f"{k}_{file_name}"
        os.link(IMG_DIR / file_name, dest_folder / out_name)
        index[k, 0] = out_name

    np.savetxt(dest_folder / "index.csv", index, fmt="%s", delimiter=",")


def creates_data_folders(
    n_centers: Optional[int],
    sub_sampling: Optional[float],
    data_samples_size: Optional[List[Dict[str, int]]],
    batch_size: int,
    kind: str = "train",
):
    """Generates separated folders containing the indexes in a csv file (index.csv) and images
    for train (n_centers) folders and one test folder. The size of the generated datasets is either a
    fraction of the whole dataset (sub_sampling) or of the exact size specified in data_samples_size.
    The data is hard linked from the original folder
    (data/tiles_0.5mpp) to the generated folder."""
    index_file = TRAIN_INDEX_FILE if kind == "train" else TEST_INDEX_FILE
    rng = np.random.default_rng(42)
    index = np.loadtxt(index_file, delimiter=",", dtype="<U64", skiprows=1)
    rng.shuffle(index)

    sizes = (
        [sizes[kind] for sizes in data_samples_size]
        if data_samples_size
        else [math.ceil(len(index) * sub_sampling)] * n_centers
    )

    indexes = [
        rng.choice(
            index,
            size=size,
            replace=True,
        )
        for size in sizes
        if size > 0
    ]

    folders = [DATA_DIR / f"{kind}_{k}" for k in range(len(indexes))]

    for folder, index in zip(folders, indexes):
        creates_data_folder(folder, index)

    # Check on the number of samples
    for folder in folders:
        len_data = len((Path(folder) / "index.csv").read_text().splitlines())
        if len_data < batch_size:
            raise ValueError(
                "The length of the dataset is smaller than the batch size, not allowed as it"
                "skews the benchmark results (the batch size gets automatically adjusted in that case)."
            )

    return folders


if __name__ == "__main__":
    fetch_camelyon()
    params = parse_params()
    reset_data_folder()
    train_folders = creates_data_folders(
        n_centers=params.get("n_centers"),
        sub_sampling=params.get("sub_sampling"),
        data_samples_size=params.get("data_samples_size"),
        batch_size=params.get("batch_size"),
        kind="train",
    )
    test_folders = creates_data_folders(
        n_centers=params.get("n_centers"),
        sub_sampling=params.get("sub_sampling"),
        data_samples_size=params.get("data_samples_size"),
        batch_size=params.get("batch_size"),
        kind="test",
    )
