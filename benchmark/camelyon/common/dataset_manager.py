import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

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
            subprocess.check_call(["gsutil", "-m", "cp", "-r", "gs://camelyon_0_5/data/", DEST_DIR])
        except subprocess.CalledProcessError:
            print(
                "Unable to download the data, please ensure that you have access to the following repo :\n"
                "https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview"  # noqa: E501
            )
            raise


def sub_sample_index(index_file: Path, sub_sampling: float, rng):
    """Random subsample from original index file of the Camelyon dataset.

    Args:
        index_file (Path): Csv file containing filename and target columns.
        sub_sampling (float): Ratio of sampling desired.
        rng (RandomGenerator): Numpy random generator.

    Returns:
        np.ndarray: 2D array of file name and associated target.
    """
    indexes = np.loadtxt(index_file, delimiter=",", dtype=str, skiprows=1)
    rng.shuffle(indexes)
    indexes = rng.choice(
        indexes,
        size=math.ceil(len(indexes) * sub_sampling),
        replace=False,
    )

    return indexes


def reset_data_folder():
    """Reset data folder to it's original state i.e. all the data in the IMG_DIR folder."""
    # Deleting old experiment folders
    old_folders = [x for x in os.listdir(DATA_DIR) if bool(re.search(r"(^train_\d$|^test$)", x))]

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
    np.savetxt(dest_folder / "index.csv", index, fmt="%s", delimiter=",")

    for f in index:
        file_name = f[0] + ".npy"
        os.link(IMG_DIR / file_name, dest_folder / file_name)


def split_dataset(n_centers, sub_sampling):
    """Generates separated folders containing the indexes in a csv file (index.csv) and images
    for train (n_centers) folders and one test folder. The size of the generated datasets is a
    fraction of the whole dataset (sub_sampling). The data is hard linked from the original folder
    (data/tiles_0.5mpp) to the generated folder.

    Args:
        n_centers (int): The number of training centers (i.e. the number of split for the training
            data).
        sub_sampling (float): the fraction of the data to use.

    Returns:
        List of train folders, test folder
    """

    rng = np.random.default_rng(42)

    # Train
    trains_indexes = np.array_split(
        sub_sample_index(TRAIN_INDEX_FILE, sub_sampling, rng),
        n_centers,
    )
    trains_folders = [DATA_DIR / f"train_{k}" for k in range(n_centers)]

    for folder, index in zip(trains_folders, trains_indexes):
        creates_data_folder(folder, index)

    # Test data
    test_indexes = sub_sample_index(TEST_INDEX_FILE, sub_sampling, rng)
    test_folder = DATA_DIR / "test"

    creates_data_folder(test_folder, test_indexes)

    return trains_folders, test_folder
