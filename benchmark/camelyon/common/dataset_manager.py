import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parents[1]


def fetch_camelyon(data_path: Path):
    """Download the data if needed from :
    https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview # noqa: E501
    """
    img_dir_path = data_path / "tiles_0.5mpp"
    index_file_path = data_path / "index.csv"
    if not img_dir_path.exists():
        img_dir_path.mkdir(parents=True)
        print("Downloading the data")
        try:
            shutil.copy(ROOT_DIR / "index.csv", index_file_path)
            indexes = np.loadtxt(index_file_path, delimiter=",", dtype=object)[1:, 0]
            subprocess.check_call(
                [
                    "gsutil",
                    "-o",
                    "GSUtil:parallel_process_count=1",
                    "-m",
                    "cp",
                    "-r",
                ]
                + ["gs://camelyon_0_5/data/tiles_0.5mpp/" + img + ".npy" for img in indexes]
                + [img_dir_path]
            )
        except subprocess.CalledProcessError:
            print(
                "Unable to download the data, please ensure that you have access to the following repo :\n"
                "https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview"  # noqa: E501
            )
            raise


def reset_data_folder(data_path: Path):
    """Reset data folder to it's original state i.e. all the data in the img_dir_path folder."""
    # Deleting old experiment folders
    old_folders = [x for x in os.listdir(data_path) if bool(re.search(r"(^train_\d$|^test_\d$|^test$|^train$)", x))]

    for folder in old_folders:
        shutil.rmtree(data_path / folder)


def creates_data_folder(img_dir_path, dest_folder, index_path):
    """Creates the `dest_folder` and hard link the data passed in index to it.

    Args:
        dest_folder (Path): Folder to put the needed data.
        index (np.ndarray): A 2D array (file_name, target) of the data wanted in the dest
        folder.
    """
    index = np.loadtxt(index_path, delimiter=",", dtype="<U16")[1:]

    dest_folder.mkdir(exist_ok=True)

    # Save our new index file
    index = index[np.argsort(index[:, 0])]

    for k, f in enumerate(index):
        file_name = f[0] + ".npy"
        out_name = f"{k}_{file_name}"
        os.link(img_dir_path / file_name, dest_folder / out_name)
        index[k, 0] = out_name

    np.savetxt(dest_folder / "index.csv", index, fmt="%s", delimiter=",")

    return dest_folder
