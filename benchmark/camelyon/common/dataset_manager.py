import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parents[1]

FILE_LIST = [
    "normal_001.tif.npy",
    "normal_002.tif.npy",
    "tumor_106.tif.npy",
    "tumor_108.tif.npy",
]
URL = "https://zenodo.org/api/files/d5520b8a-6a5b-41ef-bde0-247dbdded101/"
N_PARALLEL = 4


def fetch_camelyon(data_path: Path):
    index_filename = "index.csv"
    if not (data_path / index_filename).exists():
        print("Downloading the dataset (~400Mb), this may take a few minutes.")
        subprocess.run(
            ["wget", f"{URL}index.csv"],
            check=True,
            cwd=data_path,
        )
        subprocess.run(
            ["parallel", "-j", str(N_PARALLEL), "wget", ":::"] + [f"{URL}{filename}" for filename in FILE_LIST],
            check=True,
            cwd=data_path / "tiles_0.5mpp",
        )


def reset_data_folder(data_path: Path):
    """Reset data folder to it's original state i.e. all the data in the img_dir_path folder."""
    # Deleting old experiment folders
    if data_path.is_dir():
        shutil.rmtree(data_path)


def creates_data_folder(img_dir_path, dest_folder, index_path):
    """Creates the `dest_folder` and hard link the data passed in index to it.

    Args:
        dest_folder (Path): Folder to put the needed data.
        index (np.ndarray): A 2D array (file_name, target) of the data wanted in the dest
        folder.
    """
    index = np.loadtxt(index_path, delimiter=",", dtype="<U32")[1:]

    dest_folder.mkdir(exist_ok=True, parents=True)

    for idx, sample_arr in enumerate(index):
        file_name = sample_arr[0] + ".tif.npy"
        out_name = f"{idx}_{file_name}"
        os.link(img_dir_path / file_name, dest_folder / out_name)
        index[idx, 0] = out_name

    np.savetxt(dest_folder / "index.csv", index, fmt="%s", delimiter=",")

    return dest_folder
