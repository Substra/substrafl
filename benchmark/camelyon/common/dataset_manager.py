import os
import shutil
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parents[1]


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
