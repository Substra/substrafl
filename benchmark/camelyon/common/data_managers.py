import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset

# WARNING: the code is not tested on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Data:
    def __init__(self, paths: List[Path]):
        indexes = np.empty((0, 2), dtype=object)
        for path in paths:
            index_path = Path(path) / "index.csv"
            assert index_path.is_file(), "Wrong data sample, it must contain index.csv"
            ds_indexes = np.loadtxt(index_path, delimiter=",", dtype=object)
            ds_indexes[:, 0] = np.array(list(map(lambda x: str(os.path.join(path, x)), ds_indexes[:, 0])))

            def to_connectlib_path(x):
                return str(os.path.join(path, x))

            ds_indexes[:, 0] = np.vectorize(to_connectlib_path)(ds_indexes[:, 0])

            indexes = np.concatenate((indexes, ds_indexes))

        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)


class CamelyonDataset(Dataset):
    """Torch Dataset for the Camelyon data. Padding is done on the fly."""

    def __init__(self, data_indexes) -> None:
        self.data_indexes = (
            data_indexes if len(data_indexes.shape) > 1 else np.array(data_indexes).reshape(1, data_indexes.shape[0])
        )

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        """Get the needed item from index and preprocess them on the fly."""
        sample_file_path, target = self.data_indexes[index]
        x = torch.from_numpy(np.load(sample_file_path).astype(np.float32)[:, 3:]).to(device)

        y = torch.tensor(int(target == "Tumor")).type(torch.float32).to(device)

        missing_tiles = 10000 - x.shape[0]
        up = math.ceil(missing_tiles / 2)
        down = missing_tiles // 2

        x = F.pad(input=x, pad=(0, 0, up, down), mode="constant", value=0)

        return x, y
