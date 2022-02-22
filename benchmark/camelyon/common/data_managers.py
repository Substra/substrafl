import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# WARNING: the code is not tested on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataLoaderWithMemory:
    """Infinite data loader from torch dataset.
    Args:
        dataset (torchDataset): Torch dataset to instantiate the data loader from.
        batch_size (int, optional): Number of sample to be read from the disk per batch. Defaults to 32.
        shuffle (bool, optional): Shuffle the dataset before generating a new epoch. Defaults to True.
        drop_last (bool, optional): Drop the last batch at the end of the epoch if ti's size is smaller
            than the batch size. Defaults to False.
        num_workers (int, optional): Number of torch worker to use. Defaults to 0.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ):

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def get_samples(self):
        try:
            X, y = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            X, y = next(self._iterator)
        return X, y


class CamelyonDataset(Dataset):
    """Torch Dataset for the Camelyon data. Padding is done on the fly."""

    def __init__(self, data_indexes, img_path) -> None:
        self.data_indexes = data_indexes
        self.img_path = Path(img_path)

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        """Get the needed item from index and preprocess them on the fly."""
        sample_file_path, target = self.data_indexes[index]
        x = torch.from_numpy(np.load(self.img_path / (sample_file_path + ".npy")).astype(np.float32)[:, 3:]).to(device)

        y = torch.tensor(int(target == "Tumor")).type(torch.float32).to(device)

        missing_tiles = 10000 - x.shape[0]
        up = math.ceil(missing_tiles / 2)
        down = missing_tiles // 2

        x = F.pad(input=x, pad=(0, 0, up, down), mode="constant", value=0)

        return x, y
