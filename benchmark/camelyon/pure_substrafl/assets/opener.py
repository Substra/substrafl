import logging
from pathlib import Path
from typing import List

import numpy as np
import substratools as tools

logger = logging.getLogger(__name__)


# Duplicated of common.data_manager.Data for dependencies reasons
class Data:
    def __init__(self, paths: List[Path]):
        indexes = list()
        for path in paths:
            index_path = Path(path) / "index.csv"
            assert index_path.is_file(), "Wrong data sample, it must contain index.csv"
            ds_indexes = np.loadtxt(index_path, delimiter=",", dtype=object)
            ds_indexes[:, 0] = np.array([str(Path(path) / x) for x in ds_indexes[:, 0]])
            indexes.extend(ds_indexes)

        self._indexes = np.asarray(indexes, dtype=object)

    @property
    def indexes(self):
        return self._indexes

    def __len__(self):
        return len(self.indexes)


class MnistOpener(tools.Opener):
    def get_X(self, folders):  # noqa: N802
        return Data(folders)

    def get_y(self, folders):
        data = self.get_X(folders)
        y_true = np.array([int(x == "Tumor") for x in data.indexes[:, 1]])
        return y_true

    def get_predictions(self):
        pass

    def save_predictions(self):
        pass

    def fake_X(self, n_samples=None):  # noqa: N802
        pass

    def fake_y(self, n_samples=None):
        pass
