import shutil
from pathlib import Path

import numpy as np
import substratools as tools


class Data:
    def __init__(self, path: Path):
        index_path = Path(path) / "index.csv"
        assert index_path.is_file(), "Wrong data sample, it must contain index.csv"
        self.indexes = np.loadtxt(index_path, delimiter=",", dtype=str)
        self.indexes = (
            self.indexes[np.argsort(self.indexes[:, 0])]
            if len(self.indexes.shape) > 1
            else np.array(self.indexes).reshape(1, self.indexes.shape[0])
        )
        self.path = path

    def __len__(self):
        return len(self.indexes)


class MnistOpener(tools.Opener):
    def get_X(self, folders):  # noqa: N802
        assert len(folders) == 1, "Only one data sample accepted here"
        return Data(folders[0])

    def get_y(self, folders):
        data = self.get_X(folders)
        y_true = np.array([int(x == "Tumor") for x in data.indexes[:, 1]])
        return y_true

    def save_predictions(self, y_pred, path):
        np.save(path, y_pred)
        shutil.move(str(path) + ".npy", path)

    def get_predictions(self, path):
        return np.load(path)

    def fake_X(self, n_samples=None):  # noqa: N802
        pass

    def fake_y(self, n_samples=None):
        pass
