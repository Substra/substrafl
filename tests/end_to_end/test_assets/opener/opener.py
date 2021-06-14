import shutil

import numpy as np
import substratools as tools


class Opener(tools.Opener):
    def get_X(self, folders):
        return np.random.randn(1000, 256)

    def get_y(self, folders):
        return np.random.randint(0, 2, size=(1000, 1))

    def save_predictions(self, y_pred: np.array, path):
        np.save(path, y_pred)
        shutil.move(str(path) + ".npy", path)

    def get_predictions(self, path):
        return np.load(path)

    def fake_X(self, n_samples=None):
        raise NotImplementedError

    def fake_y(self, n_samples=None):
        raise NotImplementedError
