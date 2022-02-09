import shutil

import numpy as np
import substratools as tools


class MnistOpener(tools.Opener):
    def get_X(self, folders):  # noqa: N802
        return folders

    def get_y(self, folders):
        return folders

    def save_predictions(self, y_pred, path):
        np.save(path, y_pred)
        shutil.move(str(path) + ".npy", path)

    def get_predictions(self, path):
        return np.load(path)

    def fake_X(self, n_samples=None):  # noqa: N802
        pass

    def fake_y(self, n_samples=None):
        pass
