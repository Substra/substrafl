import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict

from connectlib.algorithms import Algo
from connectlib.operations.blueprint import blueprint


@blueprint
class MyAlgo(Algo):
    def __init__(self):
        self._shared_state = {"test": np.random.randn(8, 16)}

    def preprocessing(
        self, x: np.array, y: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
        return x, y

    def perform_update(self, x: np.array, y: np.array):
        pass

    def test(self, x: np.array):
        return np.random.randint(0, 2, size=(len(x), 1))

    @property
    def shared_state(self) -> Dict[str, np.array]:
        return self._shared_state

    @shared_state.setter
    def shared_state(self, shared_state: Dict[str, np.array]):
        self._shared_state = shared_state

    def load(self, path: Path):
        pass

    def save(self, path: Path):
        assert path.parent.exists()
        with path.open("w") as f:
            f.write("test")
