import pickle
from pathlib import Path
from typing import Any

from .serializer import Serializer


class PickleSerializer(Serializer):
    @staticmethod
    def save(state: Any, path: Path):
        with Path(path).open("wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: Path):
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        return state
