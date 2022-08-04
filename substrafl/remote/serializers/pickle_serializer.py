import pickle
from pathlib import Path
from typing import Any

from substrafl.remote.serializers.serializer import Serializer


class PickleSerializer(Serializer):
    @staticmethod
    def save(state: Any, path: Path):
        """Pickle the state to path

        Args:
            state (typing.Any): state to save
            path (pathlib.Path): path where to save it
        """
        with Path(path).open("wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: Path) -> Any:
        """Load an object from a path
        using pickle.load

        Args:
            path (pathlib.Path): path to the saved file

        Returns:
            Any: loaded state
        """
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        return state
