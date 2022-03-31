from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any


class Serializer(ABC):
    @staticmethod
    @abstractmethod
    def save(state: Any, path: Path):
        """Save the state to the path

        Args:
            state (typing.Any): state to save
            path (pathlib.Path): path where to save it
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: Path):
        raise NotImplementedError
