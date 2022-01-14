from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any


class Serializer(ABC):
    @staticmethod
    @abstractmethod
    def save(state: Any, path: Path):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: Path):
        raise NotImplementedError
