from typing import Generic, TypeVar
from dataclasses import dataclass

from connectlib.algorithms import Algo

AlgoType = TypeVar("AlgoType", bound=Algo)


@dataclass
class AlgoRef(Generic[AlgoType]):
    key: str
