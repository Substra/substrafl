from abc import ABC
from typing import TypeVar

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    pass
