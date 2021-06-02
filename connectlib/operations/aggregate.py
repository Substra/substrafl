from abc import ABC, abstractmethod
from typing import List, TypeVar, Type

from connectlib.operations.serializers import PickleSerializer

SharedState = TypeVar("SharedState")


class AggregateOp(ABC):
    @property
    def shared_state_serializer(self) -> Type[PickleSerializer]:
        return PickleSerializer

    @abstractmethod
    def __call__(self, shared_states: List[SharedState]) -> SharedState:
        raise NotImplementedError


class AvgOp(AggregateOp):
    def __init__(self, arg1):
        self.arg1 = arg1


    def __call__(self, shared_states: List[Dict[str, np.array]]) -> Dict[str, np.array]:
        pass
