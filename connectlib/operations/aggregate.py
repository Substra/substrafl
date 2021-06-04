import substratools

from abc import abstractmethod
from typing import List, TypeVar, Type, Optional
from pathlib import Path

from connectlib.operations.serializers import PickleSerializer, Serializer

SharedState = TypeVar("SharedState")


class AggregateOp(substratools.AggregateAlgo):
    @abstractmethod
    def __call__(self, shared_states: Optional[List[SharedState]]) -> SharedState:
        raise NotImplementedError

    @property
    def shared_state_serializer(self) -> Type[Serializer]:
        return PickleSerializer

    # Substra methods
    def aggregate(self, inmodels: Optional[List[SharedState]], rank: int) -> SharedState:
        return self(inmodels)

    def load_model(self, path: str):
        self.shared_state_serializer.load(Path(path))

    def save_model(self, model: SharedState, path: str):
        self.shared_state_serializer.save(model, Path(path))
