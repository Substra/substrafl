import substratools

from abc import abstractmethod
from typing import Tuple, Any, TypeVar, Optional, Type
from pathlib import Path

from connectlib.operations.serializers import PickleSerializer, Serializer

SharedState = TypeVar("SharedState")
LocalState = TypeVar("LocalState")


class RemoteTestOp(substratools.CompositeAlgo):
    @abstractmethod
    def __call__(
        self, x: Any, local_state: Optional[LocalState], shared_state: Optional[SharedState]
    ) -> Tuple[LocalState, SharedState]:
        raise NotImplementedError

    @property
    def local_state_serializer(self) -> Type[Serializer]:
        return PickleSerializer

    @property
    def shared_state_serializer(self) -> Type[Serializer]:
        return PickleSerializer

    # Substra methods
    def train(
        self,
        X: Any,
        y: Any,
        head_model: Optional[LocalState],
        trunk_model: Optional[SharedState],
        rank: int,
    ):
        raise NotImplementedError

    def predict(
        self, X: Any, head_model: Optional[LocalState], trunk_model: Optional[SharedState]
    ):
        return self(X, local_state=head_model, shared_state=trunk_model)

    def load_trunk_model(self, path: str) -> SharedState:
        return self.shared_state_serializer.load(Path(path))

    def save_trunk_model(self, model: SharedState, path: str):
        self.shared_state_serializer.save(model, Path(path))

    def load_head_model(self, path: str) -> LocalState:
        return self.local_state_serializer.load(Path(path))

    def save_head_model(self, model: LocalState, path: str):
        self.local_state_serializer.save(model, Path(path))
