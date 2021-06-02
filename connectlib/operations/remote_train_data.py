from abc import ABC, abstractmethod
from typing import Tuple, Any, TypeVar

SharedState = TypeVar("SharedState")
LocalState = TypeVar("LocalState")


class RemoteTrainDataOp(ABC):
    @abstractmethod
    def __call__(self,
                 x: Any,
                 y: Any,
                 shared_state: SharedState,
                 local_state: LocalState) -> Tuple[SharedState, LocalState]:
        raise NotImplementedError
