from abc import ABC, abstractmethod
from typing import Tuple, Any

from connectlib.operations.states import State


class RemoteTestDataOp(ABC):
    @abstractmethod
    def __call__(self,
                 x: Any,
                 shared_state: State,
                 local_state: State) -> Tuple[State, State]:
        raise NotImplementedError
