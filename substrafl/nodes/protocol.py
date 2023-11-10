from typing import Any
from typing import List
from typing import Protocol
from typing import runtime_checkable

from substrafl.nodes.references.local_state import LocalStateRef


@runtime_checkable
class TrainDataNodeProtocol(Protocol):
    organization_id: str
    data_manager_key: str
    data_sample_keys: List[str]

    def init_states(self, *args, **kwargs) -> LocalStateRef:
        pass

    def update_states(self, *args, **kwargs) -> (LocalStateRef, Any):
        pass

    def register_operations(self, *args, **kwargs) -> Any:
        pass

    def summary(self) -> dict:
        pass


@runtime_checkable
class TestDataNodeProtocol(Protocol):
    organization_id: str
    data_manager_key: str
    data_sample_keys: List[str]

    def update_states(self, *args, **kwargs) -> None:
        pass

    def register_operations(self, *args, **kwargs) -> Any:
        pass

    def summary(self) -> dict:
        pass


@runtime_checkable
class AggregationNodeProtocol(Protocol):
    organization_id: str

    def update_states(self, *args, **kwargs) -> Any:
        pass

    def register_operations(self, *args, **kwargs) -> Any:
        pass

    def summary(self) -> dict:
        pass
