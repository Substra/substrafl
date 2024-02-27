from abc import abstractmethod
from typing import Any
from typing import List
from typing import Protocol
from typing import runtime_checkable

import substra

from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.operations import RemoteOperation


@runtime_checkable
class TrainDataNodeProtocol(Protocol):
    organization_id: str
    data_manager_key: str
    data_sample_keys: List[str]

    @abstractmethod
    def init_states(self, *args, **kwargs) -> LocalStateRef:
        pass

    @abstractmethod
    def update_states(self, operation: RemoteDataOperation, *args, **kwargs) -> (LocalStateRef, Any):
        pass

    @abstractmethod
    def register_operations(self, client: substra.Client, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def summary(self) -> dict:
        pass


@runtime_checkable
class TestDataNodeProtocol(Protocol):
    organization_id: str
    data_manager_key: str
    data_sample_keys: List[str]

    @abstractmethod
    def update_states(self, operation: RemoteDataOperation, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def register_operations(self, client: substra.Client, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def summary(self) -> dict:
        pass


@runtime_checkable
class AggregationNodeProtocol(Protocol):
    organization_id: str

    @abstractmethod
    def update_states(self, operation: RemoteOperation, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def register_operations(self, client: substra.Client, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def summary(self) -> dict:
        pass
