import substratools

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, List, TypeVar
from pathlib import Path

from connectlib.operations import AggregateOp
from connectlib.nodes.pointers import SharedStatePointer

SharedState = TypeVar("SharedState")


@dataclass
class AggregationNode:
    node_id: str

    def submit(self,
               operation: AggregateOp,
               shared_state_pointer: Optional[SharedStatePointer] = None) \
            -> SharedStatePointer:
        pass


class AggregateExecutor(substratools.AggregateAlgo):
    def __init__(self, operation: AggregateOp):
        self.operation = operation

    def aggregate(self, inmodels: List[SharedState], rank) -> SharedState:
        return self.operation(inmodels)

    def load_model(self, path: str) -> SharedState:
        return self.operation.shared_state_serializer.load(Path(path))

    def save_model(self, model: SharedState, path: str):
        self.operation.shared_state_serializer.save(model, Path(path))
