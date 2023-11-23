from enum import Enum
from typing import Any
from typing import List
from typing import NewType

import pydantic

OperationKey = NewType("OperationKey", str)


class InputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    predictions = "predictions"
    opener = "opener"
    datasamples = "datasamples"
    rank = "rank"
    X = "X"
    y = "y"


class OutputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    predictions = "predictions"


class _PrettyJsonBaseModel(pydantic.BaseModel):
    """Base model configuration for pretty representation"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # pretty print
    def __str__(self):
        return self.model_dump_json(indent=4)

    def __repr__(self):
        return self.model_dump_json(indent=4)


class SimuPerformancesMemory(_PrettyJsonBaseModel):
    """Performances of a simulated experiment"""

    worker: List[str] = []
    round_idx: List[int] = []
    identifier: List[str] = []
    performance: List[float] = []

    def __add__(self, other):
        return SimuPerformancesMemory(
            worker=self.worker + other.worker,
            round_idx=self.round_idx + other.round_idx,
            identifier=self.identifier + other.identifier,
            performance=self.performance + other.performance,
        )


class SimuStatesMemory(_PrettyJsonBaseModel):
    """Intermediate states of a simulated experiment"""

    worker: List[str] = []
    round_idx: List[int] = []
    state: List[Any] = []

    @pydantic.field_serializer("state")
    def serialize_state(self, state: Any, _info):
        return [str(s) for s in state]

    def __add__(self, other):
        return SimuStatesMemory(
            worker=self.worker + other.worker,
            round_idx=self.round_idx + other.round_idx,
            state=self.state + other.state,
        )
