from enum import Enum
from typing import NewType

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
