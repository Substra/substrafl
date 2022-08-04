"""Dataclasses describing the operations
to execute on the remote.
"""
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Optional

from substrafl.remote.remote_struct import RemoteStruct


@dataclass
class AggregateOperation:
    """Aggregation operation"""

    remote_struct: RemoteStruct
    shared_states: Optional[List]


@dataclass
class DataOperation:
    """Data operation"""

    remote_struct: RemoteStruct
    data_samples: List[str]
    shared_state: Any
