from typing import List

import pydantic

from substrafl.compute_plan_builder import ComputePlanBuilder


class _Model(pydantic.BaseModel):
    """Base model configuration"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # pretty print
    def __str__(self):
        return self.model_dump_json(indent=4)

    def __repr__(self):
        return self.model_dump_json(indent=4)


class SimulationPerformances(_Model):
    """Performances of a simulated experiment"""

    worker: List[str] = []
    round_idx: List[int] = []
    identifier: List[str] = []
    performance: List[float] = []

    def __add__(self, other):
        return SimulationPerformances(
            worker=self.worker + other.worker,
            round_idx=self.round_idx + other.round_idx,
            identifier=self.identifier + other.identifier,
            performance=self.performance + other.performance,
        )


class SimulationIntermediateStates(_Model):
    """Intermediate states of a simulated experiment"""

    worker: List[str] = []
    round_idx: List[int] = []
    state: List[ComputePlanBuilder] = []

    @pydantic.field_serializer("state")
    def serialize_state(self, state: ComputePlanBuilder, _info):
        return [str(s) for s in state]

    def __add__(self, other):
        return SimulationIntermediateStates(
            worker=self.worker + other.worker,
            round_idx=self.round_idx + other.round_idx,
            state=self.state + other.state,
        )
