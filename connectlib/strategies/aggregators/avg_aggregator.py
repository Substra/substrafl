import numpy as np

from typing import List, Dict

from .aggregator import Aggregator


class AvgAggregator(Aggregator):
    def aggregate_states(self, states: List[Dict[str, np.array]]):
        if not states:
            raise FileNotFoundError("This algo needs input states")

        # get keys
        keys = states[0].keys()

        # average weights
        averaged_states = {}
        for key in keys:
            states = np.stack([local_state[key] for local_state in states])
            averaged_states[key] = np.mean(states, axis=0)

        return averaged_states
