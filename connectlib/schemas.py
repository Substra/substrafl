"""Schemas used in the application.
"""
from typing import List

import numpy as np
from pydantic import BaseModel


class ScaffoldSharedState(BaseModel):
    """Shared state returned by the train method of the algorithm for each client
    (e.g. algorithms.pytorch.scaffold.train)

    Args:
        parameters_update (typing.List[numpy.ndarray]): the weight update of the client
            (delta between fine-tuned weights and previous weights)
        control_variate_update (List[numpy.ndarray]): the control_variate update of the client
        n_samples (int): the number of samples of the client
        server_control_variate (List[numpy.ndarray]): the server control variate (``c`` in the Scaffold paper's
            Algo). It is sent by every client as the aggregation node doesn't have a persistent state, and
            should be the same for each client as it should not be modified in the client Algo
    """

    parameters_update: List[np.ndarray]
    control_variate_update: List[np.ndarray]
    n_samples: int
    server_control_variate: List[np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class ScaffoldAveragedStates(BaseModel):
    """Shared state sent by the aggregate_node (returned by the func strategies.scaffold.avg_shared_states)

    Args:
        server_control_variate (typing.List[numpy.ndarray]): the new server_control_variate sent to the clients
        avg_parameters_update (typing.List[numpy.ndarray]): the weighted average of the parameters_update from each
            client
    """

    server_control_variate: List[np.ndarray]  # the new server_control_variate sent to the clients
    avg_parameters_update: List[np.ndarray]  # the weighted average of the parameters_update from each client

    class Config:
        arbitrary_types_allowed = True
