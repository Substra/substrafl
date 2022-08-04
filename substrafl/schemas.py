"""Schemas used in the application.
"""
from enum import Enum
from typing import List

import numpy as np
import pydantic


class StrategyName(str, Enum):
    FEDERATED_AVERAGING = "Federated Averaging"
    SCAFFOLD = "Scaffold"
    ONE_ORGANIZATION = "One organization"
    NEWTON_RAPHSON = "Newton Raphson"


class _Model(pydantic.BaseModel):
    """Base model configuration"""

    class Config:
        arbitrary_types_allowed = True


class FedAvgAveragedState(_Model):
    """Shared state sent by the aggregate_organization in the federated
    averaging strategy."""

    avg_parameters_update: List[np.ndarray]


class FedAvgSharedState(_Model):
    """Shared state returned by the train method of the algorithm for each client,
    received by the aggregate function in the federated averaging strategy.
    """

    n_samples: int
    parameters_update: List[np.ndarray]


class ScaffoldSharedState(_Model):
    """Shared state returned by the train method of the algorithm for each client
    (e.g. algorithms.pytorch.scaffold.train)

    Args:
        parameters_update (typing.List[numpy.ndarray]): the weight update of the client
            (delta between fine-tuned weights and previous weights)
        control_variate_update (typing.List[numpy.ndarray]): the control_variate update of the client
        n_samples (int): the number of samples of the client
        server_control_variate (typing.List[numpy.ndarray]): the server control variate (``c`` in the Scaffold paper's
            Algo). It is sent by every client as the aggregation node doesn't have a persistent state, and
            should be the same for each client as it should not be modified in the client Algo
    """

    parameters_update: List[np.ndarray]
    control_variate_update: List[np.ndarray]
    n_samples: int
    server_control_variate: List[np.ndarray]


class ScaffoldAveragedStates(_Model):
    """Shared state sent by the aggregate_organization (returned by the func strategies.scaffold.avg_shared_states)

    Args:
        server_control_variate (typing.List[numpy.ndarray]): the new server_control_variate sent to the clients
        avg_parameters_update (typing.List[numpy.ndarray]): the weighted average of the parameters_update from each
            client
    """

    server_control_variate: List[np.ndarray]  # the new server_control_variate sent to the clients
    avg_parameters_update: List[np.ndarray]  # the weighted average of the parameters_update from each client


class NewtonRaphsonAveragedStates(_Model):
    """Shared state sent by the aggregate_organization in the Newton Raphson
    strategy.

    Args:
        parameters_update (numpy.ndarray): the new parameters_update sent to the clients
    """

    parameters_update: List[np.ndarray]


class NewtonRaphsonSharedState(_Model):
    """Shared state returned by the train method of the algorithm for each client,
    received by the aggregate function in the Newton Raphson strategy.

    Args:
        n_samples (int): number of samples of the client dataset.
        gradients (numpy.ndarray): gradients of the model parameters :math:`\\theta`.
        hessian (numpy.ndarray): second derivative of the loss function regarding the model parameters :math:`\\theta`.
    """

    n_samples: int
    gradients: List[np.ndarray]
    hessian: np.ndarray
