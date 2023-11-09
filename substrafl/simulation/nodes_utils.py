import copy
import uuid
from typing import Any
from typing import Optional

from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote.operations import RemoteOperation


def simulate_train_update_states(
    self,
    operation: RemoteOperation,
    round_idx: Optional[int] = None,
    clean_models: Optional[bool] = True,
    *args,
    **kwargs,
) -> (LocalStateRef, Any):
    """This function will execute the method to run on the train node with the argument
    `_skip=True`, to execute it directly in RAM.

    This function is expected to overload the `update_state` method of a TrainDataNode
    to simulate its execution on RAM only.

    Args:
        operation (RemoteOperation): Object containing all the information to execute
            the method.
        round_idx (Optional[int]): Current round idx. Defaults to None.
        clean_models (Optional[bool]): If set to True, the current state of the instance will
            be saved in a SimulationIntermediateStates object. Defaults to True

    Returns:
        LocalStateRef: A bogus `LocalStateRef` to ensure compatibility with the execution of the
            experiment.
        Any: the output of the execution of the method stored in `operation`.
    """
    method_name = operation.remote_struct._method_name
    method_parameters = operation.remote_struct._method_parameters

    method_parameters["shared_state"] = operation.shared_state
    method_parameters["datasamples"] = self._datasamples

    # To be compatible with custom compute plans
    try:
        method_to_run = getattr(self._strategy.algo, method_name)
    except AttributeError:
        method_to_run = getattr(self._strategy, method_name)

    shared_state = method_to_run(**method_parameters, _skip=True)

    if not clean_models:
        self._intermediate_states.state.append(copy.deepcopy(self._strategy))
        self._intermediate_states.round_idx.append(round_idx)
        self._intermediate_states.worker.append(self.organization_id)

    return LocalStateRef(key=str(uuid.uuid4())), shared_state


def simulate_test_update_states(
    self,
    operation: RemoteOperation,
    round_idx: Optional[int] = None,
    *args,
    **kwargs,
) -> None:
    """This function will execute the method to run on the test node with the argument
    `_skip=True`, to execute it directly in RAM. The executed method is expected to compute
    metrics, and to return a dictionary with the metric name as key and the result as value.
    It will also stores the computed scores in a `SimulationPerformance` object.

    This function is expected to overload the `update_state` method of a TestDataNode
    to simulate its execution on RAM only.

    Args:
        operation (RemoteOperation): Object containing all the information to execute
            the method.
        round_idx (Optional[int]): Current round idx. Defaults to None.
    """
    method_name = operation.remote_struct._method_name
    method_parameters = operation.remote_struct._method_parameters

    method_parameters["shared_state"] = operation.shared_state
    method_parameters["datasamples"] = self._datasamples
    method_to_run = getattr(self._strategy, method_name)

    scores = method_to_run(**method_parameters, _skip=True)

    for metric in scores:
        self._score.performance.append(scores[metric])
        self._score.identifier.append(metric)
        self._score.round_idx.append(round_idx)
        self._score.worker.append(self.organization_id)


def simulate_aggregate_update_states(
    self,
    operation: RemoteOperation,
    round_idx: Optional[int] = None,
    clean_models: Optional[bool] = True,
    *args,
    **kwargs,
) -> Any:
    """This function will execute the method to run on the aggregation node with the argument
    `_skip=True`, to execute it directly in RAM.

    This function is expected to overload the `update_state` method of a AggregationNode
    to simulate its execution on RAM only.

    Args:
        operation (RemoteOperation): Object containing all the information to execute
            the method.
        round_idx (Optional[int]): Current round idx. Defaults to None.
        clean_models (Optional[bool]): If set to True, the current state of the instance will
            be saved in a SimulationIntermediateStates object. Defaults to True.

    Returns:
        Any: the output of the execution of the method stored in `operation`.
    """
    method_name = operation.remote_struct._method_name
    method_parameters = operation.remote_struct._method_parameters

    method_parameters["shared_states"] = operation.shared_states
    method_to_run = getattr(self._strategy, method_name)

    shared_state = method_to_run(**method_parameters, _skip=True)

    if not clean_models:
        self._intermediate_states.state.append(copy.deepcopy(self._strategy))
        self._intermediate_states.round_idx.append(round_idx)
        self._intermediate_states.worker.append(self.organization_id)

    return shared_state
