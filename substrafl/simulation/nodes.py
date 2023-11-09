import copy
import uuid
from typing import Any
from typing import List
from typing import Optional

import substra
import substratools

from substrafl.compute_plan_builder import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import Node
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote.operations import RemoteOperation
from substrafl.simulation.schemas import SimuPerformancesMemory
from substrafl.simulation.schemas import SimuStatesMemory


def _preload_data(
    client: substra.Client,
    data_manager_key: str,
    data_sample_keys: List[str],
) -> Any:
    """Get the opener from the client using its key, and apply
    the method `get_data` to the datasamples in order to retrieve them.

    Args:
        client(substra.Client): A substra client to interact with the Substra platform, in order to retrieve the
            registered data.
        data_manager_key(str): key of the registered opener.
        data_sample_keys(List[str]): keys of the registered datasamples paths.

    Returns:
        Any: output of the opener's `get_data` method applied on the corresponding datasamples paths.
    """
    dataset_info = client.get_dataset(data_manager_key)

    opener_interface = substratools.utils.load_interface_from_module(
        "opener",
        interface_class=substratools.Opener,
        interface_signature=None,
        path=dataset_info.opener.storage_address,
    )

    data_sample_paths = [client.get_data_sample(dsk).path for dsk in data_sample_keys]

    return opener_interface.get_data(data_sample_paths)


class SimuTrainDataNode(Node):
    def __init__(self, client: substra.Client, node: TrainDataNode, strategy: ComputePlanBuilder):
        super().__init__(node.organization_id)

        self.data_manager_key = node.data_manager_key
        self.data_sample_keys = node.data_sample_keys

        self._datasamples = _preload_data(
            client=client, data_manager_key=self.data_manager_key, data_sample_keys=self.data_sample_keys
        )
        self._strategy = strategy
        self._memory = SimuStatesMemory()

    def init_states(self, *args, **kwargs) -> LocalStateRef:
        return LocalStateRef(key=str(uuid.uuid4()), init=True)

    def update_states(
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
            self._memory.state.append(copy.deepcopy(self._strategy))
            self._memory.round_idx.append(round_idx)
            self._memory.worker.append(self.organization_id)

        return LocalStateRef(key=str(uuid.uuid4())), shared_state


class SimuTestDataNode(Node):
    def __init__(self, client: substra.Client, node: TestDataNode, strategy: ComputePlanBuilder):
        super().__init__(node.organization_id)

        self.data_manager_key = node.data_manager_key
        self.test_data_sample_keys = node.test_data_sample_keys

        self._datasamples = _preload_data(
            client=client, data_manager_key=self.data_manager_key, data_sample_keys=self.test_data_sample_keys
        )

        self._strategy = strategy
        self._memory = SimuPerformancesMemory()

    def update_states(
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
            self._memory.performance.append(scores[metric])
            self._memory.identifier.append(metric)
            self._memory.round_idx.append(round_idx)
            self._memory.worker.append(self.organization_id)


class SimuAggregationNode(Node):
    def __init__(self, node: AggregationNode, strategy: ComputePlanBuilder):
        super().__init__(node.organization_id)
        self._strategy = strategy
        self._memory = SimuStatesMemory()

    def update_states(
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
            self._memory.state.append(copy.deepcopy(self._strategy))
            self._memory.round_idx.append(round_idx)
            self._memory.worker.append(self.organization_id)

        return shared_state
