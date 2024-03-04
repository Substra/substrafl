from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional

import substra

from substrafl.dependency import Dependency
from substrafl.nodes.protocol import TestDataNodeProtocol
from substrafl.nodes.schemas import InputIdentifiers
from substrafl.nodes.schemas import OperationKey
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.nodes.schemas import SimuPerformancesMemory
from substrafl.nodes.utils import preload_data
from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.register import register_function
from substrafl.remote.remote_struct import RemoteStruct

if TYPE_CHECKING:
    from substrafl.compute_plan_builder import ComputePlanBuilder


class TestDataNode(TestDataNodeProtocol):
    """A node on which you will test your algorithm.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
    ):
        self.organization_id = organization_id

        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys

        self.tasks: List[Dict] = []

    def update_states(
        self,
        traintask_id: str,
        operation: RemoteDataOperation,
        round_idx: Optional[int] = None,
    ):
        """Creating a test task based on the node characteristic.

        Args:
            traintask_id (str): The substra parent id
            operation (RemoteDataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Used in case of learning compute plans. Round number, it starts at 1. Default to None.

        """

        data_inputs = [
            substra.schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)
        ] + [
            substra.schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.data_sample_keys
        ]
        local_input = [
            substra.schemas.InputRef(
                identifier=InputIdentifiers.local,
                parent_task_key=traintask_id,
                parent_task_output_identifier=OutputIdentifiers.local,
            ),
        ]

        task_metadata = {"round_idx": str(round_idx)} if round_idx is not None else {}

        testtask = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=str(uuid.uuid4()),
            inputs=data_inputs + local_input,
            outputs={
                metric_function_id: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=True, authorized_ids=[]),
                    transient=False,
                )
                # To be able to create the right amount of task outputs, we need to know
                # what are the metrics associated with the evaluate function.
                # We get the metric_functions from the instance.
                for metric_function_id in operation.remote_struct.get_instance().metric_functions
            },
            metadata=task_metadata,
            worker=self.organization_id,
        ).model_dump()
        testtask.pop("function_key")
        testtask["remote_operation"] = operation.remote_struct
        self.tasks.append(testtask)

    def register_operations(
        self,
        *,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ):
        for task in self.tasks:
            remote_struct: RemoteStruct = task["remote_operation"]

            if remote_struct not in cache:
                # Register the evaluation task
                function_key = register_function(
                    client=client,
                    remote_struct=remote_struct,
                    permissions=permissions,
                    inputs=[
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.datasamples,
                            kind=substra.schemas.AssetKind.data_sample.value,
                            optional=False,
                            multiple=True,
                        ),
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.opener,
                            kind=substra.schemas.AssetKind.data_manager.value,
                            optional=False,
                            multiple=False,
                        ),
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.local,
                            kind=substra.schemas.AssetKind.model.value,
                            optional=False,
                            multiple=False,
                        ),
                    ],
                    outputs=[
                        substra.schemas.FunctionOutputSpec(
                            identifier=metric_function_id,
                            kind=substra.schemas.AssetKind.performance.value,
                            multiple=False,
                        )
                        # To be able to create the right amount of function outputs, we need to know
                        # what are the metrics associated with the evaluate function.
                        # We get the metric_functions from the instance.
                        for metric_function_id in remote_struct.get_instance().metric_functions
                    ],
                    dependencies=dependencies,
                )
                task["function_key"] = function_key
                cache[remote_struct] = function_key
            else:
                function_key = cache[remote_struct]
                task["function_key"] = function_key

        return cache

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {
            "organization_id": self.organization_id,
            "data_manager_key": self.data_manager_key,
            "data_sample_keys": self.data_sample_keys,
        }


class SimuTestDataNode(TestDataNodeProtocol):
    def __init__(self, client: substra.Client, node: TestDataNode, strategy: ComputePlanBuilder):
        self.organization_id = node.organization_id

        self.data_manager_key = node.data_manager_key
        self.data_sample_keys = node.data_sample_keys

        self._data_from_opener = preload_data(
            client=client, data_manager_key=self.data_manager_key, data_sample_keys=self.data_sample_keys
        )

        self._strategy = strategy
        self._memory = SimuPerformancesMemory()

    def update_states(
        self,
        operation: RemoteDataOperation,
        round_idx: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """This function will execute the method to run on the test node with the argument
        `_skip=True`, to execute it directly in RAM. The executed method is expected to compute
        metrics, and to return a dictionary with the metric name as key and the result as value.
        It will also stores the computed scores in a `SimuPerformancesMemory` object.

        This function is expected to implement the `update_state` method of a TestDataNodeProtocol
        to simulate its execution on RAM only.

        Args:
            operation (RemoteDataOperation): Object containing all the information to execute
                the method.
            round_idx (Optional[int]): Current round idx. Defaults to None.
        """
        method_name = operation.remote_struct._method_name
        method_parameters = operation.remote_struct._method_parameters

        method_parameters["shared_state"] = operation.shared_state
        method_parameters["data_from_opener"] = self._data_from_opener
        method_to_run = getattr(self._strategy, method_name)

        scores = method_to_run(**method_parameters, _skip=True)

        for metric in scores:
            self._memory.performance.append(scores[metric])
            self._memory.identifier.append(metric)
            self._memory.round_idx.append(round_idx)
            self._memory.worker.append(self.organization_id)

    def register_operations(self, *args, **kwargs) -> None:
        return

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {
            "organization_id": self.organization_id,
            "data_manager_key": self.data_manager_key,
            "data_sample_keys": self.data_sample_keys,
        }
