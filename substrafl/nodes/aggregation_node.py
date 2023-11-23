from __future__ import annotations

import copy
import uuid
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import TypeVar

import substra

from substrafl.dependency import Dependency
from substrafl.nodes.protocol import AggregationNodeProtocol
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.nodes.schemas import InputIdentifiers
from substrafl.nodes.schemas import OperationKey
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.nodes.schemas import SimuStatesMemory
from substrafl.remote.operations import RemoteOperation
from substrafl.remote.register import register_function
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.schemas import TaskType

if TYPE_CHECKING:
    from substrafl.compute_plan_builder import ComputePlanBuilder


SharedState = TypeVar("SharedState")


class AggregationNode(AggregationNodeProtocol):
    """The node which applies operations to the shared states which are received from ``TrainDataNode``
    data operations.
    The result is sent to the ``TrainDataNode`` and/or ``TestDataNode`` data operations.
    """

    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.tasks: List[Dict] = []

    def update_states(
        self,
        operation: RemoteOperation,
        *,
        authorized_ids: Set[str],
        round_idx: Optional[int] = None,
        clean_models: bool = False,
    ) -> SharedStateRef:
        """Adding an aggregated task to the list of operations to be executed by the node during the compute plan.
        This is done in a static way, nothing is submitted to substra.
        This is why the function key is a RemoteStruct (substrafl local reference of the algorithm)
        and not a substra function_key as nothing has been submitted yet.

        Args:
            operation (RemoteOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Used in case of learning compute plans. Round number, it starts at 1. Default to None.
            authorized_ids (Set[str]): Authorized org to access the output model.
            clean_models (bool): Whether outputs of this operation are transient (deleted when they are not used
                anymore) or not. Defaults to False.

        Raises:
            TypeError: operation must be an RemoteOperation, make sure to decorate your (user defined) aggregate
                function of the strategy with @remote.

        Returns:
            SharedStateRef: Identification for the result of this operation.
        """
        if not isinstance(operation, RemoteOperation):
            raise TypeError(
                "operation must be a RemoteOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote?",
            )

        op_id = str(uuid.uuid4())

        inputs = (
            [
                substra.schemas.InputRef(
                    identifier=InputIdentifiers.shared,
                    parent_task_key=ref.key,
                    parent_task_output_identifier=OutputIdentifiers.shared,
                )
                for ref in operation.shared_states
            ]
            if operation.shared_states is not None
            else None
        )

        task_metadata = {"round_idx": str(round_idx)} if round_idx is not None else {}
        aggregate_task = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=op_id,
            inputs=inputs,
            outputs={
                OutputIdentifiers.shared: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=False, authorized_ids=list(authorized_ids)),
                    transient=clean_models,
                )
            },
            metadata=task_metadata,
            tag=TaskType.AGGREGATE,
            worker=self.organization_id,
        ).model_dump()

        aggregate_task.pop("function_key")
        aggregate_task["remote_operation"] = operation.remote_struct

        self.tasks.append(aggregate_task)

        return SharedStateRef(key=op_id)

    def register_operations(
        self,
        *,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Define the functions for each operation and submit the aggregated task to substra.

        Go through every operation in the computation graph, check what function they use (identified by their
        RemoteStruct id), submit it to substra and save `RemoteStruct : function_key` into the `cache`
        (where function_key is the returned function key per substra.)
        If two tasks depend on the same function, the function won't be added twice to substra as this method check
        if a function has already been submitted to substra before adding it.

        Args:
            client (substra.Client): Substra defined client used to register the operation.
            permissions (substra.sdk.schemas.Permissions): Substra permissions attached to the registered operation.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered function identifications. The key of
                each element is the RemoteStruct id (generated by substrafl) and the value is the key generated by
                substra.
            dependencies (Dependency): Dependencies of the given operation.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """
        for task in self.tasks:
            if isinstance(task["remote_operation"], RemoteStruct):
                remote_struct: RemoteStruct = task["remote_operation"]

                if remote_struct not in cache:
                    function_key = register_function(
                        client=client,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        dependencies=dependencies,
                        inputs=[
                            substra.schemas.FunctionInputSpec(
                                identifier=InputIdentifiers.shared,
                                kind=substra.schemas.AssetKind.model.value,
                                optional=False,
                                multiple=True,
                            )
                        ],
                        outputs=[
                            substra.schemas.FunctionOutputSpec(
                                identifier=OutputIdentifiers.shared,
                                kind=substra.schemas.AssetKind.model.value,
                                multiple=False,
                            )
                        ],
                    )
                    cache[remote_struct] = function_key
                else:
                    function_key = cache[remote_struct]

                del task["remote_operation"]
                task["function_key"] = function_key

        return cache

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {
            "organization_id": self.organization_id,
        }


class SimuAggregationNode(AggregationNodeProtocol):
    def __init__(self, node: AggregationNode, strategy: ComputePlanBuilder):
        self.organization_id = node.organization_id
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

        This function is expected to implement the `update_state` method of a AggregationNodeProtocol
        to simulate its execution on RAM only.

        Args:
            operation (RemoteOperation): Object containing all the information to execute
                the method.
            round_idx (Optional[int]): Current round idx. Defaults to None.
            clean_models (Optional[bool]): If set to True, the current state of the instance will
                be saved in a SimuStatesMemory object. Defaults to True.

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

    def register_operations(self, *args, **kwargs) -> None:
        return

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {
            "organization_id": self.organization_id,
        }
