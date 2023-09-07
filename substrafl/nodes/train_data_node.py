import uuid
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import substra

from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import Node
from substrafl.nodes.node import OperationKey
from substrafl.nodes.node import OutputIdentifiers
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.operations import RemoteOperation
from substrafl.remote.register import register_function
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.schemas import TaskType


class TrainDataNode(Node):
    """
    A predefined structure that allows you to register operations
    on your train node in a static way before submitting them to substra.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        data_sample_keys (typing.List[str]): Substra data_sample_keys used for the training on this node
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
    ):
        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys

        self.init_task = None

        super().__init__(organization_id)

    def init_states(
        self,
        *,
        operation: RemoteOperation,
        round_idx: int,
        authorized_ids: Set[str],
        clean_models: bool = False,
    ) -> LocalStateRef:
        op_id = str(uuid.uuid4())

        init_task = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=op_id,
            inputs=[],
            outputs={
                OutputIdentifiers.local: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=False, authorized_ids=list(authorized_ids)),
                    transient=clean_models,
                ),
            },
            metadata={
                "round_idx": str(round_idx),
            },
            tag=TaskType.INITIALIZATION,
            worker=self.organization_id,
        ).dict()

        init_task.pop("function_key")
        init_task["remote_operation"] = operation.remote_struct

        self.init_task = init_task

        return LocalStateRef(key=op_id, init=True)

    def update_states(
        self,
        operation: RemoteDataOperation,
        *,
        authorized_ids: Set[str],
        round_idx: Optional[int] = None,
        aggregation_id: Optional[str] = None,
        clean_models: bool = False,
        local_state: Optional[LocalStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        """Adding a train task to the list of operations to
        be executed by the node during the compute plan. This is done in a static
        way, nothing is submitted to substra.
        This is why the function key is a RemoteStruct (substrafl local reference of the algorithm)
        and not a substra function_key as nothing has been submitted yet.

        Args:
            operation (RemoteDataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            authorized_ids (typing.Set[str]): Authorized org to access the output model.
            round_idx (int): Used in case of learning compute plans. Round number, it starts at 1.
                In case of a centralized strategy, it is preceded by an initialization round tagged: 0. Default to None
            aggregation_id (str): Aggregation node id to authorize access to the shared model. Defaults to None.
            clean_models (bool): Whether outputs of this operation are transient (deleted when they are not used
                anymore) or not. Defaults to False.
            local_state (typing.Optional[LocalStateRef]): The parent task LocalStateRef. Defaults to None.

        Raises:
            TypeError: operation must be a RemoteDataOperation, make sure to decorate the train and predict methods of
                your method with @remote

        Returns:
            typing.Tuple[LocalStateRef, SharedStateRef]: Identifications for the results of this operation.
        """
        if not isinstance(operation, RemoteDataOperation):
            raise TypeError(
                "operation must be a RemoteDataOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote_data?",
            )
        op_id = str(uuid.uuid4())
        data_inputs = [
            substra.schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)
        ] + [
            substra.schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.data_sample_keys
        ]
        local_inputs = (
            [
                substra.schemas.InputRef(
                    identifier=InputIdentifiers.local,
                    parent_task_key=local_state.key,
                    parent_task_output_identifier=OutputIdentifiers.local,
                )
            ]
            if local_state is not None
            else []
        )
        if operation.shared_state is not None:
            shared_inputs = [
                substra.schemas.InputRef(
                    identifier=InputIdentifiers.shared,
                    parent_task_key=operation.shared_state.key,
                    parent_task_output_identifier=OutputIdentifiers.shared,
                )
            ]

        elif local_state is not None and not local_state.init:
            # If the parent task is an init task, no shared states have been produced.
            shared_inputs = [
                substra.schemas.InputRef(
                    identifier=InputIdentifiers.shared,
                    parent_task_key=local_state.key,
                    parent_task_output_identifier=OutputIdentifiers.shared,
                )
            ]

        else:
            shared_inputs = []

        task_metadata = {"round_idx": str(round_idx)} if round_idx is not None else {}
        train_task = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=op_id,
            inputs=data_inputs + local_inputs + shared_inputs,
            outputs={
                OutputIdentifiers.shared: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(
                        public=False,
                        authorized_ids=list(authorized_ids | set([aggregation_id] if aggregation_id else [])),
                    ),
                    transient=clean_models,
                ),
                OutputIdentifiers.local: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=False, authorized_ids=list(authorized_ids)),
                    transient=clean_models,
                ),
            },
            metadata=task_metadata,
            tag=TaskType.TRAIN,
            worker=self.organization_id,
        ).dict()

        train_task.pop("function_key")
        train_task["remote_operation"] = operation.remote_struct

        self.tasks.append(train_task)

        return LocalStateRef(op_id), SharedStateRef(op_id)

    def register_operations(
        self,
        *,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Define the functions for each operation and submit the train task to substra.

        Go through every operation in the computation graph, check what function they use (identified by their
        RemoteStruct id), submit it to substra and save `RemoteStruct : function_key` into the `cache`
        (where function_key is the returned function key by substra.)
        If two tasks depend on the same function, the function won't be added twice to substra as this method check
        if a function has already been submitted to substra before adding it.

        Args:
            client (substra.Client): Substra client for the organization.
            permissions (substra.sdk.schemas.Permissions): Permissions for the function.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered function identifications. The key of
                each element is the RemoteStruct id (generated by substrafl) and the value is the key generated by
                substra.
            dependencies (Dependency): Function dependencies.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """

        if self.init_task is not None:
            # Register init task if exists
            init_remote_struct: RemoteStruct = self.init_task["remote_operation"]
            function_key = register_function(
                client=client,
                remote_struct=init_remote_struct,
                permissions=permissions,
                inputs=[],
                outputs=[
                    substra.schemas.FunctionOutputSpec(
                        identifier=OutputIdentifiers.local, kind=substra.schemas.AssetKind.model.value, multiple=False
                    ),
                ],
                dependencies=dependencies,
            )
            self.init_task["function_key"] = function_key
            cache[init_remote_struct] = function_key

        for task in self.tasks:
            if isinstance(task["remote_operation"], RemoteStruct):
                remote_struct: RemoteStruct = task["remote_operation"]

                if remote_struct not in cache:
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
                                optional=True,
                                multiple=False,
                            ),
                            substra.schemas.FunctionInputSpec(
                                identifier=InputIdentifiers.shared,
                                kind=substra.schemas.AssetKind.model.value,
                                optional=True,
                                multiple=False,
                            ),
                        ],
                        outputs=[
                            substra.schemas.FunctionOutputSpec(
                                identifier=OutputIdentifiers.local,
                                kind=substra.schemas.AssetKind.model.value,
                                multiple=False,
                            ),
                            substra.schemas.FunctionOutputSpec(
                                identifier=OutputIdentifiers.shared,
                                kind=substra.schemas.AssetKind.model.value,
                                multiple=False,
                            ),
                        ],
                        dependencies=dependencies,
                    )
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
        summary = super().summary()
        summary.update(
            {
                "data_manager_key": self.data_manager_key,
                "data_sample_keys": self.data_sample_keys,
            }
        )
        return summary
