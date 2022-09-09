import uuid
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import substra
from substra.sdk.schemas import AlgoCategory
from substra.sdk.schemas import AlgoInputSpec
from substra.sdk.schemas import AlgoOutputSpec
from substra.sdk.schemas import AssetKind
from substra.sdk.schemas import ComputeTaskOutputSpec
from substra.sdk.schemas import InputRef
from substra.sdk.schemas import Permissions

from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import Node
from substrafl.nodes.node import OperationKey
from substrafl.nodes.node import OutputIdentifiers
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote.operations import DataOperation
from substrafl.remote.register import register_algo
from substrafl.remote.remote_struct import RemoteStruct


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

        super(TrainDataNode, self).__init__(organization_id)

    def update_states(
        self,
        operation: DataOperation,
        round_idx: int,
        authorized_ids: List[str],
        clean_models: bool = False,
        local_state: Optional[LocalStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        """Adding a composite train tuple to the list of operations to
        be executed by the node during the compute plan. This is done in a static
        way, nothing is submitted to substra.
        This is why the algo key is a RemoteStruct (substrafl local reference of the algorithm)
        and not a substra algo_key as nothing has been submitted yet.

        Args:
            operation (DataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Round number, it starts at 1. In case of a centralized strategy,
                it is preceded by an initialization round tagged: 0.
            authorized_ids (List[str]): Authorized org to access the output model.
            clean_models (bool): Whether outputs of this operation are transient (deleted when they are not used
                anymore) or not. Defaults to False.
            local_state (typing.Optional[LocalStateRef]): The parent task LocalStateRef. Defaults to None.

        Raises:
            TypeError: operation must be a DataOperation, make sure to decorate the train and predict methods of
                your algorithm with @remote

        Returns:
            typing.Tuple[LocalStateRef, SharedStateRef]: Identifications for the results of this operation.
        """
        if not isinstance(operation, DataOperation):
            raise TypeError(
                "operation must be a DataOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote_data?",
            )

        op_id = str(uuid.uuid4())
        data_inputs = [InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)] + [
            InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.data_sample_keys
        ]

        local_inputs = (
            [
                InputRef(
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
                InputRef(
                    identifier=InputIdentifiers.shared,
                    parent_task_key=operation.shared_state.key,
                    parent_task_output_identifier=OutputIdentifiers.model,
                )
            ]

        elif local_state is not None:
            shared_inputs = [
                InputRef(
                    identifier=InputIdentifiers.shared,
                    parent_task_key=local_state.key,
                    parent_task_output_identifier=OutputIdentifiers.shared,
                )
            ]

        else:
            shared_inputs = []

        composite_traintuple = {
            "remote_operation": operation.remote_struct,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": operation.data_samples,
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": operation.shared_state.key
            if operation.shared_state is not None
            else None,  # user-defined id (last aggregation node task id)
            "tag": "train",
            "composite_traintuple_id": op_id,
            "inputs": data_inputs + local_inputs + shared_inputs,
            "outputs": {
                OutputIdentifiers.shared: ComputeTaskOutputSpec(
                    permissions=Permissions(public=False, authorized_ids=authorized_ids),
                    transient=clean_models,
                ),
                OutputIdentifiers.local: ComputeTaskOutputSpec(
                    permissions=Permissions(public=False, authorized_ids=[self.organization_id]),
                    transient=clean_models,
                ),
            },
            "metadata": {
                "round_idx": round_idx,
            },
        }

        self.tuples.append(composite_traintuple)

        return LocalStateRef(op_id), SharedStateRef(op_id)

    def register_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Define the algorithms for each operation and submit the composite traintuple to substra.

        Go through every operation in the computation graph, check what algorithm they use (identified by their
        RemoteStruct id), submit it to substra and save `RemoteStruct : algo_key` into the `cache` (where algo_key
        is the returned algo key by substra.)
        If two tuples depend on the same algorithm, the algorithm won't be added twice to substra as this method check
        if an algo has already been submitted to substra before adding it.

        Args:
            client (substra.Client): Substra client for the organization.
            permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered algorithm identifications. The key of
                each element is the RemoteStruct id (generated by substrafl) and the value is the key generated by
                substra.
            dependencies (Dependency): Algorithm dependencies.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """
        for tuple in self.tuples:
            if isinstance(tuple["remote_operation"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["remote_operation"]

                if remote_struct not in cache:
                    algo_key = register_algo(
                        client=client,
                        category=AlgoCategory.composite,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        inputs=[
                            AlgoInputSpec(
                                identifier=InputIdentifiers.datasamples,
                                kind=AssetKind.data_sample.value,
                                optional=False,
                                multiple=True,
                            ),
                            AlgoInputSpec(
                                identifier=InputIdentifiers.opener,
                                kind=AssetKind.data_manager.value,
                                optional=False,
                                multiple=False,
                            ),
                            AlgoInputSpec(
                                identifier=InputIdentifiers.local,
                                kind=AssetKind.model.value,
                                optional=True,
                                multiple=False,
                            ),
                            AlgoInputSpec(
                                identifier=InputIdentifiers.shared,
                                kind=AssetKind.model.value,
                                optional=True,
                                multiple=False,
                            ),
                        ],
                        outputs=[
                            AlgoOutputSpec(
                                identifier=OutputIdentifiers.local, kind=AssetKind.model.value, multiple=False
                            ),
                            AlgoOutputSpec(
                                identifier=OutputIdentifiers.shared, kind=AssetKind.model.value, multiple=False
                            ),
                        ],
                        dependencies=dependencies,
                    )
                    cache[remote_struct] = algo_key
                else:
                    algo_key = cache[remote_struct]

                tuple["algo_key"] = algo_key

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
