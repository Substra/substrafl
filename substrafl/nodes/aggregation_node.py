import uuid
from typing import Dict
from typing import List
from typing import TypeVar

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
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.remote.operations import AggregateOperation
from substrafl.remote.register import register_algo
from substrafl.remote.remote_struct import RemoteStruct

SharedState = TypeVar("SharedState")


class AggregationNode(Node):
    """The node which applies operations to the shared states which are received from ``TrainDataNode``
    data operations.
    The result is sent to the ``TrainDataNode`` and/or ``TestDataNode`` data operations.
    """

    def update_states(
        self,
        operation: AggregateOperation,
        round_idx: int,
        authorized_ids: List[str],
        clean_models: bool = False,
    ) -> SharedStateRef:
        """Adding an aggregated tuple to the list of operations to be executed by the node during the compute plan.
        This is done in a static way, nothing is submitted to substra.
        This is why the algo key is a RemoteStruct (substrafl local reference of the algorithm)
        and not a substra algo_key as nothing has been submitted yet.

        Args:
            operation (AggregateOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Round number, it starts at 1.
            authorized_ids (List[str]): Authorized org to access the output model.
            clean_models (bool): Whether outputs of this operation are transient (deleted when they are not used
                anymore) or not. Defaults to False.

        Raises:
            TypeError: operation must be an AggregateOperation, make sure to decorate your (user defined) aggregate
                function of the strategy with @remote.

        Returns:
            SharedStateRef: Identification for the result of this operation.
        """
        if not isinstance(operation, AggregateOperation):
            raise TypeError(
                "operation must be a AggregateOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote?",
            )

        op_id = str(uuid.uuid4())

        inputs = (
            [
                InputRef(
                    identifier=InputIdentifiers.models,
                    parent_task_key=ref.key,
                    parent_task_output_identifier=OutputIdentifiers.shared,
                )
                for ref in operation.shared_states
            ]
            if operation.shared_states is not None
            else None
        )

        aggregate_tuple = {
            "remote_operation": operation.remote_struct,
            "worker": self.organization_id,
            "in_models_ids": [_input.parent_task_key for _input in inputs] if inputs is not None else None,
            "tag": "aggregate",
            "aggregatetuple_id": op_id,
            "metadata": {
                "round_idx": round_idx,
            },
            "inputs": inputs,
            "outputs": {
                OutputIdentifiers.model: ComputeTaskOutputSpec(
                    permissions=Permissions(public=False, authorized_ids=authorized_ids),
                    transient=clean_models,
                )
            },
        }
        self.tuples.append(aggregate_tuple)

        return SharedStateRef(key=op_id)

    def register_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Define the algorithms for each operation and submit the aggregated tuple to substra.

        Go through every operation in the computation graph, check what algorithm they use (identified by their
        RemoteStruct id), submit it to substra and save `RemoteStruct : algo_key` into the `cache` (where algo_key
        is the returned algo key per substra.)
        If two tuples depend on the same algorithm, the algorithm won't be added twice to substra as this method check
        if an algo has already been submitted to substra before adding it.

        Args:
            client (substra.Client): Substra defined client used to register the operation.
            permissions (substra.sdk.schemas.Permissions): Substra permissions attached to the registered operation.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered algorithm identifications. The key of
                each element is the RemoteStruct id (generated by substrafl) and the value is the key generated by
                substra.
            dependencies (Dependency): Dependencies of the given operation.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """
        for tuple in self.tuples:
            if isinstance(tuple["remote_operation"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["remote_operation"]

                if remote_struct not in cache:
                    algo_key = register_algo(
                        client=client,
                        category=AlgoCategory.aggregate,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        dependencies=dependencies,
                        inputs=[
                            AlgoInputSpec(
                                identifier=InputIdentifiers.models,
                                kind=AssetKind.model.value,
                                optional=False,
                                multiple=True,
                            )
                        ],
                        outputs=[
                            AlgoOutputSpec(
                                identifier=OutputIdentifiers.model, kind=AssetKind.model.value, multiple=False
                            )
                        ],
                    )
                    cache[remote_struct] = algo_key
                else:
                    algo_key = cache[remote_struct]

                del tuple["remote_operation"]
                tuple["algo_key"] = algo_key

        return cache
