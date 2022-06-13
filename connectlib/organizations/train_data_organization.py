import uuid
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import substra

from connectlib.dependency import Dependency
from connectlib.organizations.organization import OperationKey
from connectlib.organizations.organization import Organization
from connectlib.organizations.references.local_state import LocalStateRef
from connectlib.organizations.references.shared_state import SharedStateRef
from connectlib.remote.operations import DataOperation
from connectlib.remote.register import register_algo
from connectlib.remote.remote_struct import RemoteStruct


class TrainDataOrganization(Organization):
    """
    A predefined structure that allows you to register operations
    on your train organization in a static way before submitting them to substra.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        data_sample_keys (typing.List[str]): Substra data_sample_keys used for the training on this organization
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
    ):
        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys

        super(TrainDataOrganization, self).__init__(organization_id)

    def update_states(
        self,
        operation: DataOperation,
        round_idx: int,
        local_state: Optional[LocalStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        """Adding a composite train tuple to the list of operations to
        be executed by the organization during the compute plan. This is done in a static
        way, nothing is submitted to substra.
        This is why the algo key is a RemoteStruct (connectlib local reference of the algorithm)
        and not a substra algo_key as nothing has been submitted yet.

        Args:
            operation (DataOperation): Automatically generated structure returned by
                the :py:func:`~connectlib.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Round number, it starts at 1. In case of a centralized strategy,
                it is preceded by an initialization round tagged: 0.
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

        composite_traintuple = {
            "remote_operation": operation.remote_struct,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": operation.data_samples,
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": operation.shared_state.key
            if operation.shared_state is not None
            else None,  # user-defined id (last aggregation organization task id)
            "tag": "train",
            "composite_traintuple_id": op_id,
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
        is the returned algo key per substra.)
        If two tuples depend on the same algorithm, the algorithm won't be added twice to substra as this method check
        if an algo has already been submitted to substra before adding it.

        Args:
            client (substra.Client): Substra client for the organization.
            permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered algorithm identifications. The key of
                each element is the RemoteStruct id (generated by connectlib) and the value is the key generated by
                substra.
            dependencies (Dependency): Algorithm dependencies.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """
        for tuple in self.tuples:
            if tuple.get("out_trunk_model_permissions", None) is None:
                tuple["out_trunk_model_permissions"] = permissions

            if isinstance(tuple["remote_operation"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["remote_operation"]

                if remote_struct not in cache:
                    algo_key = register_algo(
                        client=client,
                        is_composite=True,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        dependencies=dependencies,
                    )
                    cache[remote_struct] = algo_key
                else:
                    algo_key = cache[remote_struct]

                del tuple["remote_operation"]
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
