import substra
import uuid

from typing import List, Optional, Tuple

from connectlib.nodes.references import (
    LocalStateRef,
    SharedStateRef,
)
from connectlib.nodes import Node
from connectlib.remote.methods import RemoteStruct, DataOperation
from connectlib.remote.register import register_data_node_op


class TrainDataNode(Node):
    """
    A predefine structure that allows you to register operations
    on your train node in a static way before submitting them to substra.

    Inherits from :class:`connectlib.nodes.node.Node`

    Args:
        node_id (str): The substra node ID (shared with other nodes if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
    """

    def __init__(
        self,
        node_id: str,
        data_manager_key: str,
        data_sample_keys: List[str],
    ):
        self.data_manager_key = data_manager_key
        self.data_sample_keys = data_sample_keys

        super(TrainDataNode, self).__init__(node_id)

    def update_states(
        self,
        operation: DataOperation,
        local_state: Optional[LocalStateRef] = None,
    ) -> Tuple[LocalStateRef, SharedStateRef]:
        """Adding a composite train tuple to the list of operations to
        be executed by the node during the compute plan. This is done in a static
        way, nothing is submitted to substra.
        This is why the algo key is a RemoteStruct (connectlib local reference of the algorithm)
        and not a substra algo_key as nothing has been submitted yet.

        Args:
            operation (DataOperation): Automatically generated structure returned by
            :func:`connectlib.remote.methods.remote_data` decoractor. This allows to register an
            operation and execute it later on.
            local_state (Optional[LocalStateRef], optional): The parent task LocalStateRef. Defaults to None.

        Raises:
            TypeError: operation must be a DataOperation, make sure to docorate the train and predict methods of
            your algorithm with @remote

        Returns:
            Tuple[LocalStateRef, SharedStateRef]: Identifications for the results of this operation.
        """
        if not isinstance(operation, DataOperation):
            raise TypeError(
                "operation must be a DataOperation",
                f"Given: {type(operation)}",
                "Have you decorated your method with @remote_data?",
            )

        op_id = str(uuid.uuid4())

        composite_traintuple = {
            "algo_key": operation.remote_struct,
            "data_manager_key": self.data_manager_key,
            "train_data_sample_keys": operation.data_samples,
            "in_head_model_id": local_state.key if local_state is not None else None,
            "in_trunk_model_id": operation.shared_state.key
            if operation.shared_state is not None
            else None,  # user-defined id (last aggregation node task id)
            "tag": "train",
            "composite_traintuple_id": op_id,
            "metadata": dict(),  # TODO: might add info here so that on the platform we see what the tuple does ?
        }

        self.tuples.append(composite_traintuple)

        return LocalStateRef(op_id), SharedStateRef(op_id)

    def register_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        dependencies: Optional[List[str]] = None,
    ):
        """Define the algorithms for each operation and submit the composite traintuple to substra.

        Go through every operation in the computation graph, check what algorithm they use (identified by their RemoteStruct),
        submit it to substra and save the genearated algo_key to self.CACHE.
        If two tuples depend on the same algorithm, the algorithm won't be added twice to substra as
        self.CACHE keeps the submitted algo keys in memory.

        Args:
            client (substra.Client): Substra client for the node.
            permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
            dependencies (List[str]): The list of pip public dependencies your algorithm relies on (e.g. ['torch', 'pandas==1.0.1'])
        """
        for tuple in self.tuples:
            if tuple.get("out_trunk_model_permissions", None) is None:
                tuple["out_trunk_model_permissions"] = permissions

            if isinstance(tuple["algo_key"], RemoteStruct):
                remote_struct: RemoteStruct = tuple["algo_key"]

                if remote_struct not in self.CACHE:
                    # TODO : Should we remove this wrapping function ?
                    operation_key = register_data_node_op(
                        client,
                        remote_struct=remote_struct,
                        permissions=permissions,
                        dependencies=dependencies,
                    )
                    self.CACHE[remote_struct] = operation_key

                else:
                    operation_key = self.CACHE[remote_struct]

                tuple["algo_key"] = operation_key
