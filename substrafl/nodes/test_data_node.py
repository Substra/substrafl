import uuid
from typing import Dict
from typing import List

import substra

from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import Node
from substrafl.nodes.node import OperationKey
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.register import register_function
from substrafl.remote.remote_struct import RemoteStruct


class TestDataNode(Node):
    """A node on which you will test your algorithm.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        test_data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
        metric_keys (List[str]):  Keys of the functions that implement the different metrics. See
            :py:func:`~substrafl.remote.register.register.add_metric` for more information on how to register metric
            functions.
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        test_data_sample_keys: List[str],
        metric_keys: List[str],
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys

        if not isinstance(metric_keys, list):
            raise TypeError("metric keys must be of type list")
        self.metric_keys = metric_keys

        self.testtasks: List[Dict] = []
        self.predicttasks: List[Dict] = []

        super().__init__(organization_id)

    def update_states(
        self,
        traintask_id: str,
        operation: RemoteDataOperation,
        round_idx: int,
    ):
        """Creating a test task based on the node characteristic.

        Args:
            traintask_id (str): The substra parent id
            operation (RemoteDataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx: (int): Round number of the strategy starting at 1.

        """

        predicttask_id = str(uuid.uuid4())

        data_inputs = [
            substra.schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)
        ] + [
            substra.schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.test_data_sample_keys
        ]
        predict_input = [
            substra.schemas.InputRef(
                identifier=InputIdentifiers.local,
                parent_task_key=traintask_id,
                parent_task_output_identifier=OutputIdentifiers.local,
            ),
        ]

        test_input = [
            substra.schemas.InputRef(
                identifier=InputIdentifiers.predictions,
                parent_task_key=predicttask_id,
                parent_task_output_identifier=OutputIdentifiers.predictions,
            )
        ]

        predicttask = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=predicttask_id,
            inputs=data_inputs + predict_input,
            outputs={
                OutputIdentifiers.predictions: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=False, authorized_ids=[self.organization_id]),
                    transient=True,
                )
            },
            metadata={
                "round_idx": round_idx,
            },
            worker=self.organization_id,
        ).dict()

        predicttask.pop("function_key")
        predicttask["remote_operation"] = operation.remote_struct
        self.predicttasks.append(predicttask)

        for metric_key in self.metric_keys:
            testtask = substra.schemas.ComputePlanTaskSpec(
                function_key=metric_key,
                task_id=str(uuid.uuid4()),
                inputs=data_inputs + test_input,
                outputs={
                    OutputIdentifiers.performance: substra.schemas.ComputeTaskOutputSpec(
                        permissions=substra.schemas.Permissions(public=True, authorized_ids=[]),
                        transient=False,
                    )
                },
                metadata={
                    "round_idx": round_idx,
                },
                worker=self.organization_id,
            ).dict()
            self.testtasks.append(testtask)

    def register_predict_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Find the functions from the parent traintask of each predicttask and submit it with a dockerfile
        specifying the ``predict`` method as ``--function-name`` to execute.

        Go through every operation in the predict function cache, submit it to substra and save
        `RemoteStruct : function_key` into the `cache` (where function_key is the returned function key by substra.)
        If two predicttasks depend on the same function, the function won't be added twice to substra as this method
        check if an function has already been submitted as a predicttask to substra before adding it.

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

        for predicttask in self.predicttasks:
            remote_struct: RemoteStruct = predicttask["remote_operation"]
            if remote_struct not in cache:
                # Register the predictask function
                function_key = register_function(
                    client=client,
                    remote_struct=remote_struct,
                    permissions=permissions,
                    dependencies=dependencies,
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
                            identifier=OutputIdentifiers.predictions,
                            kind=substra.schemas.AssetKind.model.value,
                            multiple=False,
                        )
                    ],
                )
                predicttask["function_key"] = function_key
                cache[remote_struct] = function_key
            else:
                function_key = cache[remote_struct]
                predicttask["function_key"] = function_key

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
                "data_sample_keys": self.test_data_sample_keys,
                "metric_keys": self.metric_keys,
            }
        )
        return summary
