import uuid
from typing import Dict
from typing import List

import substra
from substra import schemas

from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import Node
from substrafl.nodes.node import OperationKey
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.operations import DataOperation
from substrafl.remote.register import register_algo
from substrafl.remote.remote_struct import RemoteStruct


class TestDataNode(Node):
    """A node on which you will test your algorithm.
    A TestDataNode must also be a train data node for now.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        test_data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
        metric_keys (List[str]):  Substra metric keys to the metrics, use substra.Client().add_algo()
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

        self.testtuples: List[Dict] = []
        self.predicttuples: List[Dict] = []

        super(TestDataNode, self).__init__(organization_id)

    def update_states(
        self,
        traintuple_id: str,
        operation: DataOperation,
        round_idx: int,
    ):
        """Creating a test tuple based on the node characteristic.

        Args:
            traintuple_id (str): The substra parent id
            operation (DataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx: (int): Round number of the strategy starting at 1.

        """

        predicttuple_id = str(uuid.uuid4())

        data_inputs = [schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)] + [
            schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.test_data_sample_keys
        ]

        predict_input = [
            schemas.InputRef(
                identifier=InputIdentifiers.local,
                parent_task_key=traintuple_id,
                parent_task_output_identifier=OutputIdentifiers.local,
            ),
            schemas.InputRef(
                identifier=InputIdentifiers.shared,
                parent_task_key=traintuple_id,
                parent_task_output_identifier=OutputIdentifiers.shared,
            ),
        ]

        test_input = [
            schemas.InputRef(
                identifier=InputIdentifiers.predictions,
                parent_task_key=predicttuple_id,
                parent_task_output_identifier=OutputIdentifiers.predictions,
            )
        ]

        predicttuple = schemas.ComputePlanTaskSpec(
            algo_key=str(uuid.uuid4()),  # bogus algo key
            task_id=predicttuple_id,
            inputs=data_inputs + predict_input,
            outputs={
                OutputIdentifiers.predictions: schemas.ComputeTaskOutputSpec(
                    permissions=schemas.Permissions(public=False, authorized_ids=[self.organization_id]),
                    transient=True,
                )
            },
            metadata={
                "round_idx": round_idx,
            },
            worker=self.organization_id,
        ).dict()

        predicttuple.pop("algo_key")
        predicttuple["remote_operation"] = operation.remote_struct
        self.predicttuples.append(predicttuple)

        for metric_key in self.metric_keys:
            testtuple = schemas.ComputePlanTaskSpec(
                algo_key=metric_key,
                task_id=str(uuid.uuid4()),
                inputs=data_inputs + test_input,
                outputs={
                    OutputIdentifiers.performance: schemas.ComputeTaskOutputSpec(
                        permissions=schemas.Permissions(public=True, authorized_ids=[]),
                        transient=False,
                    )
                },
                metadata={
                    "round_idx": round_idx,
                },
                worker=self.organization_id,
            ).dict()
            self.testtuples.append(testtuple)

    def register_predict_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Find the algorithms from the parent traintuple of each predicttuple and submit it with a dockerfile
        specifying the ``predict`` method as ``--function-name`` to execute.

        Go through every operation in the predict algo cache, submit it to substra and save `RemoteStruct : algo_key`
        into the `cache` (where algo_key is the returned algo key by substra.)
        If two predicttuples depend on the same algorithm, the algorithm won't be added twice to substra as this method
        check if an algo has already been submitted as a predicttuple to substra before adding it.

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

        for predicttuple in self.predicttuples:
            remote_struct: RemoteStruct = predicttuple["remote_operation"]
            if remote_struct not in cache:
                # Register the predictuple algorithm
                algo_key = register_algo(
                    client=client,
                    remote_struct=remote_struct,
                    permissions=permissions,
                    dependencies=dependencies,
                    inputs=[
                        schemas.AlgoInputSpec(
                            identifier=InputIdentifiers.datasamples,
                            kind=schemas.AssetKind.data_sample.value,
                            optional=False,
                            multiple=True,
                        ),
                        schemas.AlgoInputSpec(
                            identifier=InputIdentifiers.opener,
                            kind=schemas.AssetKind.data_manager.value,
                            optional=False,
                            multiple=False,
                        ),
                        schemas.AlgoInputSpec(
                            identifier=InputIdentifiers.local,
                            kind=schemas.AssetKind.model.value,
                            optional=False,
                            multiple=False,
                        ),
                        schemas.AlgoInputSpec(
                            identifier=InputIdentifiers.shared,
                            kind=schemas.AssetKind.model.value,
                            optional=False,
                            multiple=False,
                        ),
                    ],
                    outputs=[
                        schemas.AlgoOutputSpec(
                            identifier=OutputIdentifiers.predictions,
                            kind=schemas.AssetKind.model.value,
                            multiple=False,
                        )
                    ],
                )
                predicttuple["algo_key"] = algo_key
                cache[remote_struct] = algo_key
            else:
                algo_key = cache[remote_struct]
                predicttuple["algo_key"] = algo_key

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
