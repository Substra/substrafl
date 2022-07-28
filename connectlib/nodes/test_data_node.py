import uuid
from typing import Dict
from typing import List

import substra
from substra.sdk.schemas import AlgoCategory
from substra.sdk.schemas import ComputeTaskOutput
from substra.sdk.schemas import InputRef
from substra.sdk.schemas import Permissions

from connectlib.dependency import Dependency
from connectlib.nodes.node import InputIdentifiers
from connectlib.nodes.node import Node
from connectlib.nodes.node import OperationKey
from connectlib.nodes.node import OutputIdentifiers
from connectlib.remote.register import register_algo
from connectlib.remote.remote_struct import RemoteStruct


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
        metric_keys: List[str],  # key to the metric, use substra.Client().add_algo()
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys
        self.metric_keys = metric_keys

        self.testtuples: List[Dict] = []
        self.predicttuples: List[Dict] = []

        super(TestDataNode, self).__init__(organization_id)

    def update_states(
        self,
        traintuple_id: str,
        round_idx: int,
    ):
        """Creating a test tuple based on the node characteristic.

        Args:
            traintuple_id (str): The substra parent id
            round_idx: (int): Round number of the strategy starting at 1.

        """

        predicttuple_id = str(uuid.uuid4())

        data_samples_inputs = [
            InputRef(identifier=InputIdentifiers.DATASAMPLES, asset_key=data_sample_key)
            for data_sample_key in self.test_data_sample_keys
        ]
        data_manager_input = [InputRef(identifier=InputIdentifiers.OPENER, asset_key=self.data_manager_key)]

        model_input = [
            InputRef(
                identifier=InputIdentifiers.MODEL,
                parent_task_key=traintuple_id,
                parent_task_output_identifier=OutputIdentifiers.LOCAL,
            )
        ]

        predictions_input = [
            InputRef(
                identifier=InputIdentifiers.PREDICTIONS,
                parent_task_key=predicttuple_id,
                parent_task_output_identifier=OutputIdentifiers.PREDICTIONS,
            )
        ]

        self.predicttuples.append(
            {
                "predicttuple_id": predicttuple_id,
                "traintuple_id": traintuple_id,
                "data_manager_key": self.data_manager_key,
                "test_data_sample_keys": self.test_data_sample_keys,
                "inputs": data_samples_inputs + data_manager_input + model_input,
                "outputs": {
                    OutputIdentifiers.PREDICTIONS: ComputeTaskOutput(
                        permissions=Permissions(public=False, authorized_ids=[self.organization_id])
                    )
                },
                "metadata": {
                    "round_idx": round_idx,
                },
            }
        )
        for metric_key in self.metric_keys:
            self.testtuples.append(
                {
                    "algo_key": metric_key,
                    "predicttuple_id": predicttuple_id,
                    "data_manager_key": self.data_manager_key,
                    "test_data_sample_keys": self.test_data_sample_keys,
                    "inputs": data_manager_input + data_samples_inputs + predictions_input,
                    "outputs": {
                        OutputIdentifiers.PERFORMANCE: ComputeTaskOutput(
                            permissions=Permissions(public=True, authorized_ids=[])
                        )
                    },
                    "metadata": {
                        "round_idx": round_idx,
                    },
                }
            )

    def register_predict_operations(
        self,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        traintuples: List[Dict],
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Find the algorithms from the parent traintuple of each predicttuple and submit it to substra with the predict
        algo category.

        Go through every operation in the predict algo cache, submit it to substra and save `RemoteStruct : algo_key`
        into the `cache` (where algo_key is the returned algo key by substra.)
        If two predicttuples depend on the same algorithm, the algorithm won't be added twice to substra as this method
        check if an algo has already been submitted as a predicttuple to substra before adding it.

        Args:
            client (substra.Client): Substra client for the organization.
            permissions (substra.sdk.schemas.Permissions): Permissions for the algorithm.
            traintuples: (List[Dict]): List of traintuples on which to search for the predicttuples parents.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered algorithm identifications. The key of
                each element is the RemoteStruct id (generated by connectlib) and the value is the key generated by
                substra.
            dependencies (Dependency): Algorithm dependencies.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """

        # TODO: double loop might slow down a lot large experiments. Consider refactoring.
        for predicttuple in self.predicttuples:
            for traintuple in traintuples:
                if traintuple["composite_traintuple_id"] == predicttuple["traintuple_id"]:
                    if isinstance(traintuple["remote_operation"], RemoteStruct):
                        remote_struct: RemoteStruct = traintuple["remote_operation"]

                        if remote_struct not in cache:

                            # Register the traintuple algorithm as a predict algo category.
                            algo_key = register_algo(
                                client=client,
                                category=AlgoCategory.predict,
                                remote_struct=remote_struct,
                                permissions=permissions,
                                dependencies=dependencies,
                            )
                            predicttuple["algo_key"] = algo_key
                            cache[remote_struct] = algo_key
                            break
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
