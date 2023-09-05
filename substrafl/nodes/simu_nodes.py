import uuid
from copy import deepcopy

from substratools import Opener
from substratools.utils import load_interface_from_module

from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef


class SimuTrainDataNode(TrainDataNode):
    def __init__(
        self, organization_id, data_manager_key, data_sample_keys, algo, client, keep_intermediate_states=False
    ):
        super().__init__(organization_id, data_manager_key, data_sample_keys)
        self.keep_intermediate_states = keep_intermediate_states
        # Make a deep copy of the input algo
        self.algo = deepcopy(algo)

        # Already load data in memory
        self._preload_data(client)

    def init_states(self, *args, **kwargs):
        op_id = str(uuid.uuid4())
        return LocalStateRef(key=op_id, init=True)

    def update_states(self, operation, *args, **kwargs):
        _method_name = operation.remote_struct._method_name
        method_parameters = operation.remote_struct._method_parameters

        method_parameters["shared_state"] = operation.shared_state
        method_parameters["datasamples"] = self._datasamples
        method_to_run = getattr(self.algo, _method_name)

        output_method = method_to_run(**method_parameters, _skip=True)
        if self.keep_intermediate_states:
            if not hasattr(self, "intermediate_states"):
                self.intermediate_states = []
            self.intermediate_states.append(deepcopy(self.algo))
        # Return the results
        op_id = str(uuid.uuid4())
        return LocalStateRef(key=op_id), output_method

    def register_operations(self, *args, **kwargs):
        return {}

    def _preload_data(self, substra_client):
        dataset_info = substra_client.get_dataset(self.data_manager_key)

        opener_interface = load_interface_from_module(
            "opener", interface_class=Opener, interface_signature=None, path=dataset_info.opener.storage_address
        )

        data_sample_paths = [substra_client.get_data_sample(dsk).path for dsk in self.data_sample_keys]

        self._datasamples = opener_interface.get_data(data_sample_paths)


class SimuTestDataNode(TestDataNode):
    def __init__(self, organization_id, data_manager_key, test_data_sample_keys, metric_functions, algo, client):
        super().__init__(organization_id, data_manager_key, test_data_sample_keys, metric_functions)
        self.algo = algo
        self._preload_data(client)
        self.scores = []

    def update_states(self, traintask_id, operation, round_idx):
        _method_name = operation.remote_struct._method_name
        method_parameters = operation.remote_struct._method_parameters

        method_parameters["shared_state"] = operation.shared_state
        method_parameters["datasamples"] = self._datasamples
        method_parameters["return_predictions"] = True
        method_to_run = getattr(self.algo, _method_name)

        predictions = method_to_run(**method_parameters, _skip=True)

        # Evaluate the predictions with all the metrics, store the scores
        for key, metric_func in self.metric_functions.items():
            self.scores.append({key: metric_func(self._datasamples, predictions), "round_idx": round_idx})
        return None

    def register_test_operations(self, *args, **kwargs):
        return {}

    def register_predict_operations(self, *args, **kwargs):
        return {}

    def _preload_data(self, substra_client):
        dataset_info = substra_client.get_dataset(self.data_manager_key)

        opener_interface = load_interface_from_module(
            "opener", interface_class=Opener, interface_signature=None, path=dataset_info.opener.storage_address
        )

        data_sample_paths = [substra_client.get_data_sample(dsk).path for dsk in self.test_data_sample_keys]

        self._datasamples = opener_interface.get_data(data_sample_paths)

    def synchronize_with_train_node(self, train_data_nodes):
        # find the train node with the correct org id
        def match_org_id(train_node):
            return train_node.organization_id == self.organization_id

        train_data_node = list(filter(match_org_id, train_data_nodes))
        assert len(train_data_node) == 1, "Each test node should be matched to a train node"
        train_data_node = train_data_node[0]
        # synchronize the algorithms
        self.algo = train_data_node.algo


class SimuAggregationNode(AggregationNode):
    def __init__(self, organization_id, strategy):
        super().__init__(organization_id)
        self.strategy = strategy

    def update_states(self, operation, *args, **kwargs):
        method_name = operation.remote_struct._method_name
        method_parameters = operation.remote_struct._method_parameters

        method_parameters["shared_states"] = operation.shared_states
        method_to_run = getattr(self.strategy, method_name)

        shared_state = method_to_run(**method_parameters, _skip=True)

        return shared_state

    def register_operations(self, *args, **kwargs):
        return {}
