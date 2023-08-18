import uuid
from substrafl.nodes import TrainDataNode, TestDataNode, AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from copy import deepcopy
from substratools.utils import load_interface_from_module
from substratools import Opener


class SimuTrainDataNode(TrainDataNode):
    def __init__(self, organization_id, data_manager_key, data_sample_keys, algo, client):
        super().__init__(organization_id, data_manager_key, data_sample_keys)
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
        # Return the results
        return None, output_method

    def register_operations(self, *args, **kwargs):
        return {}

    def _preload_data(self, substra_client):
        dataset_info = substra_client.get_dataset(self.data_manager_key)

        opener_interface = load_interface_from_module(
            "opener",
            interface_class=Opener,
            interface_signature=None,
            path=dataset_info.opener.storage_address
        )

        data_sample_paths = [substra_client.get_data_sample(dsk).path for dsk in self.data_sample_keys]

        self._datasamples = opener_interface.get_data(data_sample_paths)


class SimuTestDatanode(TestDataNode):
    def __init__(self, organization_id, data_manager_key, test_data_sample_keys, metric_functions):
        super().__init__(self, organization_id, data_manager_key, test_data_sample_keys, metric_functions)

    def update_states(self, traintask_id, operation, round_idx):
        # load data
        shared_state = operation(self._datasamples, _skip=True)

        # return the results
        return None, shared_state

    def register_test_operations(self, *args, **kwargs):
        return {}

    def register_predict_operations(self, *args, **kwargs):
        return {}


class SimuAggregationNode(AggregationNode):

    def __init__(self, organization_id, strategy):
        super().__init__(organization_id)
        self.strategy = strategy

    def update_states(self, operation, *args, **kwargs):

        method_name = operation.remote_struct._method_name
        method_parameters = operation.remote_struct._method_parameters

        method_parameters["shared_states"] = operation.shared_states
        method_to_run = getattr(self.strategy, method_name)
        import pdb; pdb.set_trace()
        shared_state = method_to_run(**method_parameters, _skip=True)

        return shared_state

    def register_operations(self, *args, **kwargs):
        return {}
