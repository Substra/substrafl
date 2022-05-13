import numpy as np
import pytest
import torch

from connectlib.algorithms.pytorch.torch_base_algo import TorchAlgo
from connectlib.algorithms.pytorch.torch_fed_avg_algo import TorchFedAvgAlgo
from connectlib.algorithms.pytorch.torch_one_node_algo import TorchOneNodeAlgo
from connectlib.algorithms.pytorch.torch_scaffold_algo import TorchScaffoldAlgo
from connectlib.index_generator import NpIndexGenerator
from connectlib.remote.decorators import remote_data
from connectlib.remote.remote_struct import RemoteStruct


@pytest.fixture(params=[TorchAlgo, TorchFedAvgAlgo, TorchOneNodeAlgo, TorchScaffoldAlgo])
def dummy_algo_custom_init_arg(request):
    lin = torch.nn.Linear(3, 2)
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )

    class MyAlgo(request.param):
        def __init__(self, dummy_test_param=5):
            super().__init__(
                model=lin,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(lin.parameters(), lr=0.1),
                index_generator=nig,
                dummy_test_param=dummy_test_param,
            )
            self.dummy_test_param = dummy_test_param

        def _local_train(self, x, y):
            pass

        def _local_predict(self, x):
            return np.zeros(1)

        @property
        def strategies(self):
            return list()

        @remote_data
        def predict(self, x, shared_state):
            return self._local_predict(x)

        @remote_data
        def train(self, x, y, shared_state):
            # Return the parameter
            return self.dummy_test_param

    return MyAlgo


def test_base_algo_custom_init_arg_default_value(session_dir, dummy_algo_custom_init_arg):
    my_algo = dummy_algo_custom_init_arg()
    data_operation = my_algo.train(data_samples=["a", "b"])

    data_operation.remote_struct.save(session_dir)
    loaded_struct = RemoteStruct.load(session_dir)

    remote_struct = loaded_struct.get_remote_instance()
    _, result = remote_struct.train(X=None, y=None, head_model=None, trunk_model=None, rank=0)

    assert result == 5


@pytest.mark.parametrize("arg_value", [3, "test", np.ones(1)])
def test_base_algo_custom_init_arg(session_dir, dummy_algo_custom_init_arg, arg_value):
    my_algo = dummy_algo_custom_init_arg(dummy_test_param=arg_value)
    data_operation = my_algo.train(data_samples=["a", "b"])

    data_operation.remote_struct.save(session_dir)
    loaded_struct = RemoteStruct.load(session_dir)

    remote_struct = loaded_struct.get_remote_instance()
    _, result = remote_struct.train(X=None, y=None, head_model=None, trunk_model=None, rank=0)

    assert result == arg_value
