from typing import Any

import numpy as np
import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch.torch_base_algo import TorchAlgo
from connectlib.algorithms.pytorch.torch_fed_avg_algo import TorchFedAvgAlgo
from connectlib.algorithms.pytorch.torch_scaffold_algo import TorchScaffoldAlgo
from connectlib.algorithms.pytorch.torch_single_organization_algo import TorchSingleOrganizationAlgo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.index_generator import NpIndexGenerator
from connectlib.remote.decorators import remote_data
from connectlib.remote.remote_struct import RemoteStruct
from connectlib.schemas import StrategyName
from connectlib.strategies import FedAvg
from connectlib.strategies import Scaffold
from connectlib.strategies import SingleOrganization
from tests import utils


@pytest.fixture(params=[TorchAlgo, TorchFedAvgAlgo, TorchSingleOrganizationAlgo, TorchScaffoldAlgo])
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


@pytest.fixture(params=[True, False])
def use_gpu(request):
    return request.param


@pytest.fixture(
    params=[(TorchFedAvgAlgo, FedAvg), (TorchSingleOrganizationAlgo, SingleOrganization), (TorchScaffoldAlgo, Scaffold)]
)
def dummy_gpu(request, torch_linear_model, use_gpu):
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )
    perceptron = torch_linear_model()

    class MyAlgo(request.param[0]):
        def __init__(self):
            super().__init__(
                model=perceptron,
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                index_generator=nig,
                use_gpu=use_gpu,
            )

        def _local_train(self, x: Any, y: Any):
            if use_gpu:
                assert self._device == torch.device("cuda")
            else:
                assert self._device == torch.device("cpu")
            super()._local_train(
                torch.from_numpy(x).float().to(self._device), torch.from_numpy(y).float().to(self._device)
            )

        def _local_predict(self, x: Any) -> Any:
            if use_gpu:
                assert self._device == torch.device("cuda")
            else:
                assert self._device == torch.device("cpu")
            y_pred = super()._local_predict(torch.from_numpy(x).float().to(self._device))
            return y_pred.cpu().detach().numpy()

        @property
        def strategies(self):
            return list(StrategyName)

    return MyAlgo, request.param[1], use_gpu


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


@pytest.mark.gpu
@pytest.mark.substra
def test_gpu(
    dummy_gpu,
    session_dir,
    network,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
):
    num_rounds = 2
    algo_class, strategy_class, use_gpu = dummy_gpu
    my_algo = algo_class()
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy", "pytest"],
        editable_mode=True,
    )

    train_data_nodes = [train_linear_nodes[0]] if strategy_class == SingleOrganization else train_linear_nodes
    test_data_nodes = [test_linear_nodes[0]] if strategy_class == SingleOrganization else test_linear_nodes

    strategy = strategy_class()
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_data_nodes, rounds=[num_rounds]  # test only at the last round
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
        clean_models=False,
        name=f'Testing the GPU - strategy {strategy_class.__name__}, running  on {"cuda" if use_gpu else "cpu"}',
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)
