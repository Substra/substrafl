import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.algorithms.pytorch.torch_fed_avg_algo import TorchFedAvgAlgo
from substrafl.algorithms.pytorch.torch_newton_raphson_algo import TorchNewtonRaphsonAlgo
from substrafl.algorithms.pytorch.torch_scaffold_algo import TorchScaffoldAlgo
from substrafl.algorithms.pytorch.torch_single_organization_algo import TorchSingleOrganizationAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import BatchSizeNotFoundError
from substrafl.exceptions import DatasetSignatureError
from substrafl.exceptions import DatasetTypeError
from substrafl.index_generator import NpIndexGenerator
from substrafl.nodes.schemas import InputIdentifiers
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.remote.decorators import remote_data
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.remote.serializers import PickleSerializer
from substrafl.schemas import TaskType
from substrafl.strategies import FedAvg
from substrafl.strategies import NewtonRaphson
from substrafl.strategies import Scaffold
from substrafl.strategies import SingleOrganization
from substrafl.strategies.schemas import StrategyName
from substrafl.strategies.strategy import Strategy
from tests.conftest import LINEAR_N_COL
from tests.conftest import LINEAR_N_TARGET


@pytest.fixture(params=[None, 31, 42])
def rng_algo(request, torch_linear_model, numpy_torch_dataset):
    test_seed = request.param
    n_rng_sample = 10

    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )
    perceptron = torch_linear_model()

    class RngAlgo(TorchAlgo):
        def __init__(self):
            super().__init__(
                model=perceptron,
                dataset=numpy_torch_dataset,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                index_generator=nig,
                seed=test_seed,
            )

        @property
        def strategies(self):
            return ["rng_strategy"]

        @remote_data
        def train(self, data_from_opener, shared_state=None):
            return torch.rand(n_rng_sample)

    return RngAlgo, test_seed


@pytest.fixture
def rng_strategy():
    class RngStrategy(Strategy):
        def __init__(self, algo):
            self.algo = algo
            self._local_states = None
            self._shared_states = None

        @property
        def name(self) -> StrategyName:
            return "rng_strategy"

        def perform_round(
            self,
            train_data_nodes,
            aggregation_node,
            round_idx,
            clean_models: bool,
            additional_orgs_permissions=None,
        ):
            next_local_states = []
            next_shared_states = []

            if round_idx == 0:
                return

            for i, node in enumerate(train_data_nodes):
                next_local_state, next_shared_state = node.update_states(
                    operation=self.algo.train(
                        node.data_sample_keys,
                    ),
                    round_idx=round_idx,
                    authorized_ids={node.organization_id},
                    local_state=self._local_states[i] if self._local_states is not None else None,
                )

                # keep the states in a list: one/organization
                next_local_states.append(next_local_state)
                next_shared_states.append(next_shared_state)

            self._local_states = next_local_states
            self._shared_states = next_shared_states

        def perform_evaluation(
            self,
            test_data_nodes,
            train_data_nodes,
            round_idx: int,
        ):
            pass

    return RngStrategy


@pytest.fixture(
    params=[TorchAlgo, TorchFedAvgAlgo, TorchSingleOrganizationAlgo, TorchScaffoldAlgo, TorchNewtonRaphsonAlgo]
)
def dummy_algo_custom_init_arg(request, numpy_torch_dataset):
    lin = torch.nn.Linear(3, 2)
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )

    class MyAlgo(request.param):
        def __init__(self, dummy_test_param=5):
            if isinstance(self, TorchNewtonRaphsonAlgo):
                super().__init__(
                    model=lin,
                    criterion=torch.nn.MSELoss(),
                    dataset=numpy_torch_dataset,
                    batch_size=1,
                    dummy_test_param=dummy_test_param,
                )
            else:
                super().__init__(
                    model=lin,
                    criterion=torch.nn.MSELoss(),
                    optimizer=torch.optim.SGD(lin.parameters(), lr=0.1),
                    index_generator=nig,
                    dataset=numpy_torch_dataset,
                    dummy_test_param=dummy_test_param,
                )
            self.dummy_test_param = dummy_test_param

        @property
        def strategies(self):
            return list()

        @remote_data
        def train(self, data_from_opener, shared_state):
            # Return the parameter
            return self.dummy_test_param

    return MyAlgo


@pytest.fixture(params=[pytest.param(False, marks=pytest.mark.gpu), True])
def disable_gpu(request):
    return request.param


@pytest.fixture(
    params=[
        (TorchFedAvgAlgo, FedAvg),
        (TorchSingleOrganizationAlgo, SingleOrganization),
        (TorchScaffoldAlgo, Scaffold),
        (TorchNewtonRaphsonAlgo, NewtonRaphson),
    ]
)
def dummy_gpu(request, torch_linear_model, disable_gpu, numpy_torch_dataset):
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )
    perceptron = torch_linear_model()

    class MyAlgo(request.param[0]):
        def __init__(self):
            if isinstance(self, TorchNewtonRaphsonAlgo):
                super().__init__(
                    model=perceptron,
                    criterion=torch.nn.MSELoss(),
                    dataset=numpy_torch_dataset,
                    batch_size=1,
                    disable_gpu=disable_gpu,
                )
            else:
                super().__init__(
                    model=perceptron,
                    optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                    criterion=torch.nn.MSELoss(),
                    dataset=numpy_torch_dataset,
                    index_generator=nig,
                    disable_gpu=disable_gpu,
                )
            if disable_gpu or not torch.cuda.is_available():
                assert self._device == torch.device("cpu")
            else:
                assert self._device == torch.device("cuda")

        @property
        def strategies(self):
            return list(StrategyName)

    return MyAlgo, request.param[1], disable_gpu


def test_base_algo_custom_init_arg_default_value(session_dir, dummy_algo_custom_init_arg):
    my_algo = dummy_algo_custom_init_arg()
    data_operation = my_algo.train(data_samples=["a", "b"])

    data_operation.remote_struct.save(session_dir)
    loaded_struct = RemoteStruct.load(session_dir)

    remote_struct = loaded_struct.get_remote_instance()
    inputs = {
        InputIdentifiers.datasamples: None,
        InputIdentifiers.local: None,
        InputIdentifiers.shared: None,
    }
    outputs = {
        OutputIdentifiers.local: session_dir / OutputIdentifiers.local,
        OutputIdentifiers.shared: session_dir / OutputIdentifiers.shared,
    }
    task_properties = {
        InputIdentifiers.rank: 0,
    }
    remote_struct.generic_function(inputs, outputs, task_properties)

    result = remote_struct.load_shared(outputs[OutputIdentifiers.shared])

    assert result == 5


@pytest.mark.parametrize("arg_value", [3, "test", np.ones(1)])
def test_base_algo_custom_init_arg(session_dir, dummy_algo_custom_init_arg, arg_value):
    my_algo = dummy_algo_custom_init_arg(dummy_test_param=arg_value)
    data_operation = my_algo.train(data_samples=["a", "b"])

    data_operation.remote_struct.save(session_dir)
    loaded_struct = RemoteStruct.load(session_dir)

    remote_struct = loaded_struct.get_remote_instance()
    inputs = {
        InputIdentifiers.datasamples: None,
        InputIdentifiers.local: None,
        InputIdentifiers.shared: None,
        InputIdentifiers.rank: 0,
    }
    outputs = {
        OutputIdentifiers.local: session_dir / OutputIdentifiers.local,
        OutputIdentifiers.shared: session_dir / OutputIdentifiers.shared,
    }
    task_properties = {
        InputIdentifiers.rank: 0,
    }
    remote_struct.generic_function(inputs, outputs, task_properties)

    result = remote_struct.load_shared(outputs[OutputIdentifiers.shared])
    assert result == arg_value


@pytest.mark.substra
def test_rng_state_save_and_load(
    network, train_linear_nodes, session_dir, rng_strategy, rng_algo, torch_cpu_dependency
):
    """
    Test that the RNG state is well incremented through the different rounds.
    """
    n_rng_sample = 10

    algo_class, test_seed = rng_algo
    my_algo = algo_class()

    if test_seed is not None:
        torch.manual_seed(test_seed)

    expected_output_round_1 = torch.rand(n_rng_sample)
    expected_output_round_2 = torch.rand(n_rng_sample)

    strategy = rng_strategy(algo=my_algo)

    cp = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=[train_linear_nodes[0]],
        evaluation_strategy=None,
        aggregation_node=None,
        num_rounds=2,
        dependencies=torch_cpu_dependency,
        experiment_folder=session_dir / "experiment_folder",
    )
    network.clients[0].wait_compute_plan(cp.key)

    output_model = {}

    for task in network.clients[0].list_task(filters={"compute_plan_key": [cp.key]}):
        if task.tag == TaskType.TRAIN:
            download_path = network.clients[0].download_model_from_task(
                task.key,
                folder=session_dir,
                identifier=OutputIdentifiers.shared,
            )
            output_model[task.metadata["round_idx"]] = PickleSerializer().load(download_path)

    if test_seed is not None:
        assert all(output_model["1"] == expected_output_round_1)
        assert all(output_model["2"] == expected_output_round_2)
    else:
        assert not all(output_model["1"] == expected_output_round_1)
        assert not all(output_model["2"] == expected_output_round_2)


@pytest.mark.parametrize("n_samples", [1, 2])
def test_check_predict_shapes(n_samples, test_linear_data_samples, numpy_torch_dataset, torch_linear_model, seed):
    """Checks that one liner and multiple liners input can be used for inference (corner case of last batch
    shape is (1, n_cols)"""

    num_updates = 100
    torch.manual_seed(seed)
    perceptron = torch_linear_model()
    nig = NpIndexGenerator(batch_size=n_samples, num_updates=num_updates, drop_last=False)

    class MyAlgo(TorchAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                model=perceptron,
                index_generator=nig,
                dataset=numpy_torch_dataset,
            )

        def train():
            pass

        @property
        def strategies(self):
            return []

    my_algo = MyAlgo()

    x = test_linear_data_samples[0][:n_samples, :-LINEAR_N_TARGET]
    y = test_linear_data_samples[0][:n_samples, -LINEAR_N_TARGET:]

    res = my_algo.predict(data_from_opener=(x, y))
    assert res.shape == (n_samples, 1)


@pytest.mark.parametrize(
    "init_function, is_valid",
    [
        ((lambda self, data_from_opener, is_inference: None), True),
        ((lambda self, not_data_from_opener, is_inference: None), False),
    ],
)
def test_signature_error_torch_dataset(init_function, is_valid):
    lin = torch.nn.Linear(3, 2)
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )

    class TorchDataset(torch.utils.data.Dataset):
        __init__ = init_function

    class MyAlgo(TorchAlgo):
        def __init__(self):
            super().__init__(
                model=lin,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(lin.parameters(), lr=0.1),
                index_generator=nig,
                dataset=TorchDataset,
            )

        @property
        def strategies(self):
            return list()

        def predict(self, data_from_opener, shared_state):
            pass

        def train(self, data_from_opener, shared_state):
            pass

    if is_valid:
        MyAlgo()
    else:
        with pytest.raises(DatasetSignatureError):
            MyAlgo()


def test_instance_error_torch_dataset():
    lin = torch.nn.Linear(3, 2)
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=1,
    )

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, data_from_opener, is_inference):
            pass

    class MyAlgo(TorchAlgo):
        def __init__(self):
            super().__init__(
                model=lin,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(lin.parameters(), lr=0.1),
                index_generator=nig,
                dataset=TorchDataset(0, False),
            )

        @property
        def strategies(self):
            return list()

        def train(self, data_from_opener, shared_state):
            pass

    with pytest.raises(DatasetTypeError):
        MyAlgo()


def test_none_index_generator_for_predict(numpy_torch_dataset):
    lin = torch.nn.Linear(3, 2)

    class MyAlgo(TorchAlgo):
        def __init__(self):
            super().__init__(
                model=lin,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(lin.parameters(), lr=0.1),
                index_generator=None,
                dataset=numpy_torch_dataset,
            )

        @property
        def strategies(self):
            return list()

        def train(self, data_from_opener, shared_state):
            pass

    my_algo = MyAlgo()

    with pytest.raises(BatchSizeNotFoundError):
        my_algo.predict(data_from_opener=(np.zeros((1, LINEAR_N_COL)), np.zeros((1, LINEAR_N_TARGET))))


@pytest.mark.substra
def test_gpu(
    dummy_gpu,
    mae_metric,
    session_dir,
    network,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
):
    num_rounds = 2
    algo_class, strategy_class, disable_gpu = dummy_gpu
    my_algo = algo_class()
    algo_deps = Dependency(
        pypi_dependencies=["torch==2.4.1", "numpy==2.2.1", "pytest"],
        editable_mode=True,
    )

    train_data_nodes = [train_linear_nodes[0]] if strategy_class == SingleOrganization else train_linear_nodes
    test_data_nodes = [test_linear_nodes[0]] if strategy_class == SingleOrganization else test_linear_nodes

    strategy = (
        strategy_class(algo=my_algo, metric_functions=mae_metric, damping_factor=0.1)
        if strategy_class == NewtonRaphson
        else strategy_class(algo=my_algo, metric_functions=mae_metric)
    )

    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_data_nodes, eval_rounds=[num_rounds]  # test only at the last round
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
        clean_models=False,
        name=f'Testing the GPU - strategy {strategy_class.__name__}, running  on \
            {"cpu" if (disable_gpu or not torch.cuda.is_available()) else "cuda"}',
    )

    # Wait for the compute plan to be finished
    network.clients[0].wait_compute_plan(compute_plan.key)
