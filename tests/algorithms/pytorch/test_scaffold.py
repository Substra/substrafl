import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl.algorithms.pytorch import TorchScaffoldAlgo
from substrafl.algorithms.pytorch.weight_manager import increment_parameters
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import NumUpdatesValueError
from substrafl.exceptions import TorchScaffoldAlgoParametersUpdateError
from substrafl.index_generator import NpIndexGenerator
from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo
from substrafl.strategies import Scaffold
from tests import utils
from tests.algorithms.pytorch.torch_tests_utils import assert_model_parameters_equal
from tests.algorithms.pytorch.torch_tests_utils import assert_tensor_list_equal
from tests.algorithms.pytorch.torch_tests_utils import assert_tensor_list_not_zeros

logger = logging.getLogger(__name__)
current_folder = Path(__file__).parent

EXPECTED_PERFORMANCE = 0.0127768706
NUM_ROUNDS = 3


def _torch_algo(torch_linear_model, numpy_torch_dataset, seed, lr=0.1, use_scheduler=False):
    num_updates = 100
    torch.manual_seed(seed)
    perceptron = torch_linear_model()
    nig = NpIndexGenerator(
        batch_size=32,
        num_updates=num_updates,
    )
    optimizer = torch.optim.SGD(perceptron.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) if use_scheduler else None

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                model=perceptron,
                index_generator=nig,
                dataset=numpy_torch_dataset,
                scheduler=scheduler,
            )

    return MyAlgo


@pytest.fixture(scope="module")
def torch_algo(torch_linear_model, numpy_torch_dataset, seed):
    """This closure allows to parametrize the torch algo fixture"""

    def inner_torch_algo(lr=0.1, use_scheduler=False):
        return _torch_algo(
            torch_linear_model=torch_linear_model,
            numpy_torch_dataset=numpy_torch_dataset,
            lr=lr,
            use_scheduler=use_scheduler,
            seed=seed,
        )()

    return inner_torch_algo


@pytest.fixture(scope="module")
def compute_plan(torch_algo, train_linear_nodes, test_linear_nodes, aggregation_node, network, session_dir):
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )

    strategy = Scaffold(algo=torch_algo())
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes, eval_rounds=[0, NUM_ROUNDS]
    )  # test the initialization and the last round

    compute_plan = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
        clean_models=False,
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    return compute_plan


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_scaffold_algo_weights(
    network,
    compute_plan,
    torch_algo,
    session_dir,
):
    """Check the weight initialization, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """

    my_algo = torch_algo()

    rank_1_local_models = utils.download_train_task_models_by_rank(network, session_dir, my_algo, compute_plan, rank=1)
    rank_3_local_models = utils.download_train_task_models_by_rank(network, session_dir, my_algo, compute_plan, rank=3)

    # Download the aggregate output
    aggregate_model = utils.download_aggregate_model_by_rank(network, session_dir, compute_plan, rank=2)
    aggregate_update = [torch.from_numpy(x).to("cpu") for x in aggregate_model.avg_parameters_update]

    # Assert the model initialization is the same for every model
    assert_model_parameters_equal(rank_1_local_models[0].model, rank_1_local_models[1].model)
    assert_tensor_list_equal(
        rank_1_local_models[0]._client_control_variate, rank_1_local_models[1]._client_control_variate
    )

    # assert the _client_control_variate have been updated
    assert_tensor_list_not_zeros(rank_1_local_models[0]._client_control_variate)
    assert_tensor_list_not_zeros(rank_1_local_models[1]._client_control_variate)

    # Assert that the weights are well set
    for model_1, model_3 in zip(rank_1_local_models, rank_3_local_models):
        increment_parameters(model=model_1.model, updates=aggregate_update, with_batch_norm_parameters=True)
        assert_model_parameters_equal(model_1.model, model_3.model)

    # The local models and _client_control_variate are always the same on every organization, as both organizations have
    # the same data
    assert_model_parameters_equal(rank_3_local_models[0].model, rank_3_local_models[1].model)
    assert_tensor_list_equal(
        rank_3_local_models[0]._client_control_variate, rank_3_local_models[1]._client_control_variate
    )


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_scaffold_algo_performance(
    network,
    compute_plan,
    torch_linear_model,
    test_linear_data_samples,
    mae,
    rtol,
    seed,
):
    """End to end test for torch scaffold algorithm."""

    perfs = network.clients[0].get_performances(compute_plan.key)
    assert pytest.approx(EXPECTED_PERFORMANCE, rel=rtol) == perfs.performance[1]

    torch.manual_seed(seed)

    model = torch_linear_model()
    y_pred = model(torch.from_numpy(test_linear_data_samples[0][:, :-1]).float()).detach().numpy().reshape(-1)
    y_true = test_linear_data_samples[0][:, -1]

    performance_at_init = mae(y_pred, y_true)
    assert performance_at_init == pytest.approx(perfs.performance[0], abs=rtol)


@pytest.mark.parametrize("use_scheduler", [True, False])
def test_update_current_lr(rtol, torch_algo, use_scheduler, seed):
    # test the update_current_lr() fct with optimizer only and optimizer+scheduler
    torch.manual_seed(seed)
    initial_lr = 0.5
    my_algo = torch_algo(lr=initial_lr, use_scheduler=use_scheduler)

    my_algo._update_current_lr()
    assert pytest.approx(my_algo._current_lr, rel=rtol) == initial_lr

    if use_scheduler:
        my_algo._scheduler.step()
        my_algo._update_current_lr()
        assert pytest.approx(my_algo._current_lr, rel=rtol) == initial_lr * my_algo._scheduler.gamma


@pytest.mark.parametrize("num_updates", [-10, 0])
def test_pytorch_num_updates_error(num_updates, numpy_torch_dataset):
    """Check that num_updates <= 0 raise a ValueError."""
    nig = NpIndexGenerator(
        batch_size=32,
        num_updates=num_updates,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1, bias=False)

        def forward(self, x):
            out = self.linear1(x)
            return out

    dummy_model = DummyModel()

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=dummy_model,
                index_generator=nig,
                optimizer=torch.optim.SGD(dummy_model.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                dataset=numpy_torch_dataset,
            )

    with pytest.raises(NumUpdatesValueError):
        MyAlgo()


@pytest.mark.parametrize("optimizer", [torch.optim.Adagrad, torch.optim.Adam])
def test_pytorch_optimizer_error(optimizer, torch_linear_model, caplog, numpy_torch_dataset):
    "Only SGD is recommended as an optimizer for TorchScaffoldAlgo."
    perceptron = torch_linear_model()
    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=2,
    )

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=perceptron,
                index_generator=nig,
                optimizer=optimizer(perceptron.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                dataset=numpy_torch_dataset,
            )

    caplog.clear()
    MyAlgo()
    warnings = [
        record
        for record in caplog.records
        if (record.levelname == "WARNING") and (record.msg.startswith("The only optimizer theoretically guaranteed"))
    ]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "lr1, lr2, expected_lr, nb_warnings", [(0.2, 0.1, 0.1, 1), (0.4, 0, 0.4, 0), (0.5, 0.5, 0.5, 0)]
)
def test_pytorch_multiple_lr(lr1, lr2, expected_lr, nb_warnings, caplog, numpy_torch_dataset):
    "Check that the smallest (but 0) learning rate is used for the aggregation when multiple learning rate are used."

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 2)
            self.linear2 = torch.nn.Linear(2, 1)

        def forward(self, x):
            l1 = self.linear1(x)
            out = self.linear2(l1)
            return out

    model = MLP()

    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=2,
    )

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=model,
                index_generator=nig,
                optimizer=torch.optim.SGD(
                    [
                        {"params": model.linear1.parameters(), "lr": lr1},
                        {"params": model.linear2.parameters()},
                    ],
                    lr=lr2,
                ),
                criterion=torch.nn.MSELoss(),
                dataset=numpy_torch_dataset,
            )

    my_algo = MyAlgo()

    caplog.clear()
    my_algo._update_current_lr()
    warnings = [record for record in caplog.records if (record.levelname == "WARNING")]
    assert my_algo._current_lr == expected_lr
    assert len(warnings) == nb_warnings


@pytest.mark.parametrize("nb_update_params_call, num_updates", ([0, 1], [1, 2], [3, 5]))
def test_update_parameters_call(nb_update_params_call, torch_linear_model, num_updates, numpy_torch_dataset):
    "Check that _scaffold_parameters_update needs to be called at each update"

    model = torch_linear_model()

    nig = NpIndexGenerator(
        batch_size=1,
        num_updates=num_updates,
    )

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=model,
                index_generator=nig,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                dataset=numpy_torch_dataset,
            )

        def _local_train(self, train_dataset):
            for _ in self._index_generator:
                continue
            for _ in range(nb_update_params_call):
                self._scaffold_parameters_update()

    my_algo = MyAlgo()

    with pytest.raises(TorchScaffoldAlgoParametersUpdateError):
        my_algo.train(datasamples=np.random.rand(2, 10), _skip=True)

    assert my_algo._scaffold_parameters_update_num_call == nb_update_params_call


@pytest.mark.slow
@pytest.mark.substra
def test_download_load_algo(network, compute_plan, session_dir, test_linear_data_samples, mae, rtol):
    download_algo_files(
        client=network.clients[0],
        task_type="train",
        compute_plan_key=compute_plan.key,
        round_idx=NUM_ROUNDS,
        dest_folder=session_dir,
    )
    model = load_algo(input_folder=session_dir)._model

    y_pred = model(torch.from_numpy(test_linear_data_samples[0][:, :-1]).float()).detach().numpy().reshape(-1)
    y_true = test_linear_data_samples[0][:, -1:].reshape(-1)
    performance = mae(y_pred, y_true)

    assert performance == pytest.approx(EXPECTED_PERFORMANCE, rel=rtol)
