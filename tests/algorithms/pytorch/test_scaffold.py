import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchScaffoldAlgo
from connectlib.algorithms.pytorch.weight_manager import increment_parameters
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.exceptions import NumUpdatesValueError
from connectlib.exceptions import TorchScaffoldAlgoParametersUpdateError
from connectlib.index_generator import NpIndexGenerator
from connectlib.schemas import ScaffoldAveragedStates
from connectlib.schemas import ScaffoldSharedState
from connectlib.strategies import Scaffold
from tests import utils
from tests.algorithms.pytorch.torch_tests_utils import assert_model_parameters_equal
from tests.algorithms.pytorch.torch_tests_utils import assert_tensor_list_equal
from tests.algorithms.pytorch.torch_tests_utils import assert_tensor_list_not_zeros

logger = logging.getLogger(__name__)
current_folder = Path(__file__).parent


def _torch_algo(torch_linear_model, lr=0.1, use_scheduler=False):
    num_updates = 100
    seed = 42
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
                scheduler=scheduler,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    return MyAlgo


@pytest.fixture(scope="module")
def torch_algo(torch_linear_model):
    """This closure allows to parametrize the torch algo fixture"""

    def inner_torch_algo(lr=0.1, use_scheduler=False):
        return _torch_algo(torch_linear_model=torch_linear_model, lr=lr, use_scheduler=use_scheduler)()

    return inner_torch_algo


@pytest.fixture(scope="module")
def compute_plan(torch_algo, train_linear_nodes, test_linear_nodes, aggregation_node, network, session_dir):

    num_rounds = 3

    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )

    strategy = Scaffold()
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes, rounds=[num_rounds]  # test only at the last round
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=torch_algo(),
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
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
    """Check the weight initialisation, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """

    my_algo = torch_algo()

    rank_0_local_models = utils.download_composite_models_by_rank(network, session_dir, my_algo, compute_plan, rank=0)
    rank_2_local_models = utils.download_composite_models_by_rank(network, session_dir, my_algo, compute_plan, rank=2)

    # Download the aggregate output
    aggregate_task = network.clients[0].list_aggregatetuple(
        filters=[f"aggregatetuple:compute_plan_key:{compute_plan.key}", f"aggregatetuple:rank:{1}"]
    )[0]
    model_key = aggregate_task.aggregate.models[0].key
    network.clients[0].download_model(model_key, session_dir)
    model_path = session_dir / f"model_{model_key}"
    aggregate_model = pickle.loads(model_path.read_bytes())
    aggregate_update = [torch.from_numpy(x).to("cpu") for x in aggregate_model.avg_parameters_update]

    # Assert the model initialisation is the same for every model
    assert_model_parameters_equal(rank_0_local_models[0].model, rank_0_local_models[1].model)
    assert_tensor_list_equal(
        rank_0_local_models[0]._client_control_variate, rank_0_local_models[1]._client_control_variate
    )

    # assert the _client_control_variate have been updated
    assert_tensor_list_not_zeros(rank_0_local_models[0]._client_control_variate)
    assert_tensor_list_not_zeros(rank_0_local_models[1]._client_control_variate)

    # Assert that the weights are well set
    for model_0, model_2 in zip(rank_0_local_models, rank_2_local_models):
        increment_parameters(model_0.model, aggregate_update, with_batch_norm_parameters=True)
        assert_model_parameters_equal(model_0.model, model_2.model)

    # The local models and _client_control_variate are always the same on every organization, as both organizations have
    # the same data
    assert_model_parameters_equal(rank_2_local_models[0].model, rank_2_local_models[1].model)
    assert_tensor_list_equal(
        rank_2_local_models[0]._client_control_variate, rank_2_local_models[1]._client_control_variate
    )


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_scaffold_algo_performance(
    network,
    compute_plan,
    rtol,
):
    """End to end test for torch fed avg algorithm."""

    expected_performance = 0.0127768706

    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuple = testtuples[0]
    assert list(testtuple.test.perfs.values())[0] == pytest.approx(expected_performance, rel=rtol)


@pytest.mark.skip("Can't use inference mode because of the set weight function")
# TODO: fixe
def test_train_skip(rtol):
    # check the results of the train function with simple data and model
    torch.manual_seed(42)
    n_samples = 32

    class Perceptron(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1, bias=False)
            # we init the weights at 0
            self.linear1.weight.data.fill_(0)

        def forward(self, x):
            out = self.linear1(x)
            return out

    dummy_model = Perceptron()
    nig = NpIndexGenerator(
        batch_size=n_samples,
        num_updates=2,
        shuffle=True,
        drop_last=False,
    )

    class MyAlgo(TorchScaffoldAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=torch.optim.SGD(dummy_model.parameters(), lr=0.5),
                criterion=torch.nn.MSELoss(),
                model=dummy_model,
                index_generator=nig,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    my_algo = MyAlgo()

    # we generate linear data x=ay+b with a = 2, b = 0
    a = 2
    x = np.ones((n_samples, 1))  # ones
    y = np.ones((n_samples, 1)) * a  # twos
    shared_states: ScaffoldSharedState = my_algo.train(x=x, y=y, shared_state=None, _skip=True)

    # the model should overfit so that weight_updates (= the new weighs) = a in x=ay+b
    assert np.allclose(a, shared_states.parameters_update, rtol=rtol)
    # lr * num_updates = 1 so control_variate_update = - parameters_update
    assert np.allclose(-1 * a, shared_states.control_variate_update, rtol=rtol)
    assert np.allclose(n_samples, shared_states.n_samples, rtol)
    # server_control_variate is init to zero should not be modified
    assert np.allclose(np.array([[0.0]]), shared_states.server_control_variate, rtol)

    # we create a ScaffoldAveragedStates with the output state of the train function and predict on x
    avg_shared_states = ScaffoldAveragedStates(
        server_control_variate=shared_states.control_variate_update,
        avg_parameters_update=shared_states.parameters_update,
    )
    my_algo.train(x=x, y=y, shared_state=avg_shared_states, _skip=True)

    predictions = my_algo.predict(x=x, shared_state=None, _skip=True)

    # the model should overfit so that predictions = y
    assert np.allclose(y, predictions, rtol=rtol)


@pytest.mark.parametrize("use_scheduler", [True, False])
def test_update_current_lr(rtol, torch_algo, use_scheduler):
    # test the update_current_lr() fct with optimizer only and optimizer+scheduler
    torch.manual_seed(42)
    initial_lr = 0.5
    my_algo = torch_algo(lr=initial_lr, use_scheduler=use_scheduler)

    my_algo._update_current_lr()
    assert pytest.approx(my_algo._current_lr, rel=rtol) == initial_lr

    if use_scheduler:
        my_algo._scheduler.step()
        my_algo._update_current_lr()
        assert pytest.approx(my_algo._current_lr, rel=rtol) == initial_lr * my_algo._scheduler.gamma


@pytest.mark.parametrize("num_updates", [-10, 0])
def test_pytorch_num_updates_error(num_updates):
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
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    with pytest.raises(NumUpdatesValueError):
        MyAlgo()


@pytest.mark.parametrize("optimizer", [torch.optim.Adagrad, torch.optim.Adam])
def test_pytorch_optimizer_error(optimizer, torch_linear_model, caplog):
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
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

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
def test_pytorch_multiple_lr(lr1, lr2, expected_lr, nb_warnings, caplog):
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
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    my_algo = MyAlgo()

    caplog.clear()
    my_algo._update_current_lr()
    warnings = [record for record in caplog.records if (record.levelname == "WARNING")]
    assert my_algo._current_lr == expected_lr
    assert len(warnings) == nb_warnings


@pytest.mark.parametrize("nb_update_params_call, num_updates", ([0, 1], [1, 2], [3, 5]))
def test_update_parameters_call(nb_update_params_call, torch_linear_model, num_updates):
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
            )

        def _local_train(self, x: Any, y: Any):
            for _ in self._index_generator:
                continue
            for _ in range(nb_update_params_call):
                self._scaffold_parameters_update()

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())

            return y_pred.detach().numpy()

    my_algo = MyAlgo()

    with pytest.raises(TorchScaffoldAlgoParametersUpdateError):
        my_algo.train(x=np.random.random(10), y=np.random.random(10), _skip=True)

    assert my_algo._scaffold_parameters_update_num_call == nb_update_params_call
