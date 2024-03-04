import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl import simulate_experiment
from substrafl.algorithms.pytorch import TorchNewtonRaphsonAlgo
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import CriterionReductionError
from substrafl.exceptions import NegativeHessianMatrixError
from substrafl.model_loading import download_algo_state
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.strategies import NewtonRaphson

from ... import assets_factory

# For a convex problem and with damping_factor = 1, Newton Raphson is supposed to reach the global optimum
# in one round.
EXPECTED_PERFORMANCE = 0
NUM_ROUNDS = 1


@pytest.fixture(scope="module")
def perceptron():
    class Perceptron(torch.nn.Module):
        def __init__(
            self,
            linear_n_col: int = 1,
            linear_n_target: int = 1,
            constant_weight_init: float = None,
            constant_bias_init: float = None,
        ):
            super().__init__()
            self.linear1 = torch.nn.Linear(linear_n_col, linear_n_target)
            if constant_weight_init is not None:
                torch.nn.init.constant_(self.linear1.weight, constant_weight_init)
            if constant_bias_init is not None:
                torch.nn.init.constant_(self.linear1.bias, constant_bias_init)

        def forward(self, x):
            out = self.linear1(x)
            return out

    return Perceptron


@pytest.fixture(scope="module")
def torch_algo(perceptron, numpy_torch_dataset):
    class MyAlgo(TorchNewtonRaphsonAlgo):
        def __init__(self, model=None, criterion=None, batch_size=1, l2_coeff=0):
            super().__init__(
                model=model or perceptron(),
                criterion=criterion or torch.nn.MSELoss(),
                batch_size=batch_size,
                dataset=numpy_torch_dataset,
                l2_coeff=l2_coeff,
            )

    return MyAlgo


@pytest.mark.parametrize(
    "n_samples,batch_size,expected_n_updates",
    [(10, 10, 1), (1, 10, 1), (10, 1, 10), (10, None, 1)],
)
def test_index_generator(torch_algo, n_samples, batch_size, expected_n_updates):
    """Test that num update is well updated in the index generator and that
    num_updates = math.ceil(n_samples / batch_size).
    Also test the behavior for batch_size set to None (batch_size set to num_samples by the index generator and
    num_update set to 1 for Newton Raphson."""

    data_from_opener = (np.zeros([n_samples, 1]), np.zeros([n_samples, 1]))

    my_algo = torch_algo(batch_size=batch_size)

    my_algo.train(data_from_opener=data_from_opener, _skip=True)

    assert my_algo._index_generator.batch_size == batch_size or n_samples
    assert my_algo._index_generator.n_samples == n_samples
    assert my_algo._index_generator.num_updates == expected_n_updates
    assert not my_algo._index_generator._drop_last


@pytest.mark.parametrize(
    "data_from_opener,expected_gradient,expected_hessian",
    [
        (([2], [1]), [4, 2], [[8, 4], [4, 2]]),  # One input point
        (([1, 2], [2, 4]), [-5, -3], [[5, 3], [3, 2]]),  # Two input points
        ([[-1, 0, 1], [-10, 4, -8]], [0, 28 / 3], [[4 / 3, 0], [0, 2]]),  # Three input points
    ],
)
def test_train_newton_raphson_shared_states_results(
    torch_algo, perceptron, data_from_opener, expected_hessian, expected_gradient
):
    """Test the theoretical value of the gradients and Hessian for a MSE loss and f(x) = a * x + b with
    a = 1 and b = 0."""
    data_from_opener = np.array(data_from_opener)[..., None]

    model = perceptron(constant_weight_init=1, constant_bias_init=0)
    my_algo = torch_algo(model)

    shared_states = my_algo.train(data_from_opener=data_from_opener, _skip=True)

    # ensure that final result is correct up to 6 decimal points
    rel = 1e-6

    assert pytest.approx(np.array(expected_gradient), rel) == [float(p.squeeze()) for p in shared_states.gradients]
    assert pytest.approx(np.array(expected_hessian), rel) == shared_states.hessian


@pytest.mark.parametrize(
    "l2_coeff",
    [(0), (0.1)],
)
def test_l2_coeff(torch_algo, l2_coeff):
    """Test that the eigenvalues of the hessian is increase by the l2_coeff value."""

    n_samples = 2

    data_from_opener = (np.zeros([n_samples, 1]), np.ones([n_samples, 1]))

    my_algo_l2 = torch_algo(l2_coeff=l2_coeff)
    my_algo_no_l2 = torch_algo(l2_coeff=0)

    shared_states_l2 = my_algo_l2.train(data_from_opener=data_from_opener, _skip=True)
    shared_states = my_algo_no_l2.train(data_from_opener=data_from_opener, _skip=True)

    assert np.allclose(
        np.linalg.eig(shared_states_l2.hessian)[0].real, np.linalg.eig(shared_states.hessian)[0].real + l2_coeff
    )


@pytest.mark.parametrize(
    "reduction,raise_error",
    [("mean", False), ("sum", True)],
)
def test_wrong_reduction_criterion_value(torch_algo, reduction, raise_error):
    criterion = torch.nn.MSELoss(reduction=reduction)

    if raise_error:
        with pytest.raises(CriterionReductionError):
            torch_algo(criterion=criterion)
    else:
        torch_algo(criterion=criterion)


@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        (5, 3),  # x_shape > y_shape
        (3, 5),  # x_shape > y_shape
        (10, 10),  # x_shape == y_shape
    ],
)
def test_train_newton_raphson_shared_states_shape(torch_algo, perceptron, x_shape, y_shape):
    """Test the shape of the gradients and the Hessian matrix are coherent with the linear model shape."""
    n_samples = 10

    x_train = np.zeros([n_samples, x_shape])
    y_train = np.ones([n_samples, y_shape])

    model = perceptron(linear_n_col=x_shape, linear_n_target=y_shape)
    my_algo = torch_algo(model=model, batch_size=10)

    shared_states = my_algo.train(data_from_opener=(x_train, y_train), _skip=True)

    assert (
        sum([g.size for g in shared_states.gradients]) == (x_shape + 1) * y_shape
    )  # Number of weights + number of bias
    assert shared_states.hessian.size == ((x_shape + 1) * y_shape) ** 2


def test_train_newton_raphson_non_convex_cnn(torch_algo, seed):
    """Test that NegativeHessianMatrixError is raised when the Hessian matrix is non positive semi definite for a
    non-convex problem."""

    np.random.seed(seed)

    x_train = np.random.randn(1, 3, 9, 9)
    y_train = np.random.randn(1, 1)

    class CNN(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.cnn = torch.nn.Conv2d(3, 1, (3, 3), stride=(3, 3))
            self.relu = torch.nn.ReLU()
            self.linear = torch.nn.Linear(9, 1)

        def forward(self, x):
            out = self.relu(self.cnn(x))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    model = CNN()

    criterion = torch.nn.BCEWithLogitsLoss()

    my_algo = torch_algo(model=model, criterion=criterion)

    with pytest.raises(NegativeHessianMatrixError):
        my_algo.train(data_from_opener=(x_train, y_train), _skip=True)


@pytest.mark.parametrize(
    "n_samples,y_shape,batch_size",
    [(5, 1, 2), (5, 1, None), (5, 2, 2), (5, 2, None)],
)
def test_prediction_shape(perceptron, torch_algo, n_samples, y_shape, batch_size):
    x_shape = 15

    x_train = np.zeros([n_samples, x_shape])
    y_train = np.ones([n_samples, y_shape])

    model = perceptron(linear_n_col=x_shape, linear_n_target=y_shape)
    my_algo = torch_algo(model=model, batch_size=batch_size)

    prediction = my_algo.predict(data_from_opener=(x_train, y_train))
    assert prediction.shape == (n_samples, y_shape)


@pytest.fixture(scope="module")
def nr_test_data():
    return [np.array([[5, 5, 11], [6, 6, 13], [7, 7, 15]])]


@pytest.fixture(scope="module")
def simulate_compute_plan(
    network,
    nr_test_data,
    numpy_datasets,
    perceptron,
    mae_metric,
    aggregation_node,
    session_dir,
    numpy_torch_dataset,
    seed,
):
    """Compute plan for e2e test"""

    DAMPING_FACTOR = 1
    BATCH_SIZE = 1

    torch.manual_seed(seed)

    # We define several sample without noises to be sure to reach the global optimum.
    # Here, f(x1,x2) = x1 + x2 + 1
    train_sample_nodes = assets_factory.add_numpy_samples(
        contents=np.array([[[0, 0, 1], [1, 2, 4], [2, 3, 6]], [[0, 0, 1], [1, 3, 5], [4, 2, 7]]]),
        dataset_keys=numpy_datasets,
        clients=network.clients,
        tmp_folder=session_dir,
    )

    train_data_nodes = [
        TrainDataNode(
            network.msp_ids[k],
            numpy_datasets[k],
            [train_sample_nodes[k]],
        )
        for k in range(network.n_organizations)
    ]

    test_sample_nodes = assets_factory.add_numpy_samples(
        contents=nr_test_data,
        dataset_keys=[numpy_datasets[0]],
        clients=[network.clients[0]],
        tmp_folder=session_dir,
    )

    test_data_nodes = [
        TestDataNode(
            network.msp_ids[0],
            numpy_datasets[0],
            [test_sample_nodes[0]],
        )
    ]

    model = perceptron(linear_n_col=2, linear_n_target=1)
    criterion = torch.nn.MSELoss()

    class MyAlgo(TorchNewtonRaphsonAlgo):
        def __init__(self):
            super().__init__(
                model=model,
                criterion=criterion,
                batch_size=BATCH_SIZE,
                dataset=numpy_torch_dataset,
                l2_coeff=0,
                use_gpu=False,
            )

    my_algo = MyAlgo()

    strategy = NewtonRaphson(algo=my_algo, metric_functions=mae_metric, damping_factor=DAMPING_FACTOR)
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_data_nodes, eval_rounds=[0, NUM_ROUNDS]
    )  # test the initialization and the last round

    performances, _, _ = simulate_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        experiment_folder=session_dir / "experiment_folder",
    )

    return performances


@pytest.fixture(scope="module")
def compute_plan(
    network,
    nr_test_data,
    numpy_datasets,
    perceptron,
    mae_metric,
    torch_cpu_dependency,
    aggregation_node,
    session_dir,
    numpy_torch_dataset,
    seed,
):
    """Compute plan for e2e test"""

    DAMPING_FACTOR = 1
    BATCH_SIZE = 1

    torch.manual_seed(seed)

    # We define several sample without noises to be sure to reach the global optimum.
    # Here, f(x1,x2) = x1 + x2 + 1
    train_sample_nodes = assets_factory.add_numpy_samples(
        contents=np.array([[[0, 0, 1], [1, 2, 4], [2, 3, 6]], [[0, 0, 1], [1, 3, 5], [4, 2, 7]]]),
        dataset_keys=numpy_datasets,
        clients=network.clients,
        tmp_folder=session_dir,
    )

    train_data_nodes = [
        TrainDataNode(
            network.msp_ids[k],
            numpy_datasets[k],
            [train_sample_nodes[k]],
        )
        for k in range(network.n_organizations)
    ]

    test_sample_nodes = assets_factory.add_numpy_samples(
        contents=nr_test_data,
        dataset_keys=[numpy_datasets[0]],
        clients=[network.clients[0]],
        tmp_folder=session_dir,
    )

    test_data_nodes = [
        TestDataNode(
            network.msp_ids[0],
            numpy_datasets[0],
            [test_sample_nodes[0]],
        )
    ]

    model = perceptron(linear_n_col=2, linear_n_target=1)
    criterion = torch.nn.MSELoss()

    class MyAlgo(TorchNewtonRaphsonAlgo):
        def __init__(self):
            super().__init__(
                model=model,
                criterion=criterion,
                batch_size=BATCH_SIZE,
                dataset=numpy_torch_dataset,
                l2_coeff=0,
                use_gpu=False,
            )

    my_algo = MyAlgo()

    strategy = NewtonRaphson(algo=my_algo, metric_functions=mae_metric, damping_factor=DAMPING_FACTOR)
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_data_nodes, eval_rounds=[0, NUM_ROUNDS]
    )  # test the initialization and the last round

    compute_plan = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        dependencies=torch_cpu_dependency,
        experiment_folder=session_dir / "experiment_folder",
    )

    # Wait for the compute plan to be finished
    network.clients[0].wait_compute_plan(compute_plan.key)

    return compute_plan


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_nr_algo_performance(
    network,
    compute_plan,
    nr_test_data,
    perceptron,
    mae,
    rtol,
    seed,
):
    perfs = network.clients[0].get_performances(compute_plan.key)

    assert pytest.approx(EXPECTED_PERFORMANCE, abs=rtol) == perfs.performance[1]

    torch.manual_seed(seed)

    model = perceptron(linear_n_col=2, linear_n_target=1)
    y_pred = model(torch.from_numpy(nr_test_data[0][:, :-1]).float()).detach().numpy().reshape(-1)
    y_true = nr_test_data[0][:, -1]

    performance_at_init = mae(y_pred, y_true)

    # This abs_ative error is due to the l2 regularization, mandatory to reach numerical stability.
    # This fails on mac M1 pro with 1e-5
    # TODO investigate
    assert performance_at_init == pytest.approx(perfs.performance[0], abs=rtol)


@pytest.mark.substra
@pytest.mark.slow
def test_compare_execute_and_simulate_nr_performances(network, compute_plan, simulate_compute_plan, rtol):
    perfs = network.clients[0].get_performances(compute_plan.key)

    simu_perfs = simulate_compute_plan
    assert np.allclose(perfs.performance, simu_perfs.performance, rtol=rtol)


@pytest.mark.slow
@pytest.mark.substra
def test_download_load_algo(network, compute_plan, nr_test_data, mae, rtol):
    algo = download_algo_state(
        client=network.clients[0],
        compute_plan_key=compute_plan.key,
    )
    model = algo.model

    y_pred = model(torch.from_numpy(nr_test_data[0][:, :-1]).float()).detach().numpy().reshape(-1)
    y_true = nr_test_data[0][:, -1:].reshape(-1)
    performance = mae(y_pred, y_true)

    # This test fails with default approx parameters
    # TODO: investigate
    assert performance == pytest.approx(EXPECTED_PERFORMANCE, abs=rtol)


@pytest.mark.parametrize("batch_size", (1, 1_000_000_000_000_000_000))
def test_large_batch_size_in_predict(batch_size, torch_algo, perceptron, mocker):
    n_samples = 10

    x_train = np.zeros([n_samples, 10])
    y_train = np.ones([n_samples, 10])

    model = perceptron(linear_n_col=10, linear_n_target=10)
    my_algo = torch_algo(model=model, batch_size=batch_size)

    spy = mocker.spy(torch.utils.data, "DataLoader")

    # Check that no MemoryError is thrown
    my_algo.predict(data_from_opener=(x_train, y_train))

    assert spy.call_count == 1
    assert spy.spy_return.batch_size == min(batch_size, n_samples)
