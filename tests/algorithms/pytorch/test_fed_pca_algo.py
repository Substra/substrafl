import logging

import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl import simulate_experiment
from substrafl.algorithms.pytorch.torch_fed_pca_algo import TorchFedPCAAlgo
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.model_loading import download_algo_state
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.strategies.fed_pca import FedPCA
from substrafl.strategies.schemas import FedPCAAveragedState
from tests import assets_factory
from tests import utils

logger = logging.getLogger(__name__)


LINEAR_N_COL = 3
N_EIGENVALUES = 1
NUM_ROUNDS = 10


@pytest.fixture(scope="module")
def torch_pca_algo(numpy_torch_dataset, seed):
    torch.manual_seed(seed)

    class MyAlgo(TorchFedPCAAlgo):
        def __init__(self, batch_size=1):
            super().__init__(
                in_features=LINEAR_N_COL,
                out_features=N_EIGENVALUES,
                batch_size=batch_size,
                dataset=numpy_torch_dataset,
                seed=seed,
            )

    return MyAlgo


@pytest.fixture(scope="module")
def abs_metric():
    def abs_diff(data_from_opener, predictions):
        y_pred = np.array(predictions)
        y_true = data_from_opener[1]
        return (abs(y_pred) - abs(y_true)).mean()

    return abs_diff


@pytest.fixture(scope="module")
def simulate_compute_plan(
    torch_pca_algo,
    train_linear_nodes_pca,
    test_linear_nodes_pca,
    aggregation_node,
    abs_metric,
    network,
    session_dir,
):
    strategy = FedPCA(
        algo=torch_pca_algo(),
        metric_functions=abs_metric,
    )
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes_pca,
        eval_rounds=[NUM_ROUNDS],
    )

    performances, _, _ = simulate_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes_pca,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        clean_models=False,
        experiment_folder=session_dir / "experiment_folder",
    )

    return performances


@pytest.fixture(scope="module")
def compute_plan(
    torch_pca_algo,
    torch_cpu_dependency,
    train_linear_nodes_pca,
    test_linear_nodes_pca,
    aggregation_node,
    abs_metric,
    network,
    session_dir,
):
    strategy = FedPCA(
        algo=torch_pca_algo(),
        metric_functions=abs_metric,
    )
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes_pca,
        eval_rounds=[NUM_ROUNDS],
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes_pca,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        dependencies=torch_cpu_dependency,
        experiment_folder=session_dir / "experiment_folder",
        clean_models=False,
    )

    # Wait for the compute plan to be finished
    network.clients[0].wait_compute_plan(compute_plan.key)

    return compute_plan


@pytest.fixture(scope="session")
def train_linear_nodes_pca(network, numpy_datasets, train_linear_data_samples_pca, session_dir):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        train_linear_data_samples (List[np.ndarray]): A List of linear linked data.
        session_dir (Path): A temp file created for the pytest session.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every organization.
        contents=train_linear_data_samples_pca,
        dataset_keys=numpy_datasets,
        clients=network.clients,
        tmp_folder=session_dir,
    )

    train_data_nodes = [
        TrainDataNode(
            network.msp_ids[k],
            numpy_datasets[k],
            [linear_samples[k]],
        )
        for k in range(network.n_organizations)
    ]

    return train_data_nodes


@pytest.fixture(scope="session")
def test_linear_nodes_pca(
    network,
    numpy_datasets,
    test_linear_data_samples_pca,
    session_dir,
):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        test_linear_data_samples_pca (List[np.array]): A one element list containing linear linked data.
        session_dir (Path): A temp file created for the pytest session.

    Returns:
        List[TestDataNode]: A one element list containing a substrafl TestDataNode for linear data.
    """

    linear_samples = assets_factory.add_numpy_samples(
        # We set the weights seeds to ensure that all contents are linearly linked with the same weights but
        # the noise is random so the data is not identical on every organization.
        contents=test_linear_data_samples_pca,
        dataset_keys=[numpy_datasets[0]],
        clients=[network.clients[0]],
        tmp_folder=session_dir,
    )

    test_data_nodes = [
        TestDataNode(
            network.msp_ids[0],
            numpy_datasets[0],
            linear_samples,
        )
    ]

    return test_data_nodes


@pytest.fixture(scope="session")
def numpy_pca_eigen_vectors(train_linear_data_samples_pca):
    data = np.concatenate([d[:, :LINEAR_N_COL] for d in train_linear_data_samples_pca])
    cov = np.cov(data.T)
    _, eig = np.linalg.eig(cov)
    pca_eigen_vectors = eig.T[:N_EIGENVALUES]
    return pca_eigen_vectors


@pytest.fixture(scope="session")
def test_linear_data_samples_pca(numpy_pca_eigen_vectors):
    """Generates linear linked data for testing purposes. The train_linear_data_samples data and
    test_linear_data_samples data are linked with the same weights as they fixed per the same seed.

    Returns:
        List[np.ndarray]: A one element list containing linear linked data.
    """
    test_inputs_data = assets_factory.linear_data(
        n_col=LINEAR_N_COL,
        n_samples=64,
        weights_seed=42,
        noise_seed=42,
    )

    projected_data = np.matmul(test_inputs_data, numpy_pca_eigen_vectors.T)
    return [np.concatenate((test_inputs_data, projected_data), axis=1)]


@pytest.fixture(scope="session")
def train_linear_data_samples_pca(network, seed):
    """Generates linear linked data for training purposes. The train_linear_data_samples data and
    test_linear_data_samples data are linked with the same weights as they fixed per the same seed.

    Args:
        network (Network): Substra network from the configuration file.

    Returns:
        List[np.ndarray]: A list of linear data for each organization of the network.
    """
    cov_matrix = np.array([[10.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.1]])
    mean = np.array([100.0, -20.0, 10.0])

    result = []
    np.random.seed(seed)
    for _ in range(network.n_organizations):
        data = np.zeros((100, 4))
        data[:, :3] = np.random.multivariate_normal(mean, cov_matrix, size=100)
        data[:, 3] = np.random.normal(0.0, 1.0, size=100)
        result.append(data)
    return result


@pytest.mark.substra
@pytest.mark.slow
def test_torch_fed_pca_eigen_values(network, compute_plan, session_dir, numpy_pca_eigen_vectors, rtol):
    """Check the weight initialization, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """

    # Test eigen values compute during last aggregation

    # The number of rank is a first local update, and then aggregation and train times num rounds
    final_aggregated_rank = (
        1 + 2 * NUM_ROUNDS - 1
    )  # We want the last aggregated task, one rank before the final train one
    fed_pca_model = utils.download_aggregate_model_by_rank(
        network, session_dir, compute_plan, rank=final_aggregated_rank
    )

    fed_pca_eigen_vectors = fed_pca_model.avg_parameters_update[0]

    # Align eigen values using their collinear coefficient
    assert np.array(
        [
            np.allclose(np.dot(numpy_pca_eigen_vectors[i], row) * row, numpy_pca_eigen_vectors[i], rtol=rtol)
            for i, row in enumerate(fed_pca_eigen_vectors)
        ]
    ).all()


@pytest.mark.substra
@pytest.mark.slow
def test_torch_fed_pca_performance(network, compute_plan, rtol):
    # Test computed predictions (inputs projected with eigen vectors)
    perfs = network.clients[0].get_performances(compute_plan.key)
    assert pytest.approx(0, abs=rtol) == perfs.performance[0]


@pytest.mark.substra
@pytest.mark.slow
def test_compare_execute_and_simulate_fed_pca_performances(network, compute_plan, simulate_compute_plan, rtol):
    perfs = network.clients[0].get_performances(compute_plan.key)

    simu_perfs = simulate_compute_plan
    assert np.allclose(perfs.performance, simu_perfs.performance, rtol=rtol)


@pytest.mark.slow
@pytest.mark.substra
def test_download_load_algo(network, compute_plan, test_linear_data_samples_pca, numpy_pca_eigen_vectors, rtol):
    my_algo = download_algo_state(
        client=network.clients[0],
        compute_plan_key=compute_plan.key,
    )

    # Align eigen values using their collinear coefficient
    assert np.array(
        [
            np.allclose(np.dot(numpy_pca_eigen_vectors[i], row) * row, numpy_pca_eigen_vectors[i], rtol=rtol)
            for i, row in enumerate(my_algo.eigen_vectors)
        ]
    ).all()

    y_pred = my_algo.transform(torch.from_numpy(test_linear_data_samples_pca[0][:, :-1]).float()).detach().numpy()
    y_true = test_linear_data_samples_pca[0][:, -1]
    performance = (abs(y_pred) - abs(y_true)).mean()
    assert pytest.approx(0, abs=rtol) == performance


@pytest.mark.parametrize(
    "data, local_mean, local_covmat",
    [
        [
            (np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), np.array([[0], [0], [0]])),
            np.array([2, 2, 2]),
            np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        ],
        [
            (np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]]), np.array([[0], [0], [0]])),
            np.array([2, 2, 2]),
            np.array([[2, 0, -2], [0, 0, 0], [-2, 0, 2]]),
        ],
    ],
)
def test_train_pca_algo(torch_pca_algo, data, local_mean, local_covmat, rtol):
    """Data index 0 are the input data, and index 1 are unused labels."""
    my_algo = torch_pca_algo()
    assert my_algo.local_mean is None

    out = my_algo.train(data_from_opener=data, _skip=True)
    assert np.allclose(out.parameters_update[0], local_mean, rtol=rtol)
    assert np.allclose(my_algo.local_mean, local_mean, rtol=rtol)
    assert my_algo.local_covmat is None

    avg_mean = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(data_from_opener=data, shared_state=avg_mean, _skip=True)
    assert np.allclose(my_algo.local_covmat, local_covmat, rtol=rtol)


@pytest.mark.parametrize(
    "data",
    [
        (np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), np.array([[0], [0], [0]])),
        (np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]]), np.array([[0], [0], [0]])),
    ],
)
def test_predict_pca_algo(torch_pca_algo, data, rtol):
    """Data index 0 are the input data, and index 1 are unused labels."""
    my_algo = torch_pca_algo()

    predictions_round0 = my_algo.predict(data_from_opener=data)

    out = my_algo.train(data_from_opener=data, _skip=True)
    predictions_round1 = my_algo.predict(data_from_opener=data)
    assert np.allclose(predictions_round0, predictions_round1, rtol=rtol)  # Model is not updated before round 2

    avg_mean = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(data_from_opener=data, shared_state=avg_mean, _skip=True)
    predictions_round2 = my_algo.predict(data_from_opener=data)
    assert np.allclose(predictions_round1, predictions_round2, rtol=rtol)  # Model is not updated before round 2

    avg_parameters = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(data_from_opener=data, shared_state=avg_parameters, _skip=True)
    predictions_round3 = my_algo.predict(data_from_opener=data)
    assert not np.allclose(predictions_round2, predictions_round3, rtol=rtol)  # Model is updated at round 3


@pytest.mark.parametrize("batch_size", (1, 1_000_000_000_000_000_000))
def test_large_batch_size_in_predict(batch_size, torch_pca_algo, mocker):
    n_samples = 10

    x_train = np.zeros([n_samples, 3])
    y_train = np.ones([n_samples, 1])

    my_algo = torch_pca_algo(batch_size=batch_size)

    spy = mocker.spy(torch.utils.data, "DataLoader")

    # Check that no MemoryError is thrown
    my_algo.predict(data_from_opener=(x_train, y_train))

    assert spy.call_count == 1
    assert spy.spy_return.batch_size == min(batch_size, n_samples)
