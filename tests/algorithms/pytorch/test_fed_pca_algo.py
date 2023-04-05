import logging

import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl.algorithms.pytorch.torch_fed_pca_algo import TorchFedPCAAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.remote import register
from substrafl.schemas import FedPCAAveragedState
from substrafl.strategies.fed_pca import FedPCA
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
        def __init__(
            self,
        ):
            super().__init__(
                in_features=LINEAR_N_COL,
                out_features=N_EIGENVALUES,
                batch_size=1,
                dataset=numpy_torch_dataset,
                seed=seed,
            )

    return MyAlgo


@pytest.fixture(scope="module")
def compute_plan(
    torch_pca_algo,
    train_linear_nodes_pca,
    test_linear_nodes_pca,
    aggregation_node,
    network,
    session_dir,
):
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )

    strategy = FedPCA(algo=torch_pca_algo())
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
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
        clean_models=False,
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

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
    abs_diff_metric,
    test_linear_data_samples_pca,
    session_dir,
):
    """Linear linked data samples.

    Args:
        network (Network): Substra network from the configuration file.
        numpy_datasets (List[str]): Keys linked to numpy dataset (opener) on each organization.
        mae (str): Mean absolute error metric for the TestDataNode
        session_dir (Path): A temp file created for the pytest session.

    Returns:
        List[TestDataNode]: A one element list containing a substrafl TestDataNode for linear data with
          a mae metric.
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
            metric_keys=[abs_diff_metric],
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
def abs_diff_metric(network, default_permissions, mae):
    metric_deps = Dependency(pypi_dependencies=["numpy"], editable_mode=True)

    def abs_diff(datasamples, predictions_path):
        y_true = datasamples[1]
        y_pred = np.load(predictions_path)
        return (abs(y_pred) - abs(y_true)).mean()

    metric_key = register.add_metric(
        client=network.clients[0], metric_function=abs_diff, permissions=default_permissions, dependencies=metric_deps
    )

    return metric_key


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


@pytest.mark.slow
@pytest.mark.substra
def test_download_load_algo(
    network, compute_plan, session_dir, test_linear_data_samples_pca, numpy_pca_eigen_vectors, rtol
):
    download_algo_files(
        client=network.clients[0],
        compute_plan_key=compute_plan.key,
        round_idx=None,
        dest_folder=session_dir,
    )
    my_algo = load_algo(input_folder=session_dir)

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

    out = my_algo.train(datasamples=data, _skip=True)
    assert np.allclose(out.parameters_update[0], local_mean, rtol=rtol)
    assert np.allclose(my_algo.local_mean, local_mean, rtol=rtol)
    assert my_algo.local_covmat is None

    avg_mean = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(datasamples=data, shared_state=avg_mean, _skip=True)
    assert np.allclose(my_algo.local_covmat, local_covmat, rtol=rtol)


@pytest.mark.parametrize(
    "data",
    [
        (np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), np.array([[0], [0], [0]])),
        (np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]]), np.array([[0], [0], [0]])),
    ],
)
def test_predict_pca_algo(torch_pca_algo, session_dir, data, rtol):
    """Data index 0 are the input data, and index 1 are unused labels."""
    my_algo = torch_pca_algo()

    prediction_file = session_dir / "PCA_predictions"

    my_algo.predict(datasamples=data, predictions_path=prediction_file, _skip=True)
    predictions_round0 = np.load(prediction_file)

    out = my_algo.train(datasamples=data, _skip=True)
    my_algo.predict(datasamples=data, predictions_path=prediction_file, _skip=True)
    predictions_round1 = np.load(prediction_file)
    assert np.allclose(predictions_round0, predictions_round1, rtol=rtol)  # Model is not updated before round 2

    avg_mean = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(datasamples=data, shared_state=avg_mean, _skip=True)
    my_algo.predict(datasamples=data, predictions_path=prediction_file, _skip=True)
    predictions_round2 = np.load(prediction_file)
    assert np.allclose(predictions_round1, predictions_round2, rtol=rtol)  # Model is not updated before round 2

    avg_parameters = FedPCAAveragedState(avg_parameters_update=out.parameters_update)
    out = my_algo.train(datasamples=data, shared_state=avg_parameters, _skip=True)
    my_algo.predict(datasamples=data, predictions_path=prediction_file, _skip=True)
    predictions_round3 = np.load(prediction_file)
    assert not np.allclose(predictions_round2, predictions_round3, rtol=rtol)  # Model is updated at round 3
