import logging

import numpy as np
import pytest
import torch

from substrafl import execute_experiment
from substrafl.algorithms.pytorch.torch_fed_pca_algo import TorchFedPCAAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.strategies.fed_pca import FedPCA
from tests import assets_factory
from tests import utils
from tests.utils import download_last_aggregate_model

logger = logging.getLogger(__name__)


LINEAR_N_COL = 3
LINEAR_N_TARGET = 1
N_EIGENVALUES = 2
NUM_ROUNDS = 7
BATCH_SIZE = 1


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
                batch_size=BATCH_SIZE,
                dataset=numpy_torch_dataset,
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

    strategy = FedPCA()
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes_pca,
        eval_rounds=[NUM_ROUNDS],
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=torch_pca_algo(),
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
    mae_metric,
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
            metric_keys=[mae_metric],
        )
    ]

    return test_data_nodes


@pytest.fixture(scope="session")
def test_linear_data_samples_pca():
    """Generates linear linked data for testing purposes. The train_linear_data_samples data and
    test_linear_data_samples data are linked with the same weights as they fixed per the same seed.

    Returns:
        List[np.ndarray]: A one element list containing linear linked data.
    """
    return [
        assets_factory.linear_data(
            n_col=LINEAR_N_COL + LINEAR_N_TARGET,
            n_samples=64,
            weights_seed=42,
            noise_seed=42,
        )
    ]


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
def test_cp_performance(network, compute_plan, session_dir, train_linear_data_samples_pca):
    """Check the weight initialization, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """

    data = np.concatenate([d[:, :LINEAR_N_COL] for d in train_linear_data_samples_pca])
    cov = np.cov(data.T)
    _, eig = np.linalg.eig(cov)
    numpy_pca_eigen_values = eig.T[:2]
    fed_pca_model = download_last_aggregate_model(network, session_dir, compute_plan)
    fed_pca_eigen_values = fed_pca_model.avg_parameters_update[0]
    numpy_pca_eigen_values = np.array([np.sign(eigen_v[0]) * eigen_v for eigen_v in numpy_pca_eigen_values])
    fed_pca_eigen_values = np.array([np.sign(eigen_v[0]) * eigen_v for eigen_v in fed_pca_eigen_values])
    np.testing.assert_allclose(numpy_pca_eigen_values, fed_pca_eigen_values, rtol=1e-5)
