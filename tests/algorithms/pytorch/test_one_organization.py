import logging

import pytest
import torch

from substrafl import execute_experiment
from substrafl.algorithms.pytorch import TorchSingleOrganizationAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.index_generator import NpIndexGenerator
from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo
from substrafl.nodes.node import OutputIdentifiers
from substrafl.strategies import SingleOrganization
from tests import utils

logger = logging.getLogger(__name__)

EXPECTED_PERFORMANCE = 0.2774176577698596


@pytest.mark.parametrize("n_updates, n_rounds", [(1, 2), (2, 1)])  # slow test so checking only two possibilities
@pytest.mark.substra
@pytest.mark.slow
def test_one_organization(
    network,
    torch_linear_model,
    train_linear_nodes,
    test_linear_nodes,
    session_dir,
    n_updates,
    n_rounds,
    test_linear_data_samples,
    mae,
    numpy_torch_dataset,
):
    """End to end test for torch one organization algorithm. Checking that the perf are the same for :
    different combinations of n_updates and n_rounds
     The expected result was calculated to be the same for the local mode and for the in pure Substra. For the
     details of the implementation of the latter ones please go to PR #109
    """
    # Common definition
    seed = 42
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )
    BATCH_SIZE = 32

    strategy = SingleOrganization()

    torch.manual_seed(seed)
    perceptron = torch_linear_model()
    optimizer = torch.optim.SGD(perceptron.parameters(), lr=0.1)
    nig = NpIndexGenerator(
        batch_size=BATCH_SIZE,
        num_updates=n_updates,
    )

    class MyOneOrganizationAlgo(TorchSingleOrganizationAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                model=perceptron,
                index_generator=nig,
                dataset=numpy_torch_dataset,
            )

    my_algo = MyOneOrganizationAlgo()
    my_eval_strategy = EvaluationStrategy(test_data_nodes=test_linear_nodes[:1], rounds=1)  # test every round

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_linear_nodes[:1],
        evaluation_strategy=my_eval_strategy,
        num_rounds=n_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    tasks = network.clients[0].list_task(filters={"compute_plan_key": [compute_plan.key]})
    testtuples = [t for t in tasks if t.outputs.get(OutputIdentifiers.performance) is not None]
    testtuples = sorted(testtuples, key=lambda x: x.rank)

    # ensure that final result is correct up to 6 decimal points
    assert testtuples[-1].outputs[OutputIdentifiers.performance].value == pytest.approx(EXPECTED_PERFORMANCE, rel=10e-6)

    assert local_model_perf(network, compute_plan, session_dir, test_linear_data_samples, mae) == pytest.approx(
        EXPECTED_PERFORMANCE
    )


def local_model_perf(network, compute_plan, session_dir, test_linear_data_samples, mae):
    download_algo_files(
        client=network.clients[0], compute_plan_key=compute_plan.key, round_idx=None, dest_folder=session_dir
    )
    model = load_algo(input_folder=session_dir)._model

    y_pred = model(torch.from_numpy(test_linear_data_samples[0][:, :-1]).float()).detach().numpy().reshape(-1)
    y_true = test_linear_data_samples[0][:, -1:].reshape(-1)
    performance = mae.compute(y_pred, y_true)

    return performance
