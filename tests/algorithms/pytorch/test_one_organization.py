import logging
from typing import Any

import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchSingleOrganizationAlgo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.index_generator import NpIndexGenerator
from connectlib.strategies import SingleOrganization
from tests import utils

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("n_updates, n_rounds", [(1, 2), (2, 1)])  # slow test so checking only two possibilities
@pytest.mark.substra
@pytest.mark.slow
def test_one_organization(
    network, torch_linear_model, train_linear_nodes, test_linear_nodes, session_dir, n_updates, n_rounds
):
    """End to end test for torch one organization algorithm. Checking that the perf are the same for :
    different combinations of n_updates and n_rounds
     The expected result was calculated to be the same for the local mode and for the in pure Substra. For the
     details of the implementation of the latter ones please go to PR #109
    """
    # Common definition
    expected_performance = 0.2774176577698596
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
            )

        def _local_train(self, x: Any, y: Any):
            return super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

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

    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuples = sorted(testtuples, key=lambda x: x.rank)

    # ensure that final result is correct up to 6 decimal points
    assert list(testtuples[-1].test.perfs.values())[0] == pytest.approx(expected_performance, rel=10e-6)
