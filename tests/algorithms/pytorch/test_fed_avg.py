import logging
from typing import Any

import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.strategies import FedAVG
from tests import utils

logger = logging.getLogger(__name__)


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_fedavg_algo(
    network,
    torch_linear_model,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
):
    """End to end test for torch fed avg algorithm."""
    num_updates = 100
    num_rounds = 3
    expected_performance = 0.0127768361

    seed = 42
    torch.manual_seed(seed)
    perceptron = torch_linear_model()
    optimizer = torch.optim.SGD(perceptron.parameters(), lr=0.1)

    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=perceptron,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                num_updates=num_updates,
                batch_size=32,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    my_algo = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["torch", "numpy"], editable_mode=True)
    strategy = FedAVG()
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_linear_nodes, rounds=[num_rounds]  # test only at the last round
    )

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    # read the results from saved performances
    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuple = testtuples[0]
    assert list(testtuple.test.perfs.values())[0] == pytest.approx(expected_performance, rel=10e-6)
