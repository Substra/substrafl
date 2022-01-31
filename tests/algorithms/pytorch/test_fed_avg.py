import logging
from typing import Any

import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.dependency import Dependency
from connectlib.strategies import FedAVG
from tests import utils

logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.substra
def test_pytorch_fedavg_algo(
    network,
    torch_linear_model,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
):
    # End to end test for torch fed avg algorithm
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
                num_updates=100,
            )

        def _preprocess(self, x: Any, y: Any = None) -> torch.Tensor:
            # convert numpy array to tensor.
            if y is not None:
                return (
                    torch.from_numpy(x).float(),
                    torch.from_numpy(y).float(),
                )
            else:
                return torch.from_numpy(x).float()

        def _postprocess(self, y_pred: torch.Tensor):
            # convert tensor to numpy array for evaluation.
            y_pred = y_pred.detach().numpy()
            return y_pred

    num_rounds = 3

    my_algo = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["torch", "numpy"])
    strategy = FedAVG()

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        test_data_nodes=test_linear_nodes,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    # read the results from saved performances
    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuple = testtuples[0]

    assert list(testtuple.test.perfs.values())[0] == pytest.approx(0.012787394571974166, rel=10e-6)
