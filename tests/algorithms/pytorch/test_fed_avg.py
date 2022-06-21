import logging
from typing import Any

import pytest
import torch

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.algorithms.pytorch.weight_manager import increment_parameters
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.index_generator import NpIndexGenerator
from connectlib.strategies import FedAvg
from tests import utils
from tests.algorithms.pytorch.torch_tests_utils import assert_model_parameters_equal

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def torch_algo(torch_linear_model):
    num_updates = 100
    seed = 42
    torch.manual_seed(seed)
    perceptron = torch_linear_model()
    nig = NpIndexGenerator(
        batch_size=32,
        num_updates=num_updates,
    )

    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                model=perceptron,
                index_generator=nig,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    return MyAlgo


@pytest.fixture(scope="module")
def compute_plan(torch_algo, train_linear_nodes, test_linear_nodes, aggregation_node, network, session_dir):

    num_rounds = 3

    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )

    strategy = FedAvg()
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
def test_pytorch_fedavg_algo_weights(network, compute_plan, torch_algo, session_dir):
    """Check the weight initialization, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """

    my_algo = torch_algo()

    rank_0_local_models = utils.download_composite_models_by_rank(network, session_dir, my_algo, compute_plan, rank=0)
    rank_2_local_models = utils.download_composite_models_by_rank(network, session_dir, my_algo, compute_plan, rank=2)

    # Download the aggregate output
    aggregate_model = utils.download_aggregate_model_by_rank(network, session_dir, compute_plan, rank=1)
    aggregate_update = [torch.from_numpy(x).to("cpu") for x in aggregate_model.avg_parameters_update]

    # Assert the model initialization is the same for every model
    assert_model_parameters_equal(rank_0_local_models[0].model, rank_0_local_models[1].model)

    # Assert that the weights are well set
    for model_0, model_2 in zip(rank_0_local_models, rank_2_local_models):
        increment_parameters(model_0.model, aggregate_update, with_batch_norm_parameters=True)
        assert_model_parameters_equal(model_0.model, model_2.model)

    # The local models are always the same on every organization
    assert_model_parameters_equal(rank_2_local_models[0].model, rank_2_local_models[1].model)


@pytest.mark.e2e
@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_fedavg_algo_performance(
    network,
    compute_plan,
    rtol,
):
    """End to end test for torch fed avg algorithm."""

    expected_performance = 0.0127768361

    testtuples = network.clients[0].list_testtuple(filters={"compute_plan_key": [compute_plan.key]})
    testtuple = testtuples[0]

    assert list(testtuple.test.perfs.values())[0] == pytest.approx(expected_performance, rel=rtol)
