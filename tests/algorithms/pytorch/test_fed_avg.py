import logging
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from substra.sdk.models import ModelType

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.algorithms.pytorch.weight_manager import get_parameters
from connectlib.algorithms.pytorch.weight_manager import increment_parameters
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.strategies import FedAVG
from tests import utils

logger = logging.getLogger(__name__)
current_folder = Path(__file__).parent


def _assert_model_parameters_equal(model1, model2):
    model1_params = get_parameters(model1, with_batch_norm_parameters=True)
    model2_params = get_parameters(model2, with_batch_norm_parameters=True)
    assert len(model1_params) == len(model2_params)

    for params1, params2 in zip(model1_params, model2_params):
        assert torch.equal(params1, params2)


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_fedavg_algo_weights(
    network,
    torch_linear_model,
    train_linear_nodes,
    aggregation_node,
    session_dir,
):
    """Check the weight initialisation, aggregation and set weights.
    The aggregation itself is tested at the strategy level, here we test
    the pytorch layer.
    """
    num_updates = 2
    num_rounds = 2
    batch_size = 1

    seed = 42
    torch.manual_seed(seed)
    perceptron = torch_linear_model()

    class MyAlgo(TorchFedAvgAlgo):
        def __init__(self):
            super().__init__(
                model=perceptron,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                num_updates=num_updates,
                batch_size=batch_size,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    my_algo = MyAlgo()
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )
    strategy = FedAVG()
    my_eval_strategy = None

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

    def _download_composite_models_by_rank(rank: int):
        # Retrieve composite train tuple key
        train_tasks = network.clients[0].list_composite_traintuple(
            filters=[f"compositetraintuple:compute_plan_key:{compute_plan.key}", f"compositetraintuple:rank:{rank}"]
        )

        local_models = list()
        for task in train_tasks:
            for model in task.composite.models:
                network.clients[0].download_model(model.key, session_dir)
                model_path = session_dir / f"model_{model.key}"
                if model.category == ModelType.head:
                    local_models.append(my_algo.load(model_path).model)
        return local_models

    rank_0_local_models = _download_composite_models_by_rank(rank=0)
    rank_2_local_models = _download_composite_models_by_rank(rank=2)

    # Download the aggregate output
    aggregate_task = network.clients[0].list_aggregatetuple(
        filters=[f"aggregatetuple:compute_plan_key:{compute_plan.key}", f"aggregatetuple:rank:{1}"]
    )[0]
    model_key = aggregate_task.aggregate.models[0].key
    network.clients[0].download_model(model_key, session_dir)
    model_path = session_dir / f"model_{model_key}"
    aggregate_model = pickle.loads(model_path.read_bytes())

    # Assert the model initialisation is the same for every model
    _assert_model_parameters_equal(rank_0_local_models[0], rank_0_local_models[1])

    # Assert that the weights are well set
    for model_0, model_2 in zip(rank_0_local_models, rank_2_local_models):
        increment_parameters(model_0, aggregate_model.values(), with_batch_norm_parameters=True)
        _assert_model_parameters_equal(model_0, model_2)

    # The local models are always the same on every node
    _assert_model_parameters_equal(rank_2_local_models[0], rank_2_local_models[1])


@pytest.mark.substra
@pytest.mark.slow
def test_pytorch_fedavg_algo_performance(
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

    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                optimizer=torch.optim.SGD(perceptron.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                model=perceptron,
                num_updates=num_updates,
                batch_size=32,
            )

        def _local_train(self, x: Any, y: Any):
            super()._local_train(torch.from_numpy(x).float(), torch.from_numpy(y).float())

        def _local_predict(self, x: Any) -> Any:
            y_pred = super()._local_predict(torch.from_numpy(x).float())
            return y_pred.detach().numpy()

    my_algo = MyAlgo()
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy"],
        editable_mode=True,
    )

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

    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuple = testtuples[0]
    assert list(testtuple.test.perfs.values())[0] == pytest.approx(expected_performance, rel=10e-6)
