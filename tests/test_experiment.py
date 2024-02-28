from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import substra

from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.exceptions import LenMetadataError
from substrafl.experiment import execute_experiment
from substrafl.experiment import simulate_experiment
from substrafl.nodes.schemas import SimuPerformancesMemory
from substrafl.nodes.schemas import SimuStatesMemory
from substrafl.strategies import FedAvg


# mocking the add_compute_plan as we don't want to test Substra, just the execute_experiment
@patch("substra.Client.add_compute_plan", MagicMock(return_value=np.recarray(1, dtype=[("key", int)])))
def test_execute_experiment_has_no_side_effect(
    network,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
    session_dir,
    dummy_algo_class,
):
    """Ensure that the execute_experiment run twice won't fail (which would be the case if the variables passed
    changed during the run). It mocks the add_compute_plan() of Substra so that substra code is never really
    executed"""

    num_rounds = 2
    dummy_algo_instance = dummy_algo_class()
    algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
    strategy = FedAvg(algo=dummy_algo_instance)
    # test every two rounds
    my_eval_strategy = EvaluationStrategy(test_data_nodes=test_linear_nodes, eval_frequency=2)

    cp1 = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )

    # this second run fails if the variables changed in the first run
    cp2 = execute_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )

    assert sum(len(node.tasks) for node in test_linear_nodes) == 0
    assert sum(len(node.tasks) for node in train_linear_nodes) == 0
    assert len(aggregation_node.tasks) == 0
    assert cp1 == cp2


def test_too_long_additional_metadata(session_dir, dummy_strategy_class, dummy_algo_class):
    """Test if the LenMetadataError is raised when a too long Metadata
    is given to the additional_metadata dictionary."""

    client = Mock(spec=substra.Client)
    additional_metadata = {"first_arg": "size_ok", "second_arg": "size_too_long" * 10}
    with pytest.raises(LenMetadataError):
        execute_experiment(
            client=client,
            strategy=dummy_strategy_class(algo=dummy_algo_class()),
            train_data_nodes=[],
            evaluation_strategy=None,
            aggregation_node=None,
            num_rounds=2,
            dependencies=None,
            experiment_folder=session_dir / "experiment_folder",
            additional_metadata=additional_metadata,
        )


def test_simulate_experiment(
    network,
    train_linear_nodes,
    test_linear_nodes,
    aggregation_node,
    dummy_strategy_class,
    dummy_algo_class,
    session_dir,
):
    num_rounds = 2
    dummy_algo_instance = dummy_algo_class()
    strategy = dummy_strategy_class(algo=dummy_algo_instance)

    my_eval_strategy = EvaluationStrategy(test_data_nodes=test_linear_nodes, eval_frequency=1)

    perf, train_states, aggregated_states = simulate_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        clean_models=False,
        experiment_folder=session_dir / "experiment_folder",
    )

    assert isinstance(perf, SimuPerformancesMemory)
    assert isinstance(train_states, SimuStatesMemory)
    assert isinstance(aggregated_states, SimuStatesMemory)


def test_simulate_experiment_no_test_and_agg(
    network,
    train_linear_nodes,
    dummy_strategy_class,
    dummy_algo_class,
    session_dir,
):
    num_rounds = 2
    dummy_algo_instance = dummy_algo_class()
    strategy = dummy_strategy_class(algo=dummy_algo_instance)

    perf, train_states, aggregated_states = simulate_experiment(
        client=network.clients[0],
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=None,
        num_rounds=num_rounds,
        experiment_folder=session_dir / "experiment_folder",
    )

    assert perf is None
    assert isinstance(train_states, SimuStatesMemory)
    assert aggregated_states is None
