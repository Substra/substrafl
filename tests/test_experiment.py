from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import substra

from connectlib import execute_experiment
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.exceptions import IncompatibleAlgoStrategyError
from connectlib.exceptions import LenMetadataError
from connectlib.strategies import FedAvg


# mocking the add_compute_plan as we don't want to test Substra, just the execute_experiment
@patch("substra.Client.add_compute_plan", MagicMock(return_value=np.recarray(1, dtype=[("key", int)])))
def test_execute_experiment_has_no_side_effect(
    network,
    train_linear_organizations,
    test_linear_organizations,
    aggregation_organization,
    session_dir,
    dummy_algo_class,
):
    """Ensure that the execute_experiment run twice won't fail (which would be the case if the variables passed
    changed during the run). It mocks the add_compute_plan() of Substra so that substra code is never really
    executed"""

    num_rounds = 2
    algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
    strategy = FedAvg()
    # test every two rounds
    my_eval_strategy = EvaluationStrategy(test_data_organizations=test_linear_organizations, rounds=2)
    dummy_algo_instance = dummy_algo_class()

    cp1 = execute_experiment(
        client=network.clients[0],
        algo=dummy_algo_instance,
        strategy=strategy,
        train_data_organizations=train_linear_organizations,
        evaluation_strategy=my_eval_strategy,
        aggregation_organization=aggregation_organization,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )

    # this second run fails if the variables changed in the first run
    cp2 = execute_experiment(
        client=network.clients[0],
        algo=dummy_algo_instance,
        strategy=strategy,
        train_data_organizations=train_linear_organizations,
        evaluation_strategy=my_eval_strategy,
        aggregation_organization=aggregation_organization,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )

    assert sum(len(organization.tuples) for organization in test_linear_organizations) == 0
    assert sum(len(organization.tuples) for organization in train_linear_organizations) == 0
    assert len(aggregation_organization.tuples) == 0
    assert cp1 == cp2


def test_too_long_additional_metadata(session_dir, dummy_strategy_class, dummy_algo_class):
    """Test if the LenMetadataError is raised when a too long Metadata
    is given to the additional_metadata dictionary."""

    client = Mock(spec=substra.Client)
    additional_metadata = {"first_arg": "size_ok", "second_arg": "size_too_long" * 10}
    with pytest.raises(LenMetadataError):
        execute_experiment(
            client=client,
            algo=dummy_algo_class(),
            strategy=dummy_strategy_class(),
            train_data_organizations=[],
            evaluation_strategy=None,
            aggregation_organization=None,
            num_rounds=2,
            dependencies=None,
            experiment_folder=session_dir / "experiment_folder",
            additional_metadata=additional_metadata,
        )


def test_match_algo_strategy(session_dir, dummy_strategy_class, dummy_algo_class):
    client = Mock(spec=substra.Client)

    class MyAlgo(dummy_algo_class):
        @property
        def strategies(self):
            return ["not_the_dummy_strategy"]

    with pytest.raises(IncompatibleAlgoStrategyError):
        execute_experiment(
            client=client,
            algo=MyAlgo(),
            strategy=dummy_strategy_class(),
            train_data_organizations=[],
            evaluation_strategy=None,
            aggregation_organization=None,
            num_rounds=2,
            dependencies=None,
            experiment_folder=session_dir / "experiment_folder",
        )
