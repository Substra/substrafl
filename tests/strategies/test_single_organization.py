from contextlib import nullcontext as does_not_raise

import pytest

from substrafl import exceptions
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.schemas import StrategyName
from substrafl.strategies import SingleOrganization


@pytest.mark.parametrize(
    "strategy_name, expectation",
    [
        ("not_the_dummy_strategy", pytest.raises(exceptions.IncompatibleAlgoStrategyError)),
        (StrategyName.SINGLE_ORGANIZATION, does_not_raise()),
    ],
)
def test_match_algo_single_organization(strategy_name, dummy_algo_class, expectation):
    class MyAlgo(dummy_algo_class):
        @property
        def strategies(self):
            return [strategy_name]

    with expectation:
        SingleOrganization(algo=MyAlgo())


@pytest.mark.parametrize("additional_orgs_permissions", [set(), {"TestId"}, {"TestId1", "TestId2"}])
def test_single_organization_train_tasks_output_permissions(dummy_algo_class, additional_orgs_permissions):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    train_data_node = TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"])

    strategy = SingleOrganization(algo=dummy_algo_class())

    strategy.perform_round(
        train_data_nodes=[train_data_node],
        round_idx=1,
        clean_models=False,
        additional_orgs_permissions=additional_orgs_permissions,
    )

    assert all(
        [
            additional_orgs_permissions.intersection(set(task["outputs"]["local"]["permissions"]["authorized_ids"]))
            == additional_orgs_permissions
            for task in train_data_node.tasks
        ]
    )
    assert all(
        [
            additional_orgs_permissions.intersection(set(task["outputs"]["shared"]["permissions"]["authorized_ids"]))
            == additional_orgs_permissions
            for task in train_data_node.tasks
        ]
    )
