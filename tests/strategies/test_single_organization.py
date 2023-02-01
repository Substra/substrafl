import pytest

from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.strategies import SingleOrganization


@pytest.mark.parametrize("additional_orgs_permissions", [set(), {"TestId"}, {"TestId1", "TestId2"}])
def test_single_organization_train_tasks_output_permissions(dummy_algo_class, additional_orgs_permissions):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    train_data_node = TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"])

    strategy = SingleOrganization()

    strategy.perform_round(
        algo=dummy_algo_class(),
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
