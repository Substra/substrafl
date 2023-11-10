import numpy as np
import pytest

from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.strategies import FedPCA
from substrafl.strategies.schemas import FedPCASharedState


@pytest.mark.parametrize(
    "n_samples, results",
    [
        ([1, 0, 0], np.ones((5, 10))),
        ([1, 1, 1], np.ones((5, 10))),
        ([1, 0, 1], 1.5 * np.ones((5, 10))),
    ],
)
def test_avg_shared_states(dummy_algo_class, n_samples, results):
    """First shared states have ones only, second one zeros only, and third twos only.
    The n samples list in the multiplicative coefficient for each of these states.
    """
    shared_states = [
        FedPCASharedState(parameters_update=[np.ones((5, 10))], n_samples=n_samples[0]),
        FedPCASharedState(parameters_update=[np.zeros((5, 10))], n_samples=n_samples[1]),
        FedPCASharedState(parameters_update=[2 * np.ones((5, 10))], n_samples=n_samples[2]),
    ]

    MyFedPCA = FedPCA(algo=dummy_algo_class())
    averaged_states = MyFedPCA.avg_shared_states(shared_states, _skip=True)

    assert (results == averaged_states.avg_parameters_update).all()


@pytest.mark.parametrize(
    "shared_states, results",
    [
        (
            [
                FedPCASharedState(parameters_update=[np.array([[0.5, 0, 0], [1, 0, 1.5], [2, 2.5, 3]])], n_samples=2),
                FedPCASharedState(parameters_update=[np.array([[1, 0, 0], [2, 0, 3], [4, 5, 6]])], n_samples=1),
            ],
            [np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])],
        ),
        (
            [
                FedPCASharedState(
                    parameters_update=[np.array([[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]])], n_samples=1
                ),
            ],
            [np.array([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])],
        ),
    ],
)
def test_avg_shared_states_with_qr(dummy_algo_class, shared_states, results, rtol):
    MyFedPCA = FedPCA(algo=dummy_algo_class())
    averaged_states_with_qr = MyFedPCA.avg_shared_states_with_qr(shared_states, _skip=True)

    assert np.array(
        [
            np.allclose(np.dot(results[0][i], row) * row, results[0][i], rtol=rtol)
            for i, row in enumerate(averaged_states_with_qr.avg_parameters_update[0])
        ]
    ).all()


def test_fed_pca_perform_round(dummy_algo_class):
    """Test that perform round create the right number of tasks"""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    aggregation_node = AggregationNode("DummyNode0")
    strategy = FedPCA(algo=dummy_algo_class())

    strategy.perform_round(
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        round_idx=1,
        clean_models=False,
    )
    assert len(aggregation_node.tasks) == 1
    # Perform round create two train tasks at round 1, to compute one aggregation step on the first round.
    assert all([len(train_data_node.tasks) == 2 for train_data_node in train_data_nodes])

    strategy.perform_round(
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        round_idx=2,
        clean_models=False,
    )
    assert len(aggregation_node.tasks) == 1 + 1
    # Perform round create one train tasks at round 2, to compute one aggregation step on the first round.
    assert all([len(train_data_node.tasks) == 2 + 1 for train_data_node in train_data_nodes])


def test_fed_pca_predict(dummy_algo_class):
    """Test that the predict function updates the TestDataNode tasks starting from round 3."""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    test_data_nodes = [
        TestDataNode(
            "DummyNode0",
            "dummy_key",
            ["dummy_key"],
        ),
        TestDataNode(
            "DummyNode1",
            "dummy_key",
            ["dummy_key"],
        ),
    ]

    strategy = FedPCA(algo=dummy_algo_class())

    strategy._local_states = [
        LocalStateRef(key="dummy_key"),
        LocalStateRef(key="dummy_key"),
    ]

    strategy.perform_evaluation(
        test_data_nodes=test_data_nodes,
        train_data_nodes=train_data_nodes,
        round_idx=1,
    )

    assert all([len(test_data_node.tasks) == 0 for test_data_node in test_data_nodes])

    strategy.perform_evaluation(
        test_data_nodes=test_data_nodes,
        train_data_nodes=train_data_nodes,
        round_idx=3,
    )

    assert all([len(test_data_node.tasks) == 1 for test_data_node in test_data_nodes])


@pytest.mark.parametrize("additional_orgs_permissions", [set(), {"TestId"}, {"TestId1", "TestId2"}])
def test_fed_pca_train_tasks_output_permissions(dummy_algo_class, additional_orgs_permissions):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    aggregation_node = AggregationNode("DummyNode0")
    strategy = FedPCA(algo=dummy_algo_class())

    strategy.perform_round(
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        round_idx=1,
        clean_models=False,
        additional_orgs_permissions=additional_orgs_permissions,
    )

    for train_data_node in train_data_nodes:
        assert all(
            [
                additional_orgs_permissions.intersection(set(task["outputs"]["local"]["permissions"]["authorized_ids"]))
                == additional_orgs_permissions
                for task in train_data_node.tasks
            ]
        )
        assert all(
            [
                additional_orgs_permissions.intersection(
                    set(task["outputs"]["shared"]["permissions"]["authorized_ids"])
                )
                == additional_orgs_permissions
                for task in train_data_node.tasks
            ]
        )
