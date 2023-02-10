from logging import getLogger

import numpy as np
import pytest

from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.schemas import ScaffoldAveragedStates
from substrafl.schemas import ScaffoldSharedState
from substrafl.strategies import Scaffold

logger = getLogger("tests")


def assert_array_list_allclose(array_list_1, array_list_2):
    assert len(array_list_1) == len(array_list_2)

    for array1, array2 in zip(array_list_1, array_list_2):
        assert np.allclose(array1, array2)


@pytest.mark.parametrize(
    "n_samples, results",
    [
        ([1, 0, 0], [np.ones((2, 3)), np.ones((1, 2))]),
        ([1, 1, 1], [np.ones((2, 3)), np.ones((1, 2))]),
        ([1, 0, 1], [1.5 * np.ones((2, 3)), 1.5 * np.ones((1, 2))]),
    ],
)
def test_compute_aggregated_states_n_samples(n_samples, results):
    # Check that compute_aggregated_states sends the average of weight_updates and control_variate_updates
    weights = [
        [np.ones((2, 3)), np.ones((1, 2))],
        [np.zeros((2, 3)), np.zeros((1, 2))],
        [2 * np.ones((2, 3)), 2 * np.ones((1, 2))],
    ]

    shared_states = [
        ScaffoldSharedState(
            parameters_update=weight,
            control_variate_update=weight,
            n_samples=n_sample,
            server_control_variate=[np.zeros((2, 3)), np.zeros((1, 2))],
        )
        for weight, n_sample in zip(weights, n_samples)
    ]
    my_scaffold = Scaffold(aggregation_lr=1)
    averaged_states: ScaffoldAveragedStates = my_scaffold.compute_aggregated_states(shared_states, _skip=True)

    assert_array_list_allclose(array_list_1=results, array_list_2=averaged_states.avg_parameters_update)
    # as server_control_variate = np.zeros and aggregation_lr=1, the new server_control_variate is equal
    # to avg_parameters_update == results
    assert_array_list_allclose(array_list_1=results, array_list_2=averaged_states.server_control_variate)


@pytest.mark.parametrize(
    "shared_states",
    [
        [],
        ScaffoldSharedState(
            parameters_update=[np.array([0, 1, 1])],
            control_variate_update=[np.array([0, 1, 1])],
            n_samples=1,
            server_control_variate=[np.array([0, 1, 1])],
        ),
    ],
)
def test_compute_aggregated_states_type_error(shared_states):
    # check if an empty list or something else than a List is not passed into compute_aggregated_states()
    # error will be raised
    my_scaffold = Scaffold()
    with pytest.raises(AssertionError):
        my_scaffold.compute_aggregated_states(shared_states, _skip=True)


def test_scaffold_aggregation_lr_negative():
    with pytest.raises(ValueError):
        Scaffold(aggregation_lr=-1)


@pytest.mark.parametrize(
    "parameters_update, control_variate_update, server_control_variate",
    [
        ([np.zeros(5), np.zeros(5)], [np.zeros(5)], [np.zeros(5)]),
        ([np.zeros(5)], [np.zeros(5), np.zeros(5)], [np.zeros(5)]),
        ([np.zeros(5)], [np.zeros(5)], [np.zeros(5), np.zeros(5)]),
    ],
)
def test_check_len_states_same(parameters_update, control_variate_update, server_control_variate):
    """Check that for a given client parameters_update, control_variate_update and server_control_variate have the same
    length."""
    shared_states = [
        ScaffoldSharedState(
            parameters_update=parameters_update,
            control_variate_update=control_variate_update,
            n_samples=1,
            server_control_variate=server_control_variate,
        ),
    ]
    strategy = Scaffold(aggregation_lr=0)
    with pytest.raises(AssertionError):
        strategy.compute_aggregated_states(shared_states=shared_states, _skip=True)


@pytest.mark.parametrize(
    "parameters_update, control_variate_update, server_control_variate",
    [
        (
            [[np.zeros(5)], [np.zeros(5), np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5)]],
        ),
        (
            [[np.zeros(5)], [np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5), np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5)]],
        ),
        (
            [[np.zeros(5)], [np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5)]],
            [[np.zeros(5)], [np.zeros(5), np.zeros(5)]],
        ),
    ],
)
def test_check_same_len_between_clients(parameters_update, control_variate_update, server_control_variate):
    """ "Check that between all clients the parameters update have the same length, the control variate update
    have the same length and the server control variate have the same length
    """
    shared_states = [
        ScaffoldSharedState(
            parameters_update=parameters_update[0],
            control_variate_update=control_variate_update[0],
            n_samples=1,
            server_control_variate=server_control_variate[0],
        ),
        ScaffoldSharedState(
            parameters_update=parameters_update[1],
            control_variate_update=control_variate_update[1],
            n_samples=1,
            server_control_variate=server_control_variate[1],
        ),
    ]
    strategy = Scaffold(aggregation_lr=0)
    with pytest.raises(AssertionError):
        strategy.compute_aggregated_states(shared_states=shared_states, _skip=True)


@pytest.mark.parametrize(
    "aggregation_lr, expected_result",
    [
        (0, [np.zeros(5), np.zeros(5)]),
        (1, [0.75 * np.ones(5), 1.75 * np.ones(5)]),
        (2, [1.5 * np.ones(5), 3.5 * np.ones(5)]),
    ],
)
def test_scaffold_compute_aggregated_states_aggregation_lr(aggregation_lr, expected_result):
    strategy = Scaffold(aggregation_lr=aggregation_lr)
    shared_states = [
        ScaffoldSharedState(
            parameters_update=[0 * np.ones(5), np.ones(5)],
            control_variate_update=[np.ones(5), np.ones(5)],
            n_samples=1,
            server_control_variate=[np.ones(5), np.ones(5)],
        ),
        ScaffoldSharedState(
            parameters_update=[np.ones(5), 2 * np.ones(5)],
            control_variate_update=[np.ones(5), np.ones(5)],
            n_samples=3,
            server_control_variate=[np.ones(5), np.ones(5)],
        ),
    ]
    averaged_states = strategy.compute_aggregated_states(shared_states=shared_states, _skip=True)
    assert_array_list_allclose(expected_result, averaged_states.avg_parameters_update)
    # Check that the aggregation lr has no incidence on the server control variate
    assert_array_list_allclose([2 * np.ones(5), 2 * np.ones(5)], averaged_states.server_control_variate)


@pytest.mark.parametrize(
    "server_control_variate, expected_result",
    [
        ([np.zeros(5)], [np.ones(5)]),
        ([np.ones(5)], [2 * np.ones(5)]),
        ([np.ones(5), 2 * np.ones(5)], [2 * np.ones(5), 3 * np.ones(5)]),
    ],
)
def test_scaffold_compute_aggregated_states_server_control_variate(server_control_variate, expected_result):
    strategy = Scaffold(aggregation_lr=1)
    shared_states = [
        ScaffoldSharedState(
            parameters_update=[np.ones(5)] * len(server_control_variate),
            control_variate_update=[np.ones(5)] * len(server_control_variate),
            n_samples=1,
            server_control_variate=server_control_variate,
        ),
    ]
    averaged_states = strategy.compute_aggregated_states(shared_states=shared_states, _skip=True)
    assert_array_list_allclose(expected_result, averaged_states.server_control_variate)


@pytest.mark.parametrize("additional_orgs_permissions", [set(), {"TestId"}, {"TestId1", "TestId2"}])
def test_scaffold_train_tasks_output_permissions(dummy_algo_class, additional_orgs_permissions):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    aggregation_node = AggregationNode("DummyNode0")
    strategy = Scaffold()

    strategy.perform_round(
        algo=dummy_algo_class(),
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
