import numpy as np
import pytest

from substrafl.exceptions import DampingFactorValueError
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote.decorators import remote_data
from substrafl.schemas import NewtonRaphsonAveragedStates
from substrafl.schemas import NewtonRaphsonSharedState
from substrafl.strategies import NewtonRaphson


@pytest.mark.parametrize(
    "damping_factor, list_gradients, list_hessian, list_n_sample, parameters_update",
    [
        (
            1,
            [[np.array([[2]]), np.array([2])], [np.array([[1]]), np.array([1])]],
            [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
            [1, 1],
            [np.array([[-1.5]]), np.array([-1.5])],
        ),
        (
            0.8,
            [[np.array([[6, 4]]), np.array([3])], [np.array([[3, 1]]), np.array([3])]],
            [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])],
            [1, 2],
            [np.array([[-0.8 * (4 / 3), -0.8 * (2 / 3)]]), np.array([-0.8 * (3 / 3)])],
        ),
    ],
)
def test_compute_aggregated_states(damping_factor, list_gradients, list_hessian, list_n_sample, parameters_update):
    """Test that compute_aggregated_states is doing the correct calculations
    Equation used:
    H = weighted_average of hessians
    G = weighted_average of gradients
    parameters_update = -damping_factor * H^{-1}.G
    """
    strategy = NewtonRaphson(damping_factor=damping_factor)
    shared_states = []
    # create share state from each gradients, hessian and n_sample for each client
    for gradients, hessian, n_sample in zip(list_gradients, list_hessian, list_n_sample):
        shared_states.append(NewtonRaphsonSharedState(gradients=gradients, hessian=hessian, n_samples=n_sample))

    averaged_state = strategy.compute_aggregated_states(shared_states=shared_states, _skip=True)

    assert all(
        [
            np.allclose(obtained, expected)
            for obtained, expected in zip(averaged_state.parameters_update, parameters_update)
        ]
    )


@pytest.mark.parametrize(
    "damping_factor, wrong_eta_value",
    [
        (0, True),
        (1, False),
        (-1, True),
        (2, True),
    ],
)
def test_eta_value(damping_factor, wrong_eta_value):
    """Test that the EtaValueError is raised if not 0 < damping_factor <= 1"""
    if wrong_eta_value:
        with pytest.raises(DampingFactorValueError):
            NewtonRaphson(damping_factor=damping_factor)
    else:
        NewtonRaphson(damping_factor=damping_factor)


def test_newton_raphson_perform_round(dummy_algo_class):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    class MyAlgo(dummy_algo_class):
        @remote_data
        def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            shared_state,
        ):
            gradients = np.array([0, 0])
            hessian = np.eye(2)

            return NewtonRaphsonSharedState(n_samples=len(x), gradients=gradients, hessian=hessian)

        @remote_data
        def predict(self, x: np.array, shared_state: NewtonRaphsonAveragedStates):
            return shared_state.parameters_update

    # Add 0s and 1s constant to check the averaging of the function
    # We predict the shared state, an array of 0.5

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    aggregation_node = AggregationNode("DummyNode0")
    my_algo0 = MyAlgo()
    strategy = NewtonRaphson(damping_factor=1)

    strategy.perform_round(
        algo=my_algo0,
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        round_idx=1,
        clean_models=False,
    )
    assert len(aggregation_node.tasks) == 1
    assert all([len(train_data_node.tasks) == 1 for train_data_node in train_data_nodes])


def test_newton_raphson_predict(dummy_algo_class):
    """Test that the predict function updates the TestDataNode.tasks."""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    test_data_nodes = [
        TestDataNode(
            "DummyNode0",
            "dummy_key",
            ["dummy_key"],
            metric_keys=["dummy_key"],
        ),
        TestDataNode(
            "DummyNode1",
            "dummy_key",
            ["dummy_key"],
            metric_keys=["dummy_key"],
        ),
    ]

    strategy = NewtonRaphson(damping_factor=1)

    strategy._local_states = [
        LocalStateRef(key="dummy_key"),
        LocalStateRef(key="dummy_key"),
    ]

    strategy.predict(
        algo=dummy_algo_class(),
        test_data_nodes=test_data_nodes,
        train_data_nodes=train_data_nodes,
        round_idx=0,
    )

    assert all([len(test_data_node.testtasks) == 1 for test_data_node in test_data_nodes])
    assert all([len(test_data_node.predicttasks) == 1 for test_data_node in test_data_nodes])


@pytest.mark.parametrize("additional_orgs_permissions", [set(), {"TestId"}, {"TestId1", "TestId2"}])
def test_newton_raphson_train_tasks_output_permissions(dummy_algo_class, additional_orgs_permissions):
    """Test that perform round updates the strategy._local_states and strategy._shared_states"""

    train_data_nodes = [
        TrainDataNode("DummyNode0", "dummy_key", ["dummy_key"]),
        TrainDataNode("DummyNode1", "dummy_key", ["dummy_key"]),
    ]

    aggregation_node = AggregationNode("DummyNode0")
    strategy = NewtonRaphson(damping_factor=1)

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
