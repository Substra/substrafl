import numpy as np
import pytest

from connectlib.exceptions import DampingFactorValueError
from connectlib.nodes.aggregation_node import AggregationNode
from connectlib.nodes.references.local_state import LocalStateRef
from connectlib.nodes.test_data_node import TestDataNode
from connectlib.nodes.train_data_node import TrainDataNode
from connectlib.remote.decorators import remote_data
from connectlib.schemas import NewtonRaphsonAveragedStates
from connectlib.schemas import NewtonRaphsonSharedState
from connectlib.strategies import NewtonRaphson


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
def test_compute_averaged_states(damping_factor, list_gradients, list_hessian, list_n_sample, parameters_update):
    """Test that compute_averaged_states is doing the correct calculations
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

    averaged_state = strategy.compute_averaged_states(shared_states=shared_states, _skip=True)

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
        round_idx=0,
    )

    assert len(aggregation_node.tuples) == 1
    assert all([len(train_data_node.tuples) == 1 for train_data_node in train_data_nodes])


def test_newton_raphson_predict():
    """Test that the predict function updates the TestDataNode.tuples."""

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
        test_data_nodes=test_data_nodes,
        train_data_nodes=train_data_nodes,
        round_idx=0,
    )

    assert all([len(test_data_node.tuples) == 1 for test_data_node in test_data_nodes])
