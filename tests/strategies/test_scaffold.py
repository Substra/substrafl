from logging import getLogger
from pathlib import Path

import numpy as np
import pytest

from connectlib import execute_experiment
from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.nodes.aggregation_node import AggregationNode
from connectlib.nodes.test_data_node import TestDataNode
from connectlib.nodes.train_data_node import TrainDataNode
from connectlib.remote import remote_data
from connectlib.schemas import ScaffoldAveragedStates
from connectlib.schemas import ScaffoldSharedState
from connectlib.strategies import Scaffold

from .. import assets_factory
from .. import utils

logger = getLogger("tests")


def assert_array_list_allclose(array_list_1, array_list_2, rtol):
    assert len(array_list_1) == len(array_list_2)

    for array1, array2 in zip(array_list_1, array_list_2):
        assert np.allclose(array1, array2, rtol)


@pytest.mark.parametrize(
    "n_samples, results",
    [
        ([1, 0, 0], [np.ones((2, 3)), np.ones((1, 2))]),
        ([1, 1, 1], [np.ones((2, 3)), np.ones((1, 2))]),
        ([1, 0, 1], [1.5 * np.ones((2, 3)), 1.5 * np.ones((1, 2))]),
    ],
)
def test_avg_shared_states(n_samples, results, rtol):
    # Check that avg_shared_states sends the average of weight_updates and control_variate_updates
    weights = [
        [np.ones((2, 3)), np.ones((1, 2))],
        [np.zeros((2, 3)), np.zeros((1, 2))],
        [2 * np.ones((2, 3)), 2 * np.ones((1, 2))],
    ]

    shared_states = [
        ScaffoldSharedState(
            weight_update=weight,
            control_variate_update=weight,
            n_samples=n_sample,
            server_control_variate=[np.zeros((2, 3)), np.zeros((1, 2))],
        )
        for weight, n_sample in zip(weights, n_samples)
    ]
    my_scaffold = Scaffold(aggregation_lr=1)
    averaged_states: ScaffoldAveragedStates = my_scaffold.avg_shared_states(shared_states, _skip=True)

    assert_array_list_allclose(array_list_1=results, array_list_2=averaged_states.avg_weight_update, rtol=rtol)
    # as server_control_variate = np.zeros and aggregation_lr=1, the new server_control_variate is equal
    # to avg_weight_update == results
    assert_array_list_allclose(array_list_1=results, array_list_2=averaged_states.server_control_variate, rtol=rtol)


@pytest.mark.parametrize(
    "shared_states",
    [
        [],
        ScaffoldSharedState(
            weight_update=[np.array([0, 1, 1])],
            control_variate_update=[np.array([0, 1, 1])],
            n_samples=1,
            server_control_variate=[np.array([0, 1, 1])],
        ),
    ],
)
def test_avg_shared_states_type_error(shared_states):
    # check if an empty list or something else than a List is not passed into avg_shared_states() error will be raised
    my_scaffold = Scaffold()
    with pytest.raises(AssertionError):
        my_scaffold.avg_shared_states(shared_states, _skip=True)


@pytest.mark.slow
@pytest.mark.substra
@pytest.mark.parametrize(
    "aggregation_lr",
    [
        1,
        2,
        0,
    ],
)
def test_scaffold(network, constant_samples, numpy_datasets, session_dir, default_permissions, aggregation_lr):
    # makes sure that Scaffold strategy leads to the averaging output of the models from both partners.
    # The data for the two partners consists of only 0s or 1s respectively. The train() returns the data.
    # predict() returns the data, score returned by AccuracyMetric (in the metric) is the mean of all the y_pred
    # passed to it. The tests asserts if the score is 0.5
    # This test only runs on two nodes.

    # Add 0s and 1s constant to check the averaging of the function
    num_rounds = 2

    train_data_nodes = [
        TrainDataNode(network.msp_ids[0], numpy_datasets[0], [constant_samples[0]]),
        TrainDataNode(network.msp_ids[1], numpy_datasets[1], [constant_samples[1]]),
    ]

    class MyStrategy(Scaffold):
        def __init__(self):
            super().__init__(aggregation_lr=aggregation_lr)

    class MyAlgo(Algo):
        # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
        # pytest calls it)
        @property
        def model(self):
            return None

        @remote_data
        def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            shared_state,
        ):
            return ScaffoldSharedState(
                weight_update=[x],
                control_variate_update=[x * 2],
                server_control_variate=[np.zeros_like(x)],
                n_samples=len(x),
            )

        @remote_data
        def predict(self, x: np.array, shared_state: ScaffoldAveragedStates):
            # avg_weight_update = mean(weight_update) * aggregation_lr = (0+1)/2 * aggregation_lr = 1/2 * aggregation_lr
            assert shared_state.avg_weight_update[0] == np.ones(1) * 0.5 * aggregation_lr
            # server_control_variate = server_control_variate + mean(control_variate_update) = 0 + (0+2)/2 = 1
            assert shared_state.server_control_variate[0] == np.ones(1)
            # return should be 0.5 * aggregation_lr + 1
            return shared_state.avg_weight_update[0] + shared_state.server_control_variate[0]

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    # Check that all shared states values are 0.5 * aggregation_lr + 1
    metric = assets_factory.add_python_metric(
        client=network.clients[0],
        tmp_folder=session_dir,
        python_formula=f"int((y_pred == np.ones(1) * ( 0.5 * {aggregation_lr} + 1) ).all())",
        name="Average",
        permissions=default_permissions,
    )

    test_data_nodes = [
        TestDataNode(
            network.msp_ids[0],
            numpy_datasets[0],
            [constant_samples[0]],
            metric_keys=[metric],
        ),
        TestDataNode(
            network.msp_ids[1],
            numpy_datasets[1],
            [constant_samples[1]],
            metric_keys=[metric],
        ),
    ]

    aggregation_node = AggregationNode(network.msp_ids[0])
    my_algo0 = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
    strategy = MyStrategy()
    # test every two rounds
    my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=2)
    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo0,
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
        experiment_folder=session_dir / "experiment_folder",
    )
    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    # read the results from saved performances
    testtuples = network.clients[0].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])
    testtuple = testtuples[0]
    # assert that the metrics returns int(True) i.e. 1
    assert list(testtuple.test.perfs.values())[0] == 1
