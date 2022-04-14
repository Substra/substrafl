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
from connectlib.schemas import FedAvgAveragedState
from connectlib.schemas import FedAvgSharedState
from connectlib.strategies import FedAvg

from .. import assets_factory
from .. import utils

logger = getLogger("tests")


@pytest.mark.parametrize(
    "n_samples, results",
    [
        ([1, 0, 0], np.ones((5, 10))),
        ([1, 1, 1], np.ones((5, 10))),
        ([1, 0, 1], 1.5 * np.ones((5, 10))),
    ],
)
def test_avg_shared_states(n_samples, results):

    shared_states = [
        FedAvgSharedState(parameters_update=[np.ones((5, 10))], n_samples=n_samples[0]),
        FedAvgSharedState(parameters_update=[np.zeros((5, 10))], n_samples=n_samples[1]),
        FedAvgSharedState(parameters_update=[2 * np.ones((5, 10))], n_samples=n_samples[2]),
    ]

    MyFedAvg = FedAvg()
    averaged_states = MyFedAvg.avg_shared_states(shared_states, _skip=True)

    assert (results == averaged_states.avg_parameters_update).all()


def test_avg_shared_states_different_layers():
    shared_states = [
        FedAvgSharedState(
            parameters_update=[np.asarray([[0, 1], [2, 4]]), np.asarray([[6, 8], [10, 12]])], n_samples=1
        ),
        FedAvgSharedState(
            parameters_update=[np.asarray([[16, 20], [18, 20]]), np.asarray([[22, 24], [26, 28]])], n_samples=3
        ),
    ]

    MyFedAvg = FedAvg()
    avg_states = MyFedAvg.avg_shared_states(shared_states, _skip=True)
    expected_result = [np.asarray([[12, 15.25], [14, 16]]), np.asarray([[18, 20], [22, 24]])]
    assert np.allclose(avg_states.avg_parameters_update, expected_result)


def test_avg_shared_states_different_length():
    shared_states = [
        FedAvgSharedState(parameters_update=[np.ones((5, 10)), np.ones((5, 10))], n_samples=1),
        FedAvgSharedState(parameters_update=[np.zeros((5, 10))], n_samples=1),
    ]

    MyFedAvg = FedAvg()
    with pytest.raises(AssertionError):
        MyFedAvg.avg_shared_states(shared_states, _skip=True)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.substra
def test_fed_avg(network, constant_samples, numpy_datasets, session_dir, default_permissions):
    # makes sure that federated average strategy leads to the averaging output of the models from both partners.
    # The data for the two partners consists of only 0s or 1s respectively. The train() returns the data.
    # predict() returns the data, score returned by AccuracyMetric (in the metric) is the mean of all the y_pred
    # passed to it. The tests asserts if the score is 0.5
    # This test only runs on two nodes.

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
            return FedAvgSharedState(n_samples=len(x), parameters_update=[np.asarray(e) for e in x])

        @remote_data
        def predict(self, x: np.array, shared_state: FedAvgAveragedState):
            return shared_state.avg_parameters_update

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    # Add 0s and 1s constant to check the averaging of the function
    # We predict the shared state, an array of 0.5

    metric = assets_factory.add_python_metric(
        client=network.clients[0],
        tmp_folder=session_dir,
        # Check that all shared states values are 0.5
        python_formula="int((y_pred == np.ones(1)*0.5).all())",
        name="Average",
        permissions=default_permissions,
    )
    train_data_nodes = [
        TrainDataNode(network.msp_ids[0], numpy_datasets[0], [constant_samples[0]]),
        TrainDataNode(network.msp_ids[1], numpy_datasets[1], [constant_samples[1]]),
    ]

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

    num_rounds = 2
    aggregation_node = AggregationNode(network.msp_ids[0])
    my_algo0 = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
    strategy = FedAvg()
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
