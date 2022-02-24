from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from connectlib import execute_experiment
from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.remote.methods import remote_data
from connectlib.strategies import FedAVG


# mocking the add_compute_plan as we don't want to test Substra, just the execute_experiment
@patch("substra.Client.add_compute_plan", MagicMock(return_value=np.recarray(1, dtype=[("key", int)])))
def test_execute_experiment_has_no_side_effect(network, train_linear_nodes, test_linear_nodes, aggregation_node):
    """Ensure that the execute_experiment run twice won't fail (which would be the case if the variables passed
    changed during the run). It mocks the add_compute_plan() of Substra so that substra code is never really
    executed"""

    class MyAlgo(Algo):
        # No need for full Algo as it is never really submitted to Substra for a run
        @remote_data
        def train(self, x, y, shared_state):
            pass

        @remote_data
        def predict(self, x, shared_state):
            pass

        def load(self, path):
            pass

        def save(self, path):
            pass

    num_rounds = 2
    my_algo0 = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["pytest"], editable_mode=True)
    strategy = FedAVG()
    # test every two rounds
    my_eval_strategy = EvaluationStrategy(test_data_nodes=test_linear_nodes, rounds=2)

    cp1 = execute_experiment(
        client=network.clients[0],
        algo=my_algo0,
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )

    # this second run fails if the variables changed in the first run
    cp2 = execute_experiment(
        client=network.clients[0],
        algo=my_algo0,
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )

    assert sum(len(node.tuples) for node in test_linear_nodes) == 0
    assert sum(len(node.tuples) for node in train_linear_nodes) == 0
    assert len(aggregation_node.tuples) == 0
    assert cp1 == cp2
