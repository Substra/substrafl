from unittest.mock import Mock

import pytest

from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.nodes.test_data_node import TestDataNode


@pytest.mark.parametrize("rounds", [1, 2, 4, 10, [1], [1, 4], [5, 1, 7, 3]])
def test_rounds(rounds):
    # tests that each next() returns expected True or False
    # tests that next called > num_rounds raises StopIteration
    n_organizations = 3
    num_rounds = 10
    # test rounds as frequencies give expected result
    # mock the test organizations
    test_data_organization = Mock(spec=TestDataNode)
    test_data_nodes = [test_data_organization] * n_organizations

    evaluation_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=rounds)
    evaluation_strategy.num_rounds = num_rounds

    for i in range(num_rounds):
        response = next(evaluation_strategy)
        if isinstance(rounds, int):
            true_rounds = range(1, num_rounds, rounds)
        else:
            true_rounds = rounds
        if i + 1 in true_rounds or (isinstance(rounds, int) and i + 1 == num_rounds):
            assert response
        else:
            assert not response

    with pytest.raises(StopIteration):
        response = next(evaluation_strategy)


@pytest.mark.parametrize(
    "rounds, e",
    [
        [0, ValueError],
        [-2, ValueError],
        [[], ValueError],
        [4.5, TypeError],
        [[4.5], TypeError],
        [[0, 2], ValueError],
        [[4, -1, 5], ValueError],
    ],
)
def test_rounds_edges(rounds, e):
    # tests that EvaluationStrategy raises appropriate error if the rounds
    # is not correct
    n_organizations = 3

    # mock the test data nodes
    test_data_organization = Mock(spec=TestDataNode)
    test_data_nodes = [test_data_organization] * n_organizations

    with pytest.raises(e):
        EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=rounds)


@pytest.mark.parametrize(
    "rounds, num_rounds, e",
    [
        [1, 0, ValueError],
        [5, 3, ValueError],
        [[1, 2, 3], 2, ValueError],
        [[10, 4, 6], 8, ValueError],
        [1, 1, None],
        [[1], 1, None],
    ],
)
def test_rounds_inconsitancy(rounds, num_rounds, e):
    # tests that consistency between rounds and num_rounds is
    # checked for and if inconsistency is found error is raised
    n_organizations = 3

    # mock the test data nodes
    test_data_organization = Mock(spec=TestDataNode)
    test_data_nodes = [test_data_organization] * n_organizations

    evaluation_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=rounds)

    if e is None:
        evaluation_strategy.num_rounds = num_rounds
    else:
        with pytest.raises(e):
            evaluation_strategy.num_rounds = num_rounds


@pytest.mark.parametrize(
    "test_data_nodes, e",
    [
        [[Mock(spec=TestDataNode)], None],
        [[1], TypeError],
    ],
)
def test_error_on_wrong_organization_instance(test_data_nodes, e):
    # test that only list of TestDataOrganizations are accepted as test_data_nodes
    if e is not None:
        with pytest.raises(e):
            EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=3)
    else:
        EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=3)


@pytest.mark.parametrize(
    "rounds, num_rounds, result",
    [[[1, 2], 5, [True, True, False, False, False, StopIteration]], [2, 4, [True, False, True, True, StopIteration]]],
)
def test_docstring_examples(rounds, num_rounds, result):
    """tests that the examples given in the docstring of EvaluationStrategy indeed give the correct result"""
    n_organizations = 3
    data_organization = Mock(spec=TestDataNode)
    data_organizations = [data_organization] * n_organizations

    evaluation_strategy = EvaluationStrategy(test_data_nodes=data_organizations, rounds=rounds)
    evaluation_strategy.num_rounds = num_rounds

    for answer in result:
        if answer is StopIteration:
            with pytest.raises(answer):
                next(evaluation_strategy)
        else:
            assert answer == next(evaluation_strategy)


@pytest.mark.parametrize("rounds", [1, 2, 4, 10, [1], [1, 4]])
def test_restart_rounds(rounds):
    # tests running a second time an evaluation strategy after calling restart_rounds
    # give the same results

    n_organizations = 3
    num_rounds = 10

    # mock the test organizations
    test_data_organization = Mock(spec=TestDataNode)
    test_data_nodes = [test_data_organization] * n_organizations

    evaluation_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=rounds)
    evaluation_strategy.num_rounds = num_rounds

    res1 = [next(evaluation_strategy) for i in range(num_rounds)]
    evaluation_strategy.restart_rounds()

    res2 = [next(evaluation_strategy) for i in range(num_rounds)]

    assert res1 == res2
