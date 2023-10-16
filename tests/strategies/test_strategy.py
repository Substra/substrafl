import uuid
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from substrafl import exceptions
<<<<<<< HEAD
from substrafl.nodes.schemas import OutputIdentifiers
=======
from substrafl.nodes.node import OutputIdentifiers
>>>>>>> b1ee596 (test: move metric function test to strategy and add evaluate test)


@pytest.mark.parametrize(
    "metric_function, expectation",
    [
        (
            lambda wrong_arg: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            lambda datasamples: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            lambda predictions: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            lambda datasamples, predictions, wrong_arg: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            "not a function",
            pytest.raises(exceptions.MetricFunctionTypeError),
        ),
        (
            [lambda datasamples, predictions: "any_str", lambda datasamples, predictions: "any_str"],
            pytest.raises(exceptions.ExistingRegisteredMetricError),
        ),
        (
            lambda datasamples, predictions: "any_str",
            does_not_raise(),
        ),
    ],
)
def test_wrong_metric_function(metric_function, expectation, dummy_strategy_class, dummy_algo_class):
    with expectation:
        dummy_strategy_class(
            algo=dummy_algo_class(),
            metric_functions=metric_function,
        )


def test_several_metric_function(dummy_strategy_class, dummy_algo_class):
    def f(datasamples, predictions):
        return

    def g(datasamples, predictions):
        return

    def h(datasamples, predictions):
        return

    expected_results = {"f": f, "g": g, "h": h}

    strat1 = dummy_strategy_class(
        algo=dummy_algo_class(),
        metric_functions=[f, g, h],
    )

    assert strat1.metric_functions == expected_results

    strat2 = dummy_strategy_class(
        algo=dummy_algo_class(),
        metric_functions={"f": f, "g": g, "h": h},
    )

    assert strat2.metric_functions == expected_results

    strat3 = dummy_strategy_class(
        algo=dummy_algo_class(),
        metric_functions=np.array([f, g, h]),
    )

    assert strat3.metric_functions == expected_results

    strat4 = dummy_strategy_class(
        algo=dummy_algo_class(),
        metric_functions={f, g, h},
    )

    assert strat4.metric_functions == expected_results


@pytest.mark.parametrize("identifier", OutputIdentifiers)
def test_metric_identifier_in_output_id(identifier, dummy_strategy_class, dummy_algo_class):
    with pytest.raises(exceptions.InvalidMetricIdentifierError):
        dummy_strategy_class(
            algo=dummy_algo_class(),
            metric_functions={identifier.value: lambda datasamples, predictions: "any_str"},
        )


@pytest.mark.parametrize(
    "metric_name, expectation",
    [
        ("hello world", does_not_raise()),
        ("hellô", pytest.raises(exceptions.InvalidMetricIdentifierError)),
        ("|hello", pytest.raises(exceptions.InvalidMetricIdentifierError)),
    ],
)
def test_metric_identifier_unauthorized_characters(metric_name, expectation, dummy_strategy_class, dummy_algo_class):
    with expectation:
        dummy_strategy_class(
            algo=dummy_algo_class(),
            metric_functions={metric_name: lambda datasamples, predictions: "any_str"},
        )


@pytest.mark.parametrize(
    "metric_name, expectation",
    [
        (str(uuid.uuid4()), does_not_raise()),
        ("", pytest.raises(exceptions.InvalidMetricIdentifierError)),
        (str(uuid.uuid4()) + "too_many_char", pytest.raises(exceptions.InvalidMetricIdentifierError)),
    ],
)
def test_metric_identifier_wrong_length(metric_name, expectation, dummy_strategy_class, dummy_algo_class):
    with expectation:
        dummy_strategy_class(
            algo=dummy_algo_class(),
            metric_functions={metric_name: lambda datasamples, predictions: "any_str"},
        )


@pytest.mark.parametrize(
    "strategy_name, expectation",
    [
        ("not_the_dummy_strategy", pytest.raises(exceptions.IncompatibleAlgoStrategyError)),
        ("dummy", does_not_raise()),
    ],
)
def test_match_algo_fedavg(strategy_name, dummy_strategy_class, dummy_algo_class, expectation):
    class MyAlgo(dummy_algo_class):
        @property
        def strategies(self):
            return [strategy_name]

    with expectation:
        dummy_strategy_class(algo=MyAlgo())


@pytest.mark.parametrize("data", [0, 2, 10])
def test_compute_score(data, dummy_strategy_class, dummy_algo_class):
    strat = dummy_strategy_class(
        algo=dummy_algo_class(),
        metric_functions={
            "add": lambda datasamples, predictions: datasamples + predictions["datasamples"],
            "mul": lambda datasamples, predictions: datasamples * predictions["datasamples"],
        },
    )

    res = strat.evaluate(datasamples=data, _skip=True)
    assert res["add"] == data + data
    assert res["mul"] == data * data
