from contextlib import nullcontext as does_not_raise

import pytest

from substrafl import exceptions
from substrafl.nodes import TestDataNode


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
            lambda predictions_path: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            lambda datasamples, predictions_path, wrong_arg: "any_str",
            pytest.raises(exceptions.MetricFunctionSignatureError),
        ),
        (
            "not a function",
            pytest.raises(exceptions.MetricFunctionTypeError),
        ),
        (
            [lambda datasamples, predictions_path: "any_str", lambda datasamples, predictions_path: "any_str"],
            pytest.raises(exceptions.ExistingRegisteredMetricError),
        ),
        (
            lambda datasamples, predictions_path: "any_str",
            does_not_raise(),
        ),
    ],
)
def test_wrong_metric_function(metric_function, expectation):
    with expectation:
        TestDataNode(
            organization_id="fake_id",
            data_manager_key="fake_id",
            test_data_sample_keys=["fake_id"],
            metric_functions=metric_function,
        )


def test_several_metric_function():
    def f(datasamples, predictions_path):
        return

    def g(datasamples, predictions_path):
        return

    def h(datasamples, predictions_path):
        return

    expected_results = {"f": f, "g": g, "h": h}

    test_data_node_1 = TestDataNode(
        organization_id="fake_id",
        data_manager_key="fake_id",
        test_data_sample_keys=["fake_id"],
        metric_functions=[f, g, h],
    )

    assert test_data_node_1.metric_functions == expected_results

    test_data_node_2 = TestDataNode(
        organization_id="fake_id",
        data_manager_key="fake_id",
        test_data_sample_keys=["fake_id"],
        metric_functions={"f": f, "g": g, "h": h},
    )

    assert test_data_node_2.metric_functions == expected_results
