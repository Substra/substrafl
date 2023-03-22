from contextlib import nullcontext as does_not_raise

import pytest

from substrafl import exceptions


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
