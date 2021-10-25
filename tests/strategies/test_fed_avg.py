import numpy as np
import pytest

from connectlib.strategies import FedAVG


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
        {"weights": np.ones((5, 10)), "n_samples": n_samples[0]},
        {"weights": np.zeros((5, 10)), "n_samples": n_samples[1]},
        {"weights": 2 * np.ones((5, 10)), "n_samples": n_samples[2]},
    ]

    MyFedAVG = FedAVG(num_rounds=0, num_updates=0, batch_size=0)
    averaged_states = MyFedAVG.avg_shared_states(shared_states, _skip=True)

    assert (results == averaged_states["weights"]).all()


@pytest.mark.parametrize(
    "shared_states",
    [
        [{"key1": np.array([1, 2, 3])}],
        [],
        [{}],
        [{"n_samples": 1}],
        [
            {"n_samples": 1, "weights": np.array([0, 1, 1])},
            {"n_samples": 1, "weights": [0, 1, 1]},
        ],
    ],
)
def test_avg_shared_states_no_n_samples_error(shared_states):
    # check if n_samples is not passed into avg_shared_states() error will be raised
    # check if no key is in the shared states error will be raised
    MyFedAVG = FedAVG(num_rounds=0, num_updates=0, batch_size=0)
    with pytest.raises(TypeError):
        MyFedAVG.avg_shared_states(shared_states, _skip=True)
