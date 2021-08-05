import numpy as np
import pytest

from connectlib.strategies import FedAVG


@pytest.mark.parametrize(
    "len_data, num_updates, batch_size",
    [
        (5, 3, 4),  # len_data < (num_updates * batch_size)
        (15, 3, 5),  # len_data == (num_updates * batch_size)
        (20, 3, 4),  # len_data > (num_updates * batch_size)
        (3, 4, 4),  # len_data < batch_size
    ],
)
@pytest.mark.parametrize("num_rounds", [0, 1, 2, 10])
@pytest.mark.parametrize("drop_last", [True, False])
def test_data_indexer(len_data, num_updates, batch_size, num_rounds, drop_last):
    """Test if given different parameters we receive the expected indices"""

    if drop_last:
        expect_last_batch = batch_size
    else:
        expect_last_batch = len_data % batch_size
        if not expect_last_batch:
            expect_last_batch = batch_size

    strategy = FedAVG(num_rounds=num_rounds, num_updates=num_updates, batch_size=batch_size)

    x = np.zeros(len_data)
    y = np.ones(len_data)

    if batch_size > len_data and drop_last:
        # ensure that we are not allowing the larger batch size
        # than a data size if drop_last is set to True
        with pytest.raises(ValueError) as exc:
            indices = strategy.data_indexer(
                _skip=True,
                x=x,
                y=y,
                num_rounds=num_rounds,
                num_updates=num_updates,
                batch_size=batch_size,
                drop_last=drop_last,
            )
        assert "batch_size cannot be larger " in str(exc.value)
    else:
        indices = strategy.data_indexer(
            _skip=True,
            x=x,
            y=y,
            num_rounds=num_rounds,
            num_updates=num_updates,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        assert type(indices["minibatch_indices"]) == list
        assert len(indices["minibatch_indices"]) == num_rounds

        # clever way to flatten the list
        flat_ind = sum(sum(indices["minibatch_indices"], []), [])
        # ensure that all the first indices are unique as expected
        if len_data <= num_rounds * num_updates * batch_size:
            if not drop_last:
                # ensure that we used all the indices during the first round
                assert len(np.unique(flat_ind[: len_data + 1])) == len_data
            else:
                # ensure that we used unique indices during the num_updates * batch_size
                len_unique = len_data - (len_data % batch_size)
                assert len(np.unique(flat_ind[: len_unique + 1])) == len_unique

        for round in indices["minibatch_indices"]:
            assert type(round) == list
            assert len(round) == num_updates
            for update in round:
                # make sure it is a list
                assert type(update) == list
                # make sure the len of batch is either as set or
                # as leftover indices if drop_last is False
                assert len(update) in [min(batch_size, len_data), expect_last_batch]
                for idx in update:
                    assert type(idx) is int
