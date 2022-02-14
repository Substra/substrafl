import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from connectlib.index_generator.np_index_generator import NpIndexGenerator


@pytest.mark.parametrize("batch_size", [0, 3, 5, 8])
def test_np_index_generator_drop_last(batch_size):
    # Check that if the last batch is dropped, all batches always have the same size, i.e. the batch size.
    nig = NpIndexGenerator(n_samples=10, batch_size=batch_size, shuffle=True, drop_last=True, seed=42)
    for _ in range(9):
        assert len(nig.__next__()) == batch_size, "All batches do not have the same size when drop_last is set to True."


@pytest.mark.parametrize("n_samples,batch_size", [(10, 3), (12, 5), (17, 11), (10, 5)])
def test_np_index_generator_keep_last(n_samples, batch_size):
    # If there are n_batch_per_epoch, and for multiple epochs,
    # check that the first n-1's are of size : batch size and that the nth one is of size n_samples%batch_size.

    nig = NpIndexGenerator(
        n_samples=n_samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        seed=42,
    )

    n_batch_per_epoch = nig._n_batch_per_epoch
    last_batch_size = n_samples % batch_size

    last_batch_size = last_batch_size if last_batch_size > 0 else batch_size

    # For 3 epoch, we are checking that the each batch has the expected size
    batch_number = 0
    while nig.n_epoch_generated < 3:
        # Last batch fo an epoch
        if batch_number % n_batch_per_epoch == n_batch_per_epoch - 1:
            assert len(nig.__next__()) == last_batch_size
        # Otherwise
        else:
            assert len(nig.__next__()) == batch_size

        batch_number += 1


@pytest.mark.parametrize(
    "n_samples,batch_size,drop_last",
    [(10, 3, True), (10, 5, True), (10, 3, False), (10, 5, False)],
)
def test_np_index_generator_100_epoch(n_samples, batch_size, drop_last):
    # Check that we can generate enough batches for 100 epoch.
    # The generator has also been tested locally to generates 10 000 epochs.

    nig = NpIndexGenerator(
        n_samples=n_samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        seed=42,
    )
    for _ in range(100 * nig._n_batch_per_epoch):
        assert len(nig.__next__()) > 0


def test_np_index_generator_n_samples_negative():
    # n_samples can't be negative

    with pytest.raises(ValueError):
        _ = NpIndexGenerator(
            n_samples=-4,
            batch_size=12,
            shuffle=True,
            drop_last=True,
            seed=42,
        )


def test_np_index_generator_batch_size_negative():
    # batch_size can't be negative

    with pytest.raises(ValueError):
        _ = NpIndexGenerator(
            n_samples=4,
            batch_size=-12,
            shuffle=True,
            drop_last=True,
            seed=42,
        )


@pytest.mark.parametrize(
    "seed,results",
    [
        (42, [np.array([2, 1, 0]), np.array([0, 2, 1]), np.array([1, 2, 0])]),
        (12, [np.array([1, 2, 0]), np.array([1, 2, 0]), np.array([1, 0, 2])]),
    ],
)
def test_np_index_generator_batch_shuffle(seed, results):
    # Check that the shuffling is properly seeded.
    nig = NpIndexGenerator(n_samples=3, batch_size=3, shuffle=True, drop_last=True, seed=seed)
    for result in results:
        assert np.array_equal(result, nig.__next__())


def test_np_index_generator_not_batch_shuffle():
    # Check that we can have not shuffled batches.
    nig = NpIndexGenerator(n_samples=3, batch_size=3, shuffle=False, drop_last=True, seed=42)
    res = np.arange(3)

    for _ in range(10):
        assert np.array_equal(res, nig.__next__())


@pytest.mark.parametrize(
    "n_samples,batch_size,drop_last",
    [(10, 3, True), (10, 5, True), (10, 3, False), (10, 5, False)],
)
def test_np_index_generator_check_epoch_consitency(n_samples, batch_size, drop_last):
    # Check that for each epoch, all indexes are not duplicated
    nig = NpIndexGenerator(
        n_samples=n_samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        seed=42,
    )

    n_epoch = nig.n_epoch_generated
    generated_epoch_indexes = np.array([])
    for _ in range(30):
        generated_epoch_indexes = np.concatenate((generated_epoch_indexes, nig.__next__()), axis=0)
        if n_epoch != nig.n_epoch_generated:

            # Check that there is no duplicates for the epoch
            assert len(set(generated_epoch_indexes)) == generated_epoch_indexes.shape[0]

            # Reinitialize for the next epoch
            n_epoch += 1
            generated_epoch_indexes = np.array([])


@pytest.mark.parametrize(
    "n_samples,batch_size,drop_last,expected",
    [(10, 3, True, 3), (10, 5, True, 2), (10, 3, False, 4), (10, 5, False, 2)],
)
def test_np_index_generator_count_batch_per_epoch(n_samples, batch_size, drop_last, expected):
    # Check that for each epoch, the right number of batch is generated
    nig = NpIndexGenerator(
        n_samples=n_samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        seed=42,
    )
    n_epoch = 0

    for i in range(1, 30):
        _ = nig.__next__()
        if i % expected == 0:
            n_epoch += 1

        assert nig.n_epoch_generated == n_epoch


def test_np_index_generator_statefulness(session_dir):
    # Check that after saving and loading, the batch generator still has the same outputs.
    indexer_path = Path(tempfile.mkdtemp(dir=session_dir)) / "my_indexer"

    nig = NpIndexGenerator(n_samples=5, batch_size=3, shuffle=True, drop_last=True, seed=42)

    # Artificial 10 first batch
    for _ in range(10):
        nig.__next__()

    # Dump
    with open(indexer_path, "wb") as f:
        pickle.dump(nig, f)
        f.close()

    # Load
    with open(indexer_path, "rb") as f:
        loaded_nig = pickle.load(f)
        f.close()

    # Check that both of them returns the same results
    for _ in range(20):
        assert np.array_equal(nig.__next__(), loaded_nig.__next__())
