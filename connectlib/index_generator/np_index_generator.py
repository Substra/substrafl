import logging
from typing import Optional

import numpy as np

from connectlib import exceptions
from connectlib.index_generator.base import BaseIndexGenerator

logger = logging.getLogger(__name__)


class NpIndexGenerator(BaseIndexGenerator):
    """An index based batch generator. It returns an array of size ``batch_size`` indexes.
    If ``batch_size`` is equal to zero, this returns an empty array.

    Each batch is generated and returned via the method
    :py:func:`~connectlib.index_generator.np_index_generator.NpIndexGenerator.__next__`:

    .. code-block:: python

        batch_generator = NpIndexGenerator(batch_size=32, num_updates=100)
        batch_generator.n_samples = 10

        batch_1 = next(batch_generator)
        batch_2 = next(batch_generator)
        # ...
        batch_n = next(batch_generator)

    In that case, as the default seed is set, the results are deterministic:

    .. code-block:: python

            batch_1 = np.array([5, 6, 0])
            batch_12 = np.array([8, 4, 0])

    This class is stateful and can be saved and loaded with the pickle library:

    .. code-block:: python

        # Saving
        with open(indexer_path, "wb") as f:
            pickle.dump(batch_generator, f)
            f.close()

        # Loading
        with open(indexer_path, "rb") as f:
            loaded_batch_generator = pickle.load(f)
            f.close()
    """

    def __init__(
        self,
        batch_size: Optional[int],
        num_updates: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            batch_size (typing.Optional[int]): The size of each batch. If set to None, the batch_size is the
                number of samples.
            num_updates (int): The number of updates. After num_updates, the generator raises a StopIteration error.
                To reset it for the next round, use the
                :py:func:`~connectlib.index_generator.np_index_generator.NpIndexGenerator.reset_counter` function.
            shuffle (bool, Optional): Set to True to shuffle the indexes before each new epoch. Defaults to True.
            drop_last (bool, Optional): Set to True to drop the last incomplete batch, if the dataset size is not
                divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then
                the last batch is smaller. Defaults to False.
            seed (int, Optional): The seed to set the randomness of the generator and have reproducible results.
                Defaults to 42.
        """

        # Initialization
        super().__init__(
            batch_size=batch_size,
            num_updates=num_updates,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def __iter__(self):
        """Required methods for generators, returns ``self``."""
        return self

    def __next__(self):
        """Generates the next batch.

        At the start of each iteration through the whole dataset, if ``shuffle`` is True then all the indices are
        shuffled.
        If there are less elements left in the dataset than ``batch_size``, then if ``drop_last`` is False, a batch
        containing the remaining elements is returned, else the last batch is dropped and the batch is created from the
        whole dataset.
        Each calls updates the ``counter`` by one, and each time it goes through an epoch, increases
        ``n_epoch_generated`` by one.

        Raises:
            StopIteration: when this function has been called ``num_updates`` times.

        Returns:
            numpy.ndarray: The batch indexes as a numpy array.
        """
        if self._n_samples is None:
            raise exceptions.IndexGeneratorSampleNoneError(
                "Please set the number of samples using the" "n_samples function before iterating through the batches."
            )

        if self._counter == self._num_updates:
            raise StopIteration

        batch, self._to_draw = np.split(self._to_draw, [self._batch_size])

        # If there are not enough indexes left for a complete round we re initialize
        if (self._drop_last and self._to_draw.shape[0] < self._batch_size) or (self._to_draw.shape[0] == 0):
            self._to_draw = np.arange(self._n_samples)

            # we shuffle if needed
            if self._shuffle:
                self._to_draw = self._rng.permutation(self._to_draw)

            self._n_epoch_generated += 1

        self._counter += 1
        return batch

    @BaseIndexGenerator.n_samples.setter
    def n_samples(self, _n_samples: int):
        """Set the number of samples to draw from, then initialize
        the indexes to draw from when generating the batches.

        Args:
            _n_samples (int): number of samples in the dataset.
        """
        super(NpIndexGenerator, self.__class__).n_samples.fset(self, _n_samples)
        self._n_batch_per_epoch: int = (
            (
                int(np.floor(self._n_samples / self._batch_size))
                if self._drop_last
                else int(np.ceil(self._n_samples / self._batch_size))
            )
            if self._batch_size != 0
            else 0
        )
        self._to_draw = np.arange(self._n_samples)
        if self._shuffle:
            self._to_draw = self._rng.permutation(self._to_draw)
