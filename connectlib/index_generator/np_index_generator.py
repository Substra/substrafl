import logging
from typing import Optional

import numpy as np

from connectlib.index_generator.base import BaseIndexGenerator

logger = logging.getLogger(__name__)


class NpIndexGenerator(BaseIndexGenerator):
    """An index based batch generator. It will return an array of size ``batch_size`` indexes.
    If ``batch_size`` is equal to zero, this will return an empty array.

    Each batch is generated and returned via the method :func:`next`:

    .. code-block:: python

        batch_generator = NpIndexGenerator(n_samples=10, batch_size=3)

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


    Attributes:
        n_samples (int): The number of samples in one epoch, i.e. the number of indexes you want to draw your batches
            from.
        batch_size (Optional[int]): The size of each batch. If set to None, the batch_size will be the number of
            samples.
        num_updates (int): The number of updates. After num_updates, the generator raises a StopIteration error.
            To reset it for the next round, use the `reset` function.
        shuffle (bool, Optional): Set to True to shuffle the indexes before each new epoch. Defaults to True.
        drop_last (bool, Optional): Set to True to drop the last incomplete batch, if the dataset size is not divisible
            by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to True.
        seed (int, Optional): The seed to set the randomness of the generator and have reproducible results.
            Defaults to 42.
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: Optional[int],
        num_updates: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        """Initialization of the generator."""

        # Initialization
        super().__init__(
            n_samples=n_samples,
            batch_size=batch_size,
            num_updates=num_updates,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

        # New properties
        self._n_batch_per_epoch: int = (
            (
                int(np.floor(self._n_samples / self._batch_size))
                if drop_last
                else int(np.ceil(self._n_samples / self._batch_size))
            )
            if self._batch_size != 0
            else 0
        )
        self._to_draw = np.arange(self._n_samples)
        if self._shuffle:
            self._to_draw = self._rng.permutation(self._to_draw)

    def __iter__(self):
        """Required methods for generators."""
        return self

    def __next__(self):
        """Generates the next batch and modifies the state of the class.

        At each call, this function will update the ``self._to_draw`` argument as ``self._to_draw`` contains
        the indexes that have not been already drawn during the current epoch. At the beginning of each epoch,
        all indexes will be shuffled if needed but not the first time as it has already been done at
        the initialization of the class.
        Each calls updates the ``self._counter`` argument by one, and each time it goes through an epoch, raises
        ``self._n_epoch_generated`` by one.

        Raises:
            StopIteration: when this function has been called ``self._num_updates`` times.

        Returns:
            np.ndarray: The batch indexes as a numpy array.
        """
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
