import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NpIndexGenerator:
    """An infinite index based batch generator. It will return an array of size `batch_size` indexes.
    If batch size is equal to zero, this will return an empty array.
    Each batch is generated and returned via the class method `self.__next__()`. E.g. :

        batch_generator = NpIndexGenerator(n_samples=10, batch_size=3)

        batch_1 = NpIndexGenerator.__next__()
        batch_2 = NpIndexGenerator.__next__()
        ...
        batch_n = NpIndexGenerator.__next__()

    In that case, as the default seed is set, the results are deterministic :
        batch_1 = np.array([5, 6, 0])
        batch_12 = np.array([8, 4, 0])

    This class is stateful and can be saved and loaded with the pickle library :

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
        shuffle (bool, optional): Set to True to shuffle the indexes before each new epoch. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size is not divisible
            by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to True.
        seed (int, optional): The seed to set the randomness of the generator and have reproducible results.
            Defaults to 42.
        n_epoch_generated (int): The number of epoch already generated. Automatically initialize to 0 then incremented
            for each epoch.
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: Optional[int],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        """Initialization of the generator."""

        # Initialization
        self._shuffle: bool = shuffle
        self._drop_last: bool = drop_last
        self._seed: int = seed

        # Validates n_samples
        if n_samples < 0:
            raise ValueError(f"n_samples must be non negative but {n_samples} was passed.")

        self._n_samples: int = n_samples

        # Validates batch_size
        if batch_size is None:
            logger.info("None was passed as a batch size. It will be set to n_sample size.")
            self._batch_size: int = self._n_samples

        elif batch_size < 0:
            raise ValueError(f"batch_size must be non negative but {batch_size} was passed.")

        elif batch_size > n_samples:
            logger.info(
                (
                    "The batch size ({batch_size}) is greater than the number of samples: n_samples."
                    "This is not allowed. Batch_size is now updated to equal number of samples ({n_samples})"
                ).format(batch_size=batch_size, n_samples=n_samples)
            )
            self._batch_size = self._n_samples

        else:
            self._batch_size = batch_size

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
        self._rng = np.random.default_rng(self._seed)
        self._to_draw = np.arange(self._n_samples)
        if self._shuffle:
            self._to_draw = self._rng.permutation(self._to_draw)

        self.n_epoch_generated: int = 0

    def __iter__(self):
        """Required methods for generators."""
        return self

    def __next__(self):
        """Generates the next batch and modifies the state of the class. At each call, this function will update the
        `self._to_draw` argument as `self._to_draw` contains the indexes that have not been already drawn during the
        current epoch. At the beginning of each epoch, all indexes will be shuffled if needed but not the first time
        as it has already been done at the initialization of the class.

        Returns:
            np.ndarray: The batch indexes as a numpy array.
        """

        batch, self._to_draw = np.split(self._to_draw, [self._batch_size])

        # If there are not enough indexes left for a complete round we re initialize
        if (self._drop_last and self._to_draw.shape[0] < self._batch_size) or (self._to_draw.shape[0] == 0):
            self._to_draw = np.arange(self._n_samples)
            self.n_epoch_generated += 1

            # we shuffle if needed
            if self._shuffle:
                self._to_draw = self._rng.permutation(self._to_draw)

        return batch
