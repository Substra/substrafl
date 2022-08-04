import abc
import logging
from typing import Any
from typing import Optional

import numpy as np

from substrafl import exceptions

logger = logging.getLogger(__name__)


class BaseIndexGenerator(abc.ABC):
    """Base class for the index generator, must be
    subclassed.
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
            batch_size (typing.Optional[int]): The size of each batch. If set to None, the batch_size will be
                the number of samples.
            num_updates (int): Number of local updates at each round
            shuffle (bool, Optional): Shuffle the indexes or not. Defaults to True.
            drop_last (bool, Optional): Drop the last batch if its size is inferior to the batch size. Defaults to
                False.
            seed (int, Optional): Random seed. Defaults to 42.

        Raises:
            ValueError: if batch_size is negative
        """
        if batch_size is None:
            logger.info("None was passed as a batch size. It will be set to n_sample size.")
        elif batch_size < 0:
            raise ValueError(f"batch_size must be positive but {batch_size} was passed.")

        self._batch_size: Optional[int] = batch_size
        self._rng = np.random.default_rng(seed)
        self._shuffle: bool = shuffle
        self._drop_last: bool = drop_last
        self._num_updates: int = num_updates
        self._counter: int = 0
        self._n_epoch_generated: int = 0

        self._n_samples: Optional[int] = None

    @property
    def batch_size(self) -> int:
        """Number of samples used per batch.

        Returns:
            int: Batch size used by the index generator
        """
        return self._batch_size

    @property
    def counter(self) -> int:
        """Number of calls made to the iterator since the last counter reset.

        Returns:
            int: Number of calls made to the iterator
        """
        return self._counter

    @property
    def n_epoch_generated(self) -> int:
        """Number of epochs generated

        Returns:
            int: number of epochs generated
        """
        return self._n_epoch_generated

    @property
    def num_updates(self) -> int:
        """Number of batches generated between resets of the counter.

        Returns:
            int: number of updates
        """
        return self._num_updates

    @property
    def n_samples(self) -> Optional[int]:
        """Returns the number of samples in the dataset.

        Returns:
            typing.Optional[int]: number of samples in the dataset.
        """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, _n_samples: int):
        """Set the number of samples in the dataset.

        The indexes returned at each batch are between 0 and ``_n_samples - 1``.

        Args:
            _n_samples (int): Number of samples in the dataset

        Raises:
            ValueError: if _n_samples is negative.
        """
        # Validates n_samples
        if _n_samples < 0:
            raise ValueError(f"n_samples must be non negative but {_n_samples} was passed.")

        self._n_samples: int = _n_samples

        # Validates batch_size
        if self._batch_size is None:
            logger.info("None was passed as a batch size. It is set to n_sample size.")
            self._batch_size: int = self._n_samples
        elif self._batch_size > _n_samples:
            logger.info(
                (
                    "The batch size ({batch_size}) is greater than the number of samples: n_samples."
                    "This is not allowed. Batch_size is now updated to equal number of samples ({n_samples})"
                ).format(batch_size=self._batch_size, n_samples=_n_samples)
            )
            self._batch_size = self._n_samples

    def __iter__(self) -> "BaseIndexGenerator":
        """Required methods for generators, returns ``self``."""
        return self

    @abc.abstractclassmethod
    def __next__(self) -> Any:
        """Shall return a python object (batch_index) which
        is used for selecting each batch in the training loop method during training in this way :
        `x[batch_index], y[batch_index]`

        Shall also update self._counter and self._n_epoch_generated

        After ``num_updates`` call, raises StopIteration.

        Returns:
            Any: The batch indexes.
        """
        raise NotImplementedError

    def reset_counter(self):
        """Reset the counter to
        prepare for the next generation
        of batches.
        """
        self._counter = 0

    def check_num_updates(self):
        """Check if the counter is equal to ``num_updates``, which means that ``num_updates``
        batches have been generated since this instance has been created or the counter has been reset.

        Raises:
            exceptions.IndexGeneratorUpdateError: if the counter is different from ``num_updates``.
        """
        if self.counter != self._num_updates:
            raise exceptions.IndexGeneratorUpdateError(
                "The batch index generator has not been updated properly, it was called"
                f" {self.counter} times against {self._num_updates} expected."
            )
