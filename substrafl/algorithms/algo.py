from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import List

from substrafl.remote.decorators import remote

if TYPE_CHECKING:
    from substrafl.strategies.schemas import StrategyName


class Algo(abc.ABC):
    """The base class to be inherited for substrafl algorithms."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    @abc.abstractmethod
    def model(self) -> Any:
        """Model exposed when the user downloads the model

        Returns:
            typing.Any: model
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        raise NotImplementedError

    # The abstractmethod decorator has no effect when combined with @remote_data
    # and @remote_data is there to indicate that it should be on the child class
    # train function
    # @remote_data
    @abc.abstractmethod
    def train(self, data_from_opener, shared_state: Any) -> Any:
        """Is executed for each TrainDataOrganizations.
        This functions takes the output of the ``get_data`` method from the opener, plus the shared state from the
        aggregator if there is one, and returns a shared state (state to send to the aggregator). Any variable that
        needs to be saved and updated from one round to another should be an attribute of ``self``
        (e.g. ``self._my_local_state_variable``), and be saved and loaded in the
        :py:func:`~substrafl.algorithms.algo.Algo.save_local_state` and
        :py:func:`~substrafl.algorithms.algo.Algo.load_local_state` functions.

        Args:
            data_from_opener (typing.Any): The output of the ``get_data`` method of the opener.
            shared_state (typing.Any): None for the first round of the computation graph
                then the returned object from the previous organization of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            typing.Any: The object passed to the next organization of the computation graph.
        """
        raise NotImplementedError

    # The abstractmethod decorator has no effect when combined with @remote_data
    # and @remote_data is there to indicate that it should be on the child class
    # predict function
    # @remote_data
    @abc.abstractmethod
    def predict(self, data_from_opener: Any, shared_state: Any = None) -> Any:
        """Is executed for each TestDataOrganizations. Compute the predictions from
        data outputed by the opener.

        Args:
            data_from_opener (typing.Any): The output of the ``get_data`` method of the opener.
            shared_state (typing.Any): None for the first round of the computation graph
                then the returned object from the previous organization of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            Any: The computed predictions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_local_state(self, path: Path) -> Any:
        """Executed at the beginning of each step of the computation graph so for each organization, at each step of
        the computation graph the previous local state can be retrieved.

        Args:
            path (pathlib.Path): The path where the previous local state has been saved.

        Raises:
            NotImplementedError

        Returns:
            typing.Any: The loaded element.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def save_local_state(self, path: Path) -> None:
        """Executed at the end of each step of the computation graph so for each organization,
        the local state can be saved.

        Args:
            path (pathlib.Path): The path where the previous local state has been saved.

        Raises:
            NotImplementedError

        Returns:
            None
        """
        raise NotImplementedError

    @remote
    def initialize(self, shared_states):
        """Empty function, useful to load the algo in the different organizations
        in order to perform an evaluation before any training step.

        Args:
            shared_states: Unused but enforced signature due to the @remote decorator.
        """
        return

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file.
        For child classes, override this function and add ``super.summary()``

        Example:

                .. code-block:: python

                    def summary(self):

                        summary = super().summary()
                        summary.update(
                            {
                                "attribute": self.attribute,
                                ...
                            }
                        )
                        return summary

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {"type": str(type(self))}
