import abc
from pathlib import Path
from typing import Any
from typing import List

from connectlib.schemas import StrategyName


class Algo(abc.ABC):
    """The base class to be inherited for connectlib algorithms."""

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
    def train(self, x: Any, y: Any, shared_state: Any) -> Any:
        """Is executed for each TrainDataOrganizations.
        This functions takes the x, y from the opener, plus the shared state from the aggregator if there is one,
        and returns a shared state (state to send to the aggregator). Any variable that needs to be saved and updated
        from one round to another should be an attribute of ``self`` (e.g. ``self._my_local_state_variable``), and be
        saved and loaded in the :py:func:`~connectlib.algorithms.algo.Algo.save` and
        :py:func:`~connectlib.algorithms.algo.Algo.load` functions.

        Args:
            x (typing.Any): The output of the ``get_x`` method of the opener.
            y (typing.Any): The output of the ``get_y`` method of the opener.
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
    def predict(self, x: Any, shared_state: Any) -> Any:
        """Is executed for each TestDataOrganizations. The returned object will be passed to the ``save_predictions``
        function of the opener. The predictions are then loaded and used to calculate the metric.

        Args:
            x (typing.Any): The output of the ``get_X`` method of the opener.
            shared_state (typing.Any): None for the first round of the computation graph
                then the returned object from the previous organization of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            typing.Any: The model prediction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: Path) -> Any:
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
    def save(self, path: Path):
        """Executed at the end of each step of the computation graph so for each organization,
        the local state can be saved.

        Args:
            path (pathlib.Path): The path where the previous local state has been saved.

        Raises:
            NotImplementedError
        """

        raise NotImplementedError

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
