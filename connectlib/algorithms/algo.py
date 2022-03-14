import abc
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np


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
            Any: model
        """
        raise NotImplementedError

    # The abstractmethod decorator has no effect when combined with @remote_data
    # and @remote_data is there to indicate that it should be on the child class
    # train function
    # @remote_data
    @abc.abstractmethod
    def train(self, x: Any, y: Any, shared_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Will be executed for each TrainDataNodes.

        Args:
            x (Any): The output of the ``get_x`` method of the opener.
            y (Any): The output of the ``get_y`` method of the opener.
            shared_state (Dict[str, np.ndarray]): None for the first round of the computation graph
                then the returned object from the previous node of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            Dict[str, np.ndarray]: The object passed to the next node of the computation graph.
        """
        raise NotImplementedError

    # The abstractmethod decorator has no effect when combined with @remote_data
    # and @remote_data is there to indicate that it should be on the child class
    # predict function
    # @remote_data
    @abc.abstractmethod
    def predict(self, x: Any, shared_state: Dict[str, np.ndarray]) -> Any:
        """Will be executed for each TestDataNodes. The returned object will be passed to the ``save_predictions``
        function of the opener. The predictions are then loaded and used to calculate the metric.

        Args:
            x (Any): The output of the ``get_x`` method of the opener.
            shared_state (Dict[str, np.ndarray]): None for the first round of the computation graph
                then the returned object from the previous node of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            Any: The model prediction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: Path) -> Any:
        """Executed at the beginning of each step of the computation graph so for each organization, at each step of
        the computation graph the previous local state can be retrieved.

        Args:
            path (Path): The path where the previous local state has been saved.

        Raises:
            NotImplementedError

        Returns:
            Any: The loaded element.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: Path):
        """Executed at the end of each step of the computation graph so for each organization,
        the local state can be saved.

        Args:
            path (Path): The path where the previous local state has been saved.

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
