from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

from connectlib.remote import remote_data

Weights = Dict[str, np.ndarray]


class Algo:
    """The base class to be inherited for connectlib algorithms."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote_data
    @abstractmethod
    def train(self, x: Any, y: Any, shared_state: Weights) -> Weights:
        """Will be executed for each TrainDataNodes.

        Args:
            x (Any): The output of the `get_x` method of the opener.
            y (Any): The output of the `get_y` method of the opener.
            shared_state (Weights): None for the first round of the computation graph
            then the returned object from the pervious node of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            Weights: The object passed to the next node of the computation graph.
        """
        raise NotImplementedError

    @remote_data
    @abstractmethod
    def predict(self, x: Any, shared_state: Weights) -> Any:
        """Will be executed for each TestDataNodes. The returned object will be passed to the `save_predictions`
        function of the opener. The predictions are then loaded and used to calculate the metric.

        Args:
            x (Any): The output of the `get_x` method of the opener.
            shared_state (Weights): None for the first round of the computation graph
            then the returned object from the pervious node of the computation graph.

        Raises:
            NotImplementedError

        Returns:
            Any: The model prediction.
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def save(self, path: Path):
        """Executed at the end of each step of the computation graph so for each organization,
        the local state can be saved.

        Args:
            path (Path): The path where the previous local state has been saved.

        Raises:
            NotImplementedError
        """

        raise NotImplementedError

    def _preprocess(self, x: Any, y: Any = None) -> Union[Any, Tuple[Any, Any]]:
        """A 'glue' function to be used at the beginning of the `train` and `predict` methods to adapt the output
        of the opener to the expected type of the `train` and `predict` methods.

        Args:
            x (Any): Could be the input of the train methods hence the results of the `get_x` method of the opener.
            y (Any): Could be the input of the train methods hence the results of the `get_y` method of the opener.

        Returns:
            Union[Any, Tuple[Any, Any]]: (x, y) or y under an adapted format.
        """

        # For train
        if y is not None:
            return x, y

        # For predict
        return x

    def _postprocess(self, y_proba: Any) -> Any:
        """A 'glue' function to be used at the end of the `predict` method to adapt the output
        of the opener to the expected input of the metric `save_prediction` function.

        Args:
            y_proba (Any): Could be the output of the predict method.

        Returns:
            Any: y_proba under an adapted format.
        """
        return y_proba
