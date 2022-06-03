"""
Methods inherited from connect-tools.
"""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type

import substratools

from connectlib.remote.serializers.pickle_serializer import PickleSerializer
from connectlib.remote.serializers.serializer import Serializer


class RemoteMethod(substratools.AggregateAlgo):
    """Aggregate algo to register to Connect."""

    def __init__(
        self,
        instance,
        method_name: str,
        method_parameters: Dict,
        shared_state_serializer: Type[Serializer] = PickleSerializer,
    ):
        self.instance = instance

        self.method_name = method_name
        self.method_parameters = method_parameters

        self.shared_state_serializer = shared_state_serializer

    def aggregate(self, models, rank) -> Any:
        """Aggregation operation

        Args:
            models (list): list of in models to aggregate
            rank (int): rank in the CP

        Returns:
            Any: aggregated model
        """
        method_to_call = getattr(self.instance, self.method_name)
        next_shared_state = method_to_call(shared_states=models, _skip=True, **self.method_parameters)

        return next_shared_state

    def predict(self, X, model):
        """This predict method is required by substratools"""
        return

    def load_model(self, path: str) -> Any:
        """Load the model from disk, may be a in model of the aggregate
        or the out aggregated model.

        Args:
            path (str): Path where the model is saved

        Returns:
            Any: Loaded model
        """
        return self.shared_state_serializer.load(Path(path))

    def save_model(self, model, path: str):
        self.shared_state_serializer.save(model, Path(path))


class RemoteDataMethod(substratools.CompositeAlgo):
    """Composite algo to register to Connect"""

    def __init__(
        self,
        instance,
        method_name: str,
        method_parameters: Dict,
        shared_state_serializer: Type[Serializer] = PickleSerializer,
    ):
        self.instance = instance

        self.method_name = method_name
        self.method_parameters = method_parameters

        self.shared_state_serializer = shared_state_serializer

    def train(
        self,
        X: Any,
        y: Any,
        head_model: Any,  # instance of algo
        trunk_model: Any,  # shared state
        rank: int,
    ) -> Tuple:
        """train method

        Args:
            X (typing.Any): X as returned by the opener get_X
            y (typing.Any): y as returned by the opener get_y
            head_model (typing.Any): incoming local state
            trunk_model (typing.Any): incoming shared state
            rank (int): rank in the CP, by order of execution

        Returns:
            Tuple: output head_model, trunk_model
        """
        # head_model should be None only at initialization
        if head_model is not None:
            instance = head_model
        else:
            instance = self.instance

        method_to_call = instance.train
        next_shared_state = method_to_call(x=X, y=y, shared_state=trunk_model, _skip=True, **self.method_parameters)

        return instance, next_shared_state

    def predict(self, X: Any, head_model: Any, trunk_model: Any) -> Any:
        """predict function

        Args:
            X (typing.Any): X as returned by the opener get_X
            head_model (typing.Any): incoming local state
            trunk_model (typing.Any): incoming shared state

        Returns:
            Any: predictions
        """
        assert head_model is not None, "head model is None. Possibly you did not train() before running predict()"
        instance = head_model

        method_to_call = instance.predict
        predictions = method_to_call(x=X, shared_state=trunk_model, _skip=True, **self.method_parameters)

        return predictions

    def load_trunk_model(self, path: str) -> Any:
        """Load the trunk model from disk

        Args:
            path (str): path to the saved trunk model

        Returns:
            Any: loaded trunk model
        """
        return self.shared_state_serializer.load(Path(path))

    def save_trunk_model(self, model, path: str) -> None:
        """Save the trunk model

        Args:
            model (typing.Any): Trunk model to save
            path (str): Path where to save the model
        """
        self.shared_state_serializer.save(model, Path(path))

    def load_head_model(self, path: str) -> Any:
        """Load the head model from disk

        Args:
            path (str): path to the saved head model

        Returns:
            Any: loaded head model
        """
        return self.instance.load(Path(path))

    def save_head_model(self, model, path: str) -> None:
        """Save the head model

        Args:
            model (typing.Any): Head model to save
            path (str): Path where to save the model
        """
        model.save(Path(path))
