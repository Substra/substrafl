"""
Methods inherited from substratools.
"""
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypedDict

import substratools

from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.serializers.pickle_serializer import PickleSerializer
from substrafl.remote.serializers.serializer import Serializer


class RemoteMethod(substratools.AggregateAlgo):
    """Aggregate algo to register to Substra."""

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

    def aggregate(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.models: List[os.PathLike],
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict("outputs", {OutputIdentifiers.model: os.PathLike}),
    ) -> None:
        """Aggregation operation

        Args:
            inputs (typing.TypedDict): dict containing the list of models path loaded with `AggregateAlgo.load_model()`;
                the rank of the aggregate task.
            outputs (typing.TypedDict):dict containing the output model path to save the aggregated model.
        """
        models = []
        for m_path in inputs[InputIdentifiers.models]:
            models.append(self.load_model(m_path))
        outputs
        method_to_call = getattr(self.instance, self.method_name)
        next_shared_state = method_to_call(shared_states=models, _skip=True, **self.method_parameters)

        self.save_model(next_shared_state, outputs[OutputIdentifiers.model])
        # return next_shared_state

    def predict(self, inputs, outputs):
        """This predict method is required by substratools"""
        pass

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
    """Composite algo to register to Substra"""

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
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                InputIdentifiers.y: Any,
                InputIdentifiers.local: Optional[os.PathLike],
                InputIdentifiers.shared: Optional[os.PathLike],
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.local: os.PathLike,
                OutputIdentifiers.shared: os.PathLike,
            },
        ),  # outputs contains a dict where keys are identifiers and values are paths on disk
    ) -> None:
        """train method

        Args:
            inputs (typing.TypedDict): dict containing the training data samples loaded with `Opener.get_X()`;
                the training data samples labels loaded with `Opener.get_y()`;
                the head model loaded with `CompositeAlgo.load_head_model()` (may be None);
                the trunk model loaded with `CompositeAlgo.load_trunk_model()` (may be None);
                the rank of the training task.
            outputs (typing.TypedDict): dict containing the output head model path to save the head model;
                the output trunk model path to save the trunk model.
        """
        # head_model should be None only at initialization
        head_model_path = inputs.get(InputIdentifiers.local)
        trunk_model_path = inputs.get(InputIdentifiers.shared)

        if head_model_path is not None:
            instance = self.load_head_model(head_model_path)
        else:
            instance = self.instance

        trunk_model = self.load_trunk_model(trunk_model_path) if trunk_model_path else None
        X = inputs[InputIdentifiers.X]
        y = inputs[InputIdentifiers.y]

        method_to_call = instance.train
        next_shared_state = method_to_call(x=X, y=y, shared_state=trunk_model, _skip=True, **self.method_parameters)

        self.save_head_model(instance, outputs[OutputIdentifiers.local])
        self.save_trunk_model(next_shared_state, outputs[OutputIdentifiers.shared])

    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                InputIdentifiers.local: os.PathLike,
                InputIdentifiers.shared: os.PathLike,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.predictions: os.PathLike,
            },
        ),
    ) -> None:
        """predict function

        Args:
            inputs (typing.TypedDict): dict containing the testing data samples loaded with `Opener.get_X()`;
                the head model loaded with `CompositeAlgo.load_head_model()`;
                the trunk model loaded with `CompositeAlgo.load_trunk_model()`;
            outputs (typing.TypedDict): dict containing the output predictions path to save the predictions.
        """
        head_model_path = inputs.get(InputIdentifiers.local)
        assert head_model_path is not None, "head model is None. Possibly you did not train() before running predict()"
        instance = self.load_head_model(head_model_path)

        method_to_call = instance.predict
        trunk_model = self.load_trunk_model(inputs.get(InputIdentifiers.shared))
        X = inputs[InputIdentifiers.X]

        predictions_path = outputs[OutputIdentifiers.predictions]

        method_to_call(
            x=X, shared_state=trunk_model, predictions_path=predictions_path, _skip=True, **self.method_parameters
        )

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
