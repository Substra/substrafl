from pathlib import Path
from typing import Any
from typing import Dict
from typing import Type
from typing import TypedDict

import substratools as tools

from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.serializers.pickle_serializer import PickleSerializer
from substrafl.remote.serializers.serializer import Serializer


class RemoteMethod:
    """Methods to register to Substra"""

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

    def load_method_inputs(self, inputs: TypedDict, outputs: TypedDict):
        """Load the different parameters needed from the inputs and outputs dictionaries
        and increment a loaded_inputs dictionary depending on the InputIdentifiers or
        OutputIdentifiers of the parameter.

        Args:
            inputs (TypedDict):  dictionary containing the paths where to load the arguments for the method.
            outputs (TypedDict):  dictionary containing the paths where to save the output for the method.

        Returns:
            TypedDict: dictionary containing the kwargs of the method to call.
        """

        loaded_inputs = {}

        instance_path = inputs.get(InputIdentifiers.local)
        if instance_path is not None:
            self.instance = self.load_instance(instance_path)

        if InputIdentifiers.models in inputs:
            models = []
            for m_path in inputs[InputIdentifiers.models]:
                models.append(self.load_model(m_path))
            loaded_inputs["shared_states"] = models

        if InputIdentifiers.datasamples in inputs:
            loaded_inputs["datasamples"] = inputs[InputIdentifiers.datasamples]

        if InputIdentifiers.shared in inputs:
            loaded_inputs["shared_state"] = (
                self.load_model(inputs[InputIdentifiers.shared]) if inputs[InputIdentifiers.shared] else None
            )

        if InputIdentifiers.predictions in inputs:
            loaded_inputs["predictions_path"] = inputs[InputIdentifiers.predictions]

        if OutputIdentifiers.predictions in outputs:
            loaded_inputs["predictions_path"] = outputs[OutputIdentifiers.predictions]

        return loaded_inputs

    def save_method_output(self, method_output: Any, outputs: TypedDict):
        """Save the method output on the path given in outputs,
        depending on the value of the OutputIdentifiers.

        Args:
            method_output (Any): return value from the called method.
            outputs (TypedDict): dictionary containing the paths where to save the output for the method.
        """

        if OutputIdentifiers.local in outputs:
            self.save_instance(outputs[OutputIdentifiers.local])

        if OutputIdentifiers.model in outputs:
            self.save_model(method_output, outputs[OutputIdentifiers.model])

        elif OutputIdentifiers.shared in outputs:
            self.save_model(method_output, outputs[OutputIdentifiers.shared])

        elif OutputIdentifiers.performance in outputs:
            tools.save_performance(method_output, outputs[OutputIdentifiers.performance])

    def generic_function(
        self,
        inputs: TypedDict,
        outputs: TypedDict,  # outputs contains a dict where keys are identifiers and values are paths on disk
        task_properties: TypedDict,
    ) -> None:
        """Generic function to be registered and executed on the Substra platform using substra-tools.

        Args:
            inputs (TypedDict): dictionary containing the paths where to load the arguments for the method.
            outputs (TypedDict): dictionary containing the paths where to save the output of the method.
            task_properties (TypedDict): Unused.
        """

        method_inputs = self.load_method_inputs(inputs, outputs)
        method_to_call = getattr(self.instance, self.method_name)

        method_inputs["_skip"] = True

        method_output = method_to_call(
            **method_inputs,
            **self.method_parameters,
        )

        self.save_method_output(method_output, outputs)

    def load_model(self, path: str) -> Any:
        """Load the model from disk

        Args:
            path (str): path to the saved model

        Returns:
            Any: loaded model
        """
        return self.shared_state_serializer.load(Path(path))

    def save_model(self, model, path: str) -> None:
        """Save the model

        Args:
            model (Any): Model to save
            path (str): Path where to save the model
        """
        self.shared_state_serializer.save(model, Path(path))

    def load_instance(self, path: str) -> Any:
        """Load the instance from disk

        Args:
            path (str): path to the saved instance

        Returns:
            Any: loaded instance
        """
        return self.instance.load(Path(path))

    def save_instance(self, path: str) -> None:
        """Save the instance

        Args:
            model (Any): Instance to save
            path (str): Path where to save the instance
        """
        self.instance.save(Path(path))

    def register_substratools_function(self):
        """Register the function that can be accessed and executed by substratools."""

        tools.register(
            function=self.generic_function,
            function_name=self.method_name,
        )
