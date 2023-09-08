import inspect
import uuid
from collections.abc import Iterable
from string import printable
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import substra

from substrafl import exceptions
from substrafl.dependency import Dependency
from substrafl.nodes.node import InputIdentifiers
from substrafl.nodes.node import Node
from substrafl.nodes.node import OperationKey
from substrafl.nodes.node import OutputIdentifiers
from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.register import register_function
from substrafl.remote.register import register_metrics
from substrafl.remote.remote_struct import RemoteStruct


class TestDataNode(Node):
    """A node on which you will test your algorithm.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        test_data_sample_keys (List[str]): Substra data_sample_keys used for the training on this node
        metric_functions (Union[Dict[str, Callable], List[Callable], Callable]): Dictionary of Functions, Function or
            list of Functions that implement the different metrics. If a Dict is given, the keys will be used to
            register the result of the associated function. If a Function or a List is given, function.__name__
            will be used to store the result.
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        test_data_sample_keys: List[str],
        metric_functions: Union[Dict[str, Callable], List[Callable], Callable],
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys

        if isinstance(metric_functions, dict):
            for metric_id, metric_function in metric_functions.items():
                _check_metric_function(metric_function)
                _check_metric_identifier(metric_id)
            self.metric_functions = metric_functions

        elif isinstance(metric_functions, Iterable):
            self.metric_functions = {}
            for metric_function in metric_functions:
                _check_metric_function(metric_function)
                _check_metric_identifier(metric_function.__name__)
                if metric_function.__name__ in self.metric_functions:
                    raise exceptions.ExistingRegisteredMetricError
                self.metric_functions[metric_function.__name__] = metric_function

        elif callable(metric_functions):
            self.metric_functions = {}
            _check_metric_function(metric_functions)
            _check_metric_identifier(metric_functions.__name__)
            self.metric_functions[metric_functions.__name__] = metric_functions

        else:
            raise exceptions.MetricFunctionTypeError("Metric functions must be of type dictionary, list or callable")

        self.testtasks: List[Dict] = []
        self.predicttasks: List[Dict] = []

        super().__init__(organization_id)

    def update_states(
        self,
        traintask_id: str,
        operation: RemoteDataOperation,
        round_idx: Optional[int] = None,
    ):
        """Creating a test task based on the node characteristic.

        Args:
            traintask_id (str): The substra parent id
            operation (RemoteDataOperation): Automatically generated structure returned by
                the :py:func:`~substrafl.remote.decorators.remote_data` decorator. This allows to register an
                operation and execute it later on.
            round_idx (int): Used in case of learning compute plans. Round number, it starts at 1. Default to None.

        """

        predicttask_id = str(uuid.uuid4())

        data_inputs = [
            substra.schemas.InputRef(identifier=InputIdentifiers.opener, asset_key=self.data_manager_key)
        ] + [
            substra.schemas.InputRef(identifier=InputIdentifiers.datasamples, asset_key=data_sample)
            for data_sample in self.test_data_sample_keys
        ]
        predict_input = [
            substra.schemas.InputRef(
                identifier=InputIdentifiers.local,
                parent_task_key=traintask_id,
                parent_task_output_identifier=OutputIdentifiers.local,
            ),
        ]

        test_input = [
            substra.schemas.InputRef(
                identifier=InputIdentifiers.predictions,
                parent_task_key=predicttask_id,
                parent_task_output_identifier=OutputIdentifiers.predictions,
            )
        ]
        task_metadata = {"round_idx": str(round_idx)} if round_idx is not None else {}

        predicttask = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=predicttask_id,
            inputs=data_inputs + predict_input,
            outputs={
                OutputIdentifiers.predictions: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=False, authorized_ids=[self.organization_id]),
                    transient=True,
                )
            },
            metadata=task_metadata,
            worker=self.organization_id,
        ).dict()

        predicttask.pop("function_key")
        predicttask["remote_operation"] = operation.remote_struct
        self.predicttasks.append(predicttask)

        testtask = substra.schemas.ComputePlanTaskSpec(
            function_key=str(uuid.uuid4()),  # bogus function key
            task_id=str(uuid.uuid4()),
            inputs=data_inputs + test_input,
            outputs={
                metric_function_id: substra.schemas.ComputeTaskOutputSpec(
                    permissions=substra.schemas.Permissions(public=True, authorized_ids=[]),
                    transient=False,
                )
                for metric_function_id in self.metric_functions
            },
            metadata=task_metadata,
            worker=self.organization_id,
        ).dict()
        testtask.pop("function_key")
        testtask["remote_operation"] = operation.remote_struct
        self.testtasks.append(testtask)

    def register_test_operations(
        self,
        *,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ):
        for testtask in self.testtasks:
            remote_struct: RemoteStruct = testtask["remote_operation"]
            if remote_struct not in cache:
                # Register the metrics
                function_key = register_metrics(
                    client=client,
                    permissions=permissions,
                    dependencies=dependencies,
                    metric_functions=self.metric_functions,
                )
                testtask["function_key"] = function_key
                cache[remote_struct] = function_key
            else:
                function_key = cache[remote_struct]
                testtask["function_key"] = function_key

        return cache

    def register_predict_operations(
        self,
        *,
        client: substra.Client,
        permissions: substra.sdk.schemas.Permissions,
        cache: Dict[RemoteStruct, OperationKey],
        dependencies: Dependency,
    ) -> Dict[RemoteStruct, OperationKey]:
        """Find the functions from the parent traintask of each predicttask and submit it with a dockerfile
        specifying the ``predict`` method as ``--function-name`` to execute.

        Go through every operation in the predict function cache, submit it to substra and save
        `RemoteStruct : function_key` into the `cache` (where function_key is the returned function key by substra.)
        If two predicttasks depend on the same function, the function won't be added twice to substra as this method
        check if an function has already been submitted as a predicttask to substra before adding it.

        Args:
            client (substra.Client): Substra client for the organization.
            permissions (substra.sdk.schemas.Permissions): Permissions for the function.
            cache (typing.Dict[RemoteStruct, OperationKey]): Already registered function identifications. The key of
                each element is the RemoteStruct id (generated by substrafl) and the value is the key generated by
                substra.
            dependencies (Dependency): Function dependencies.

        Returns:
            typing.Dict[RemoteStruct, OperationKey]: updated cache
        """

        for predicttask in self.predicttasks:
            remote_struct: RemoteStruct = predicttask["remote_operation"]
            if remote_struct not in cache:
                # Register the predictask function
                function_key = register_function(
                    client=client,
                    remote_struct=remote_struct,
                    permissions=permissions,
                    dependencies=dependencies,
                    inputs=[
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.datasamples,
                            kind=substra.schemas.AssetKind.data_sample.value,
                            optional=False,
                            multiple=True,
                        ),
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.opener,
                            kind=substra.schemas.AssetKind.data_manager.value,
                            optional=False,
                            multiple=False,
                        ),
                        substra.schemas.FunctionInputSpec(
                            identifier=InputIdentifiers.local,
                            kind=substra.schemas.AssetKind.model.value,
                            optional=False,
                            multiple=False,
                        ),
                    ],
                    outputs=[
                        substra.schemas.FunctionOutputSpec(
                            identifier=OutputIdentifiers.predictions,
                            kind=substra.schemas.AssetKind.model.value,
                            multiple=False,
                        )
                    ],
                )
                predicttask["function_key"] = function_key
                cache[remote_struct] = function_key
            else:
                function_key = cache[remote_struct]
                predicttask["function_key"] = function_key

        return cache

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        summary.update(
            {
                "data_manager_key": self.data_manager_key,
                "data_sample_keys": self.test_data_sample_keys,
                "metric_functions": list(self.metric_functions.keys()),
            }
        )
        return summary


def _check_metric_function(metric_function: Callable) -> None:
    """Function to check the type and the signature of a given metric function.

    Args:
        metric_function (Callable): function to check.

    Raises:
        exceptions.MetricFunctionTypeError: metric_function must be of type "function"
        exceptions.MetricFunctionSignatureError: metric_function must ONLY contains
            datasamples and predictions_path as parameters
    """

    if not inspect.isfunction(metric_function):
        raise exceptions.MetricFunctionTypeError("Metric functions must be a callable or a list of callable")

    signature = inspect.signature(metric_function)
    parameters = signature.parameters

    if "datasamples" not in parameters:
        raise exceptions.MetricFunctionSignatureError(
            f"The metric_function: {metric_function.__name__} must contain datasamples as parameter."
        )
    elif "predictions_path" not in parameters:
        raise exceptions.MetricFunctionSignatureError(
            "The metric_function: {metric_function.__name__}  must contain predictions_path as parameter."
        )
    elif len(parameters) != 2:
        raise exceptions.MetricFunctionSignatureError(
            """The metric_function: {metric_function.__name__}  must ONLY contains datasamples and predictions_path as
            parameters."""
        )


def _check_metric_identifier(identifier: str) -> None:
    """Check if the identifier used to register the user given function does not interfere with the value internally
    used stored in the OutputIdentifiers enum.

    Args:
        identifier (str): identifier used for the registration of the metric function given by the user.

    Raises:
        exceptions.InvalidMetricIdentifierError: the identifier must not be in the OutputIdentifiers list used
        internally by SubstraFL.
    """

    if identifier in list(OutputIdentifiers):
        raise exceptions.InvalidMetricIdentifierError(
            f"A metric name or identifier cannot be in {[id.value for id in list(OutputIdentifiers)]}. \
            These values are used internally by SusbtraFL."
        )

    unauthorized_characters = set(identifier).difference(
        set(printable) - {"|"}
    )  # | is used in the backend as a separator for identifiers.
    if unauthorized_characters:
        raise exceptions.InvalidMetricIdentifierError(
            f"{unauthorized_characters} cannot be used to define a metric name."
        )

    if identifier == "":
        raise exceptions.InvalidMetricIdentifierError("A metric name cannot be an empty string.")
    elif len(identifier) > 36:  # Max length is the uuid4 length.
        raise exceptions.InvalidMetricIdentifierError("A metric name must be of length inferior to 36.")
