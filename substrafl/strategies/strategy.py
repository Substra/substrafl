import inspect
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from string import printable
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

from tqdm.auto import tqdm

from substrafl import exceptions
from substrafl.algorithms.algo import Algo
from substrafl.compute_plan_builder import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol
from substrafl.nodes import TestDataNodeProtocol
from substrafl.nodes import TrainDataNodeProtocol
from substrafl.nodes.schemas import OutputIdentifiers
from substrafl.remote.decorators import remote_data
from substrafl.strategies.schemas import StrategyName

SharedState = TypeVar("SharedState")


class Strategy(ComputePlanBuilder):
    """Base strategy to be inherited from SubstraFL strategies."""

    def __init__(
        self,
        algo: Algo,
        metric_functions: Optional[Union[Dict[str, Callable], List[Callable], Callable]] = None,
        *args,
        **kwargs,
    ):
        """
        All child class arguments need to be passed to it through its ``args`` and ``kwargs``
        in order to use them when instantiating it as a RemoteStruct in each process.

        Example:

            .. code-block:: python

                class MyStrat(Strategy):
                    def __init__(self, algo, my_custom_arg):
                        super().__init__(algo=algo, my_custom_arg=my_custom_arg)

        Args:
            algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
            metric_functions (Optional[Union[Dict[str, Callable], List[Callable], Callable]]):
                list of Functions that implement the different metrics. If a Dict is given, the keys will be used to
                register the result of the associated function. If a Function or a List is given, function.__name__
                will be used to store the result.

        Raises:
            exceptions.IncompatibleAlgoStrategyError: Raise an error if the strategy name is not in ``algo.strategies``.
        """

        super().__init__(*args, algo=algo, metric_functions=metric_functions, **kwargs)

        self.metric_functions = _validate_metric_functions(metric_functions)
        self.algo = algo
        if self.name not in algo.strategies:
            raise exceptions.IncompatibleAlgoStrategyError(
                f"The algo {self.algo.__class__.__name__} is not compatible with the strategy "
                f"{self.__class__.__name__}, "
                f"named {self.name}. Check the algo strategies property: algo.strategies to see the list of compatible "
                "strategies."
            )

    @property
    @abstractmethod
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        raise NotImplementedError

    def initialization_round(
        self,
        *,
        train_data_nodes: List[TrainDataNodeProtocol],
        clean_models: bool,
        round_idx: Optional[int] = 0,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Call the initialize function of the algo on each train node.

        Args:
            train_data_nodes (typing.List[TrainDataNodeProtocol]): list of the train organizations
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            round_idx (typing.Optional[int]): index of the round. Defaults to 0.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization. Default to None
        """
        next_local_states = []

        for node in train_data_nodes:
            # define train tasks (do not submit yet)
            # for each train task give description of Algo instead of a key for an algo
            next_local_state = node.init_states(
                operation=self.algo.initialize(
                    _algo_name=f"Initializing with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
                authorized_ids=set([node.organization_id]) | additional_orgs_permissions,
                clean_models=clean_models,
            )
            next_local_states.append(next_local_state)
        self._local_states = next_local_states

    @abstractmethod
    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNodeProtocol],
        aggregation_node: Optional[AggregationNodeProtocol],
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Perform one round of the strategy

        Args:
            train_data_nodes (List[TrainDataNodeProtocol]): list of the train organizations
            aggregation_node (Optional[AggregationNodeProtocol]): aggregation node, necessary for
                centralized strategy, unused otherwise
            round_idx (int): index of the round
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        raise NotImplementedError

    @abstractmethod
    def perform_evaluation(
        self,
        test_data_nodes: List[TestDataNodeProtocol],
        train_data_nodes: List[TrainDataNodeProtocol],
        round_idx: int,
    ):
        """Perform the evaluation of the algo on each test nodes.
        Gets the model from a train organization and evaluate it on the
        test nodes.

        Args:
            test_data_nodes (typing.List[TestDataNodeProtocol]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNodeProtocol]): list of nodes on which the model has
                been trained
            round_idx (int): index of the round
        """
        raise NotImplementedError

    @remote_data
    def evaluate(self, data_from_opener: Any, shared_state: Any = None) -> Dict[str, float]:
        """Is executed for each TestDataOrganizations.

        Args:
            data_from_opener (typing.Any): The output of the ``get_data`` method of the opener.
            shared_state (typing.Any): None for the first round of the computation graph
                then the returned object from the previous organization of the computation graph.

        Returns:
            Dict[str, float]: keys of the dict are the metric name, and values are the computed
                performances.
        """
        predictions = self.algo.predict(data_from_opener, shared_state)
        return {
            metric_function_id: metric_function(data_from_opener=data_from_opener, predictions=predictions)
            for metric_function_id, metric_function in self.metric_functions.items()
        }

    def build_compute_plan(
        self,
        train_data_nodes: List[TrainDataNodeProtocol],
        aggregation_node: Optional[List[AggregationNodeProtocol]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: int,
        clean_models: Optional[bool] = True,
    ) -> None:
        """Build the compute plan of the strategy.
        The built graph will be stored by side effect in the given train_data_nodes,
        aggregation_nodes and evaluation_strategy.
        This function create a graph be first calling the initialization_round method of the strategy
        at round 0, and then call the perform_round method for each new round.
        If the current round is part of the evaluation strategy, the perform_evaluation method is
        called to complete the graph.

        Args:
            train_data_nodes (List[TrainDataNodeProtocol]): list of the train organizations
            aggregation_node (Optional[AggregationNodeProtocol]): aggregation node, necessary for
                centralized strategy, unused otherwise
            evaluation_strategy (Optional[EvaluationStrategy]): evaluation strategy to follow for testing models.
            num_rounds (int): Number of times to repeat the compute plan sub-graph (define in perform round).
            clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
                if you want to download or re-use intermediary models. This causes the disk space to fill
                quickly so should be set to True unless needed. Defaults to True.

        Returns:
            None
        """
        additional_orgs_permissions = (
            evaluation_strategy.test_data_nodes_org_ids if evaluation_strategy is not None else set()
        )
        with tqdm(
            total=num_rounds,
            desc="Rounds progress",
        ) as progress_bar:
            # create computation graph.
            for round_idx in range(0, num_rounds + 1):
                if round_idx == 0:
                    self.initialization_round(
                        train_data_nodes=train_data_nodes,
                        additional_orgs_permissions=additional_orgs_permissions,
                        clean_models=clean_models,
                    )
                else:
                    if round_idx == num_rounds:
                        clean_models = False  # Enforce to keep at least the outputs of the last round.

                    self.perform_round(
                        train_data_nodes=train_data_nodes,
                        aggregation_node=aggregation_node,
                        additional_orgs_permissions=additional_orgs_permissions,
                        round_idx=round_idx,
                        clean_models=clean_models,
                    )

                    progress_bar.update()

                if evaluation_strategy is not None and next(evaluation_strategy):
                    self.perform_evaluation(
                        train_data_nodes=train_data_nodes,
                        test_data_nodes=evaluation_strategy.test_data_nodes,
                        round_idx=round_idx,
                    )

    def save_local_state(self, path: Path) -> None:
        self.algo.save_local_state(path)

    def load_local_state(self, path: Path) -> Any:
        self.algo = self.algo.load_local_state(path)
        return self


def _validate_metric_functions(metric_functions):
    if metric_functions is None:
        return {}

    elif isinstance(metric_functions, dict):
        for metric_id, metric_function in metric_functions.items():
            _check_metric_function(metric_function)
            _check_metric_identifier(metric_id)
        return metric_functions

    elif isinstance(metric_functions, Iterable):
        metric_functions_dict = {}
        for metric_function in metric_functions:
            _check_metric_function(metric_function)
            _check_metric_identifier(metric_function.__name__)
            if metric_function.__name__ in metric_functions_dict:
                raise exceptions.ExistingRegisteredMetricError
            metric_functions_dict[metric_function.__name__] = metric_function
        return metric_functions_dict

    elif callable(metric_functions):
        metric_functions_dict = {}
        _check_metric_function(metric_functions)
        _check_metric_identifier(metric_functions.__name__)
        metric_functions_dict[metric_functions.__name__] = metric_functions
        return metric_functions_dict

    else:
        raise exceptions.MetricFunctionTypeError("Metric functions must be of type dictionary, list or callable")


def _check_metric_function(metric_function: Callable) -> None:
    """Function to check the type and the signature of a given metric function.

    Args:
        metric_function (Callable): function to check.

    Raises:
        exceptions.MetricFunctionTypeError: metric_function must be of type "function"
        exceptions.MetricFunctionSignatureError: metric_function must ONLY contains
            data_from_opener and predictions as parameters
    """

    if not inspect.isfunction(metric_function):
        raise exceptions.MetricFunctionTypeError("Metric functions must be a callable or a list of callable")

    signature = inspect.signature(metric_function)
    parameters = signature.parameters

    if "data_from_opener" not in parameters:
        raise exceptions.MetricFunctionSignatureError(
            f"The metric_function: {metric_function.__name__} must contain data_from_opener as parameter."
        )
    elif "predictions" not in parameters:
        raise exceptions.MetricFunctionSignatureError(
            "The metric_function: {metric_function.__name__}  must contain predictions as parameter."
        )
    elif len(parameters) != 2:
        raise exceptions.MetricFunctionSignatureError(
            """The metric_function: {metric_function.__name__}  must ONLY contains data_from_opener and predictions as
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
