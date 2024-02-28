import abc
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol
from substrafl.nodes import TrainDataNodeProtocol


class ComputePlanBuilder(abc.ABC):
    """Base compute plan builder to be inherited from SubstraFL compute plans."""

    def __init__(self, *args, **kwargs):
        """
        All child class arguments need to be passed to it through its ``args`` and ``kwargs``
        in order to use them when instantiating it as a RemoteStruct in each process.

        Example:

            .. code-block:: python

                class MyComputePlan(ComputePlanBuilder):
                    def __init__(self, custom_arg, my_custom_kwargs="value"):
                        super().__init__(custom_arg, my_custom_kwargs=my_custom_kwargs)
        """
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def build_compute_plan(
        self,
        train_data_nodes: Optional[List[TrainDataNodeProtocol]],
        aggregation_node: Optional[List[AggregationNodeProtocol]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: Optional[int],
        clean_models: Optional[bool] = True,
    ) -> None:
        """Build the compute plan to be executed. All arguments are optional and will be feed within the
        :func:`~substrafl.experiment.execute_experiment` function.

        Args:
            train_data_nodes (List[TrainDataNodeProtocol]): list of the train organizations
            aggregation_node (Optional[AggregationNodeProtocol]): aggregation node, necessary for
                centralized strategy, unused otherwise
            evaluation_strategy (Optional[EvaluationStrategy]): evaluation strategy to follow for testing models.
            num_rounds (int): Number of times to repeat the compute plan sub-graph (define in perform round). It is
                useful in recurring graphs, but can be ignored in other cases.
            clean_models (bool): Clean the intermediary models on the Substra platform. Set it to False
                if you want to download or re-use intermediary models. This causes the disk space to fill
                quickly so should be set to True unless needed. Defaults to True.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def save_local_state(self, path: Path) -> None:
        """Executed at the end of each step of the computation graph to save
        the local state locally on each organization.

        Args:
            path (pathlib.Path): The path where the previous local state has been saved.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def load_local_state(self, path: Path) -> Any:
        """Executed at the beginning of each step of the computation graph to load on each organization
        the previously saved local state.

        Args:
            path (pathlib.Path): The path where the previous local state has been saved.

        Returns:
            typing.Any: The loaded element.
        """
        pass
