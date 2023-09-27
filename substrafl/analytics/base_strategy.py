"""Define the abstract substrafl stragegy."""
import pickle as pk
from pathlib import Path
from typing import Any
from typing import Optional

from substrafl.compute_plan_builder import ComputePlanBuilder


class AnalyticStrategy(ComputePlanBuilder):
    """Implement the abstract substrafl AnalyticsStrategy.

    This class is the first one of a new kind of strategies, AnalyticsStrategy, vs
    OptimizationStrategy which are the current strategies.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the strategy.

        Define self.aggregation and self.local_computations,
        which are used by the build_graph method to
        define the compute plan in the following way:
         - run self.local_computations[0] on every train_nodes,
         store the results in list
         - run self.aggregations[0] on aggregation node
         using the list of previous results
         - run self.local_computations[1] on every train_nodes, ...
         And so on.

        Parameters
        ----------
        args: Any
            extra arguments
        kwargs: Any
            extra keyword arguments
        """
        super().__init__()
        self.statistics_result = None
        self.args = args
        self.kwargs = kwargs
        self.local_computations = []
        self.aggregations = []
        self.save_round = None

    # pylint: disable=unexpected-keyword-arg
    def build_compute_plan(
        self,
        train_data_nodes,
        aggregation_node,
        evaluation_strategy=None,
        num_rounds: int = 0,
        clean_models: Optional[bool] = False,
    ):  # pylint: disable=unused-argument
        """Build the computation graph of the strategy.

        The built graph will be stored by side effect in the
        given train_data_nodes, aggregation_nodes and
        evaluation_strategy. This function create a graph
        be first calling the initialization_round method
        of the strategy at round 0, and then call the
        perform_round method for each new round. If the
        current round is part of the evaluation strategy,
        the perform_predict method is called to complete the graph.

        Parameters
        ----------
        train_data_nodes : List[TrainDataNode]
            list of the train organizations
        aggregation_node : List[AggregationNode]
            aggregation node, necessary for
                centralized strategy, unused otherwise
        evaluation_strategy : Optional[EvaluationStrategy], optional
            Unused in AnalyticsStrategy, by default None
        num_rounds : int
            Number of round. Unused in AnalyticsStrategy. by default 0
        clean_models : Optional[bool], optional
            Clean the intermediary models on the Substra platform.
            Set it to False if you want to download or re-use intermediary models.
            This causes the disk space to fill quickly so should be
            set to True unless needed. by default False
        """
        assert len(self.local_computations) == len(self.aggregations)
        agg_shared_state = None
        step_idx = 0
        for step_idx, (local_computation_fct, aggregation_fct) in enumerate(
            zip(self.local_computations, self.aggregations)
        ):
            shared_states = []
            for node in train_data_nodes:
                # define composite tasks (do not submit yet)
                # for each composite task give description of
                # Algo instead of a key for an algo
                _, next_shared_state = node.update_states(
                    local_computation_fct(
                        node.data_sample_keys,
                        shared_state=agg_shared_state,
                        _algo_name=local_computation_fct.__doc__.split("\n")[0],
                    ),
                    local_state=None,
                    round_idx=step_idx,
                    authorized_ids=set([node.organization_id]),
                    aggregation_id=aggregation_node.organization_id,
                    clean_models=False,
                )
                # keep the states in a list: one/organization
                shared_states.append(next_shared_state)

            agg_shared_state = aggregation_node.update_states(
                aggregation_fct(
                    shared_states=shared_states,
                    _algo_name=aggregation_fct.__doc__.split("\n")[0],
                ),
                round_idx=step_idx,
                authorized_ids=set(train_data_node.organization_id for train_data_node in train_data_nodes),
                clean_models=False,
            )

    @property
    def num_round(self):
        """Return the number of round in the strategy.

        Returns
        -------
        int
            Number of round in the strategy.
        """
        return len(self.local_computations)

    def save_local_state(self, path: Path):
        """Save the object on the disk.

        Should be used only by the backend, to define the local_state.

        Parameters
        ----------
        path : Path
            Where to save the object.
        """
        with open(path, "wb") as file:
            pk.dump(self.statistics_result, file)

    def load_local_state(self, path: Path) -> Any:
        """Load the object from the disk.

        Should be used only by the backend, to define the local_state.

        Parameters
        ----------
        path : Path
            Where to find the object.

        Returns
        -------
        Any
            Previously saved instance.
        """
        with open(path, "rb") as file:
            self.statistics_result = pk.load(file)
        return self
