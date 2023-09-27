"""Define the substrafl stragegy to compute a mean."""
import numpy as np
from cancerlinq.strategies.base_strategy import AnalyticStrategy
from cancerlinq.utils.aggregation import aggregate_means

from substrafl.remote import remote
from substrafl.remote import remote_data


class StrategyMean(AnalyticStrategy):
    """Implement the substrafl AnalyticsStrategy for the mean.

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
        super().__init__(*args, **kwargs)
        self.local_computations = [self.local_mean]
        self.aggregations = [self.aggregate_mean]

    @remote_data
    def local_mean(self, datasamples, shared_state=None):  # pylint: disable=unused-argument
        """Compute the local mean.

        This method is transformed by the decorator to meet Substra API,
        and is executed in the training nodes. See build_graph.

        Parameters
        ----------
        datasamples : pd.DataFrame
            Dataframe returned by the opener.
        shared_state : None, optional
            Given by the aggregation node, here nothing, by default None.

        Returns
        -------
        dict
            Local results to be shared via shared_state to the aggregation node.
        """
        return {
            "mean": datasamples.mean(numeric_only=True, skipna=True),
            "n_samples": datasamples.select_dtypes(include=np.number).count(),
        }

    @remote
    def aggregate_mean(self, shared_states):
        """Compute the global mean given the local results.

        Parameters
        ----------
        shared_states : List
            List of results (local_mean, n_samples) from training nodes.

        Returns
        -------
        dict
            Global results to be shared with train nodes via shared_state.
        """
        tot_mean = aggregate_means([s["mean"] for s in shared_states], [s["n_samples"] for s in shared_states])
        return {"global_mean": tot_mean}
