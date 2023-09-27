"""Define the substrafl stragegy to compute a variance."""
import numpy as np

from substrafl.analytics.base_analytic import BaseAnalytic
from substrafl.analytics.utils.aggregation import aggregate_centered_moments
from substrafl.analytics.utils.moments import compute_centered_moment
from substrafl.remote import remote
from substrafl.remote import remote_data


class StrategyVariance(BaseAnalytic):
    """Implement the substrafl AnalyticsStrategy for the variance.

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
        self.local_computations = [self.local_moments_1_and_2]
        self.aggregations = [self.aggregate_variance]

    @remote_data
    def local_moments_1_and_2(self, datasamples, shared_state=None):  # pylint: disable=unused-argument
        """Compute the local mean, and the 2nd order moment.

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
            "moment0": compute_centered_moment(datasamples, 0),
            "moment1": compute_centered_moment(datasamples, 1),
            "moment2": compute_centered_moment(datasamples, 2),
            "mean": datasamples.mean(numeric_only=True, skipna=True),
            "n_samples": datasamples.select_dtypes(include=np.number).count(),
        }

    @remote
    def aggregate_variance(self, shared_states):
        """Compute the global mean and variance given the local results.

        Parameters
        ----------
        shared_states : List
            List of results (local_m1, local_m2, n_samples) from training nodes.

        Returns
        -------
        dict
            Global results to be shared with train nodes via shared_state.
        """
        (global_centered_moments, global_n_samples, global_mean,) = aggregate_centered_moments(
            [[shared_state[f"moment{k}"] for k in range(0, 2 + 1)] for shared_state in shared_states],
            [shared_state["n_samples"] for shared_state in shared_states],
            [shared_state["mean"] for shared_state in shared_states],
        )
        return {
            "global_mean": global_mean,
            "global_variance": global_centered_moments[2],
            "global_n_samples": global_n_samples,
        }
