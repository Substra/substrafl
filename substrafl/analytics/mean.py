"""Define the substrafl stragegy to compute a mean."""

from typing import Any
from typing import List

import numpy as np
from analytics.base_analytic import BaseAnalytic

from substrafl.remote import remote
from substrafl.remote import remote_data


class Mean(BaseAnalytic):
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

    def _aggregate_means(self, local_means: List[Any], n_local_samples: List[int]):
        """Aggregate local means.

        Aggregate the local means into a global mean by using the local number of samples.

        Parameters
        ----------
        local_means : List[Any]
            List of local means. Could be array, float, Series.
        n_local_samples : List[int]
            List of number of samples used for each local mean.

        Returns
        -------
        Any
            Aggregated mean. Same type of the local means
        """
        tot_samples = np.copy(n_local_samples[0])
        tot_mean = np.copy(local_means[0])
        for mean, n_sample in zip(local_means[1:], n_local_samples[1:]):
            mean = np.nan_to_num(mean, nan=0, copy=False)
            tot_mean *= tot_samples / (tot_samples + n_sample)
            tot_mean += mean * (n_sample / (tot_samples + n_sample))
            tot_samples += n_sample

        return tot_mean

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
        tot_mean = self._aggregate_means([s["mean"] for s in shared_states], [s["n_samples"] for s in shared_states])
        return {"global_mean": tot_mean}
