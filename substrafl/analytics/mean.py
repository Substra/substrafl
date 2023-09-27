"""Define the substrafl stragegy to compute a mean."""

from typing import Any
from typing import List
from typing import Optional

import numpy as np

from substrafl.analytics.base_analytic import BaseAnalytic
from substrafl.remote import remote
from substrafl.remote import remote_data


class Mean(BaseAnalytic):
    """Implement the substrafl AnalyticsStrategy for the mean.

    This class is the first one of a new kind of strategies, AnalyticsStrategy, vs
    OptimizationStrategy which are the current strategies.
    """

    def __init__(
        self,
        differentially_private: bool = False,
        epsilon: Optional[float] = None,
        lower_bounds: Optional[List[float]] = None,
        upper_bounds: Optional[List[float]] = None,
        *args,
        **kwargs,
    ):
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
        differentially_private: bool
            If True, the mean is computed in a differential private way.
            Defaults to False.
        args: Any
            extra arguments
        kwargs: Any
            extra keyword arguments
        """
        super().__init__(differentially_private, epsilon, lower_bounds, upper_bounds, *args, **kwargs)
        if differentially_private and (epsilon is None or lower_bounds is None or upper_bounds is None):
            raise RuntimeError("epsilon, lower_bound and upper_bound must be set to use differential privacy")
        # TODO check that numpy seed is not set if differentially_private = True
        self.differentially_private = differentially_private
        self.epsilon = epsilon
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
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
        if not self.differentially_private:
            return {
                "mean": datasamples.mean(numeric_only=True, skipna=True),
                "n_samples": datasamples.select_dtypes(include=np.number).count(),
            }
        n_samples = datasamples.select_dtypes(include=np.number).count()
        # we don't want to rely on  broadcasting because we want a different noise value for each column
        bounded_mean = datasamples.clip(lower=self.lower_bounds, upper=self.upper_bounds).mean(
            numeric_only=True, skipna=True
        ) + np.random.laplace(scale=1 / (self.epsilon * n_samples), size=datasamples.shape[1])
        return {
            "mean": bounded_mean,
            "n_samples": n_samples,
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
