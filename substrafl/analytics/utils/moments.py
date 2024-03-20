"""A module containing utils to compute high-order moments using Newton's formeanla."""

import numpy as np
import pandas as pd
from scipy.stats import moment


def compute_centered_moment(data, order):
    """Compute the centered moment of order k.

    Parameters
    ----------
    data : pd.DataFrame, np.array
        dataframe.
    order : int
        order of the moment.

    Returns
    -------
    pd.DataFrame, np.array
        Moment of order k.

    Raises
    ------
    NotImplementedError
        Raised if the data type is not Dataframe nor np.ndarray.
    """
    if isinstance(data, pd.Series):
        moment_value = moment(data, order, nan_policy="omit")
    elif isinstance(data, pd.DataFrame):
        moment_value = data.select_dtypes(include=np.number).apply(compute_centered_moment, order=order)
    elif isinstance(data, np.ndarray):
        moment_value = moment(data, order, nan_policy="omit")
    else:
        raise NotImplementedError("Only DataFrame or numpy array are currently handled.")
    return moment_value
