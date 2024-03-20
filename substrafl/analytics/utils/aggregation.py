"""Aggregation functions."""
from copy import copy

import pandas as pd
from scipy.special import binom


def aggregate_two_centered_moments(  # noqa: C901
    list_p_moments_a, list_p_moments_b, nb_samples_a, nb_samples_b, mean_a, mean_b
):
    r"""Aggregate two locally centered moments.

    Given two populations A and B, one can compute locally centered p-first moments
    for A and B. This function merges all those quantities to compute p-order
    centered moments for population A+B using the
      method of https://www.osti.gov/biblio/1426900, Section3.1, eq 3.1

    The only change we do is we use :math: \mu_p and not M_p to avoid overflow by
    multiplication of the number of samples.


    Final equation:

    .. math::
        \mu_p = \frac{N^A}{N}\mu_p^A +
            \frac{N^B}{N}\mu_p^B +
            \frac{N^A}{N} \left(-\frac{N^B}{N}\delta_{B,A}\right)^p +
            \frac{N^B}{N} \left(\frac{N^A}{N}\delta_{B,A}\right)^p +
            \sum_{k=1}^{p-2} \binom{p}{k}
            \left[\frac{N^A}{N} \mu_{p-k}^A \left(-\frac{N^B}{N}\delta_{B,A}\right)^k +
            \frac{N^B}{N} \mu_{p-k}^B \left(\frac{N^A}{N}\delta_{B,A}\right)^k\right]

    where N denotes the number of samples, and \delta_{B,A} = \bar x^B - \bar x^A
    Note that by definition \mu_0 = 1 and \mu_1 = 1.

    Parameters
    ----------
    list_p_moments_a : list[Union(float, pd.Series)]
        List of the p-order first moments for population A.
    list_p_moments_b : list[Union(float, pd.Series)]
        List of the p first moments for population B.
    nb_samples_a : Union(int, pd.Series)
        Number of samples in population A.
    nb_samples_b : Union(int, pd.Series)
        Number of samples in population B.
    mean_a : Union(float, pd.Series)
        Mean of the population A.
    mean_b : Union(float, pd.Series)
        Mean of the population B.

    Returns
    -------
    (list[Union(float, pd.Series)],  Union(int, pd.Series), Union(float, pd.Series)
        List of the p first moments for population A+B.
        Number of samples in population A+B.
        Mean of the population A+B.
    """
    assert len(list_p_moments_a) == len(list_p_moments_b)
    assert len(list_p_moments_a) > 2

    if isinstance(nb_samples_a, int):
        if nb_samples_a == 0:
            return list_p_moments_b, nb_samples_b, mean_b
        if nb_samples_b == 0:
            return list_p_moments_a, nb_samples_a, mean_a

    delta_b_a = mean_b - mean_a
    nb_samples = nb_samples_a + nb_samples_b
    rel_n_a = nb_samples_a / nb_samples
    rel_n_b = nb_samples_b / nb_samples
    mean = mean_a + rel_n_b * delta_b_a

    list_p_moments = [list_p_moments_a[0], list_p_moments_a[1]]

    for order in range(2, len(list_p_moments_a)):
        mu_p = rel_n_a * list_p_moments_a[order] + rel_n_b * list_p_moments_b[order]
        mu_p += rel_n_a * (-rel_n_b * delta_b_a) ** order + rel_n_b * (rel_n_a * delta_b_a) ** order
        for k in range(1, order - 1):
            mu_p += binom(order, k) * (
                rel_n_a * list_p_moments_a[order - k] * (-rel_n_b * delta_b_a) ** k
                + rel_n_b * list_p_moments_b[order - k] * (rel_n_a * delta_b_a) ** k
            )
        list_p_moments.append(copy(mu_p))
    if isinstance(nb_samples_a, pd.Series):
        # if one of the moment is Nan, just use the value of the other.
        zero_columns_a = nb_samples_a == 0
        if zero_columns_a.any():
            for order, p_moment in enumerate(list_p_moments):
                p_moment[zero_columns_a] = list_p_moments_b[order][zero_columns_a]
            mean[zero_columns_a] = mean_b[zero_columns_a]
        zero_columns_b = nb_samples_b == 0
        if zero_columns_b.any():
            for order, p_moment in enumerate(list_p_moments):
                p_moment[zero_columns_b] = list_p_moments_a[order][zero_columns_b]
            mean[zero_columns_b] = mean_a[zero_columns_b]
        # no need to change the nb of samples, because we added zero

    return list_p_moments, nb_samples, mean


def aggregate_centered_moments(list_list_p_moments, list_nb_samples, list_means):
    """Aggregate locally centered moments.

    Parameters
    ----------
    list_list_p_moments : list[list[Union(float, pd.Series)]]
        List of N list. Each element is a list of
        the p order first moments for population N.
    list_nb_samples : list[Union(int, pd.Series)]
        List of N list. Each element is number of
        samples in population N.
    list_means : list[Union(float, pd.Series)]
        List of N list. Each element is the
        mean of the population N.

    Returns
    -------
    (list[Union(float, pd.Series)],  Union(int, pd.Series), Union(float, pd.Series)
        List of the p first moments for pooled populations.
        Number of samples in  pooled populations.
        Mean of the  pooled populations.
    """
    list_p_moments_a = list_list_p_moments[0]
    nb_samples_a = list_nb_samples[0]
    mean_a = list_means[0]

    for list_p_moments_b, nb_samples_b, mean_b in zip(list_list_p_moments[1:], list_nb_samples[1:], list_means[1:]):
        (list_p_moments_a, nb_samples_a, mean_a) = aggregate_two_centered_moments(
            list_p_moments_a,
            list_p_moments_b,
            nb_samples_a,
            nb_samples_b,
            mean_a,
            mean_b,
        )

    return (list_p_moments_a, nb_samples_a, mean_a)
