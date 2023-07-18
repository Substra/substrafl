from typing import Dict
from typing import List

from substra import Client

BenchmarkResult = Dict[str, Dict[str, float]]


class PerformanceError(Exception):
    """Generic error during benchmark"""


def get_performances(key: str, clients: List[Client]) -> BenchmarkResult:
    """
    Get benchmark performances.

    Args:
        key (str): Compute Plan key
        clients (typing.List[substra.Client]): Substra clients

    Returns:
        typing.Dict[str, typing.Dict[str, float]

    Raises:
        PerformanceError:
            - if the number of worker in Performances differ from the number of Substra clients
            - if one identifier is evaluated several times for one worker
    """
    raw_performances = clients[1].get_performances(key=key)

    n_workers, n_clients = len(set(raw_performances.worker)), len(clients)
    if n_workers != n_clients:
        raise PerformanceError(f"Found {n_workers} workers, expected {n_clients}")

    performances = {}
    raw_performances = list(zip(raw_performances.worker, raw_performances.identifier, raw_performances.performance))
    for org, metric, value in raw_performances:
        if org in performances and metric in performances[org]:
            raise PerformanceError(f"Metric {metric} evaluated several times on worker {org}")
        performances.setdefault(org, {})[metric] = float(value)

    return performances


def compare_performances(left: BenchmarkResult, right: BenchmarkResult, ndigits: int = 2) -> None:
    """
    Compare performances.

    Args:
        left (BenchmarkResult): First BenchmarkResult to compare
        right (BenchmarkResult): Second BenchmarkResult to compare
        ndigits (int): Tolerance applied on metric values during comparison

    Returns:
        None

    Raises
        PerformanceError:
            - If the metrics are not the same between left and right performances
            - If the metric results are not the same between left and right in respect to the tolerance applied
    """
    for li, ri in zip(left.values(), right.values()):
        if li.keys() != ri.keys():
            raise PerformanceError(f"Cannot compare non identical metrics: {tuple(li.keys())} and {tuple(ri.keys())}")
        if not all([round(li[k], ndigits) == round(ri[k], ndigits) for k in li.keys()]):
            err = (
                f"Benchmark results are not equals with tolerance of {1 * 10 ** (-1 * ndigits)}:"
                f"\nleft: {li}"
                f"\nright: {ri}"
            )
            raise PerformanceError(err)
    return
