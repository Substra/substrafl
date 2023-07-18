import datetime
from typing import Dict
from typing import Union

from substra.sdk.models import Performances

N_DIGITS_TOL: int = 4
RawPerformances = Dict[str, Dict[str, float]]


class PerformanceError(Exception):
    """Generic error during benchmark"""


class BenchmarkResults:
    """Benchmark metrics class."""

    def __init__(self, name: str, exec_time: float, performances: RawPerformances):
        """
        Constructor.

        Args:
            name (str): Experiment name
            exec_time (float): Experiment execution time
            performances (RawPerformances): Raw experiment performances
        """
        self.name = name
        self.exec_time = exec_time
        self.performances = self._cast_performances(performances=performances)

    def __str__(self) -> str:
        return f"{self.name}: {self.performances}"

    def __eq__(self, other: "BenchmarkResults") -> bool:
        return self.performances == other.performances

    @property
    def to_dict(self) -> Dict[str, Dict[str, Union[float, RawPerformances]]]:
        """Get BenchmarkMetrics as a dictionary."""
        return {self.name: {"exec_time": round(self.exec_time, 2), "performances": self.performances}}

    def assert_performance_equals(self, other: "BenchmarkResults") -> None:
        """Assert two BenchmarkResults.performances are equals."""
        lkeys, rkeys = self.performances.keys(), other.performances.keys()
        if lkeys != rkeys:
            raise PerformanceError(f"Found different metrics {tuple(rkeys)} and {tuple(lkeys)} \n{self}\n{other}")
        if not self == other:
            raise PerformanceError(f"Performances are not equals with 1e-{N_DIGITS_TOL} tol \n{self}\n{other}")

    @staticmethod
    def _cast_performances(performances: RawPerformances) -> Dict[str, float]:
        """
        Cast raw performances.

        Args:
            performances (RawPerformances): Raw experiment performances

        Returns:
            (Dict[str, float]): Performances
        """
        if not performances:
            raise TypeError(f"Found {performances}, expected {RawPerformances}")

        perf_list = list(performances.values())
        if not all([p == perf_list[0] for p in perf_list[1:]]):
            raise PerformanceError("Performances from all clients are expected to be equal.")

        return {key: round(val, N_DIGITS_TOL) for key, val in perf_list[0].items()}


def assert_expected_results(cl_metrics: BenchmarkResults, sa_metrics: BenchmarkResults, mode: str) -> None:
    """
    Assert benchmark results are the one expected, ie:
        - Assert substrafl and pure-torch performances are equals
        - Assert performances values are the one expected regarding the execution mode

    Args:
        cl_metrics (BenchmarkResults): Benchmark results for the substrafl experiment
        sa_metrics (BenchmarkResults): Benchmark results for the pure torch experiment
        mode:

    Returns:
        None

    Raises:
        PerformanceError if results are not the one expected.
    """
    cl_metrics.assert_performance_equals(other=sa_metrics)

    if mode == "subprocess":
        expected_performances = {"Accuracy": 0.5, "ROC AUC": 1.0}
    elif mode == "remote":
        expected_performances = {"Accuracy": 1.0, "ROC AUC": 1.0}
    else:
        return

    cl_metrics.assert_performance_equals(
        other=BenchmarkResults(name=f"expected_{mode}", exec_time=0, performances={"0": expected_performances})
    )

    return


def get_metrics_from_substra_performances(performances: Performances) -> BenchmarkResults:
    """
    Get benchmark performances.

    Args:
        performances (Performances): substra Performances

    Returns:
        typing.Dict[str, typing.Dict[str, float]

    Raises:
        PerformanceError:
            - if the number of worker in Performances differ from the number of Substra clients
            - if one identifier is evaluated several times for one worker
    """
    execution_time: datetime.timedelta = performances.compute_plan_end_date[0] - performances.compute_plan_start_date[0]
    execution_time: float = execution_time.total_seconds()

    workers, metrics, values = performances.worker, performances.identifier, performances.performance
    performances = {}

    for org, metric, value in list(zip(workers, metrics, values)):
        if org not in performances:
            performances[org] = {}
        if metric in performances[org]:
            raise PerformanceError(f"Metric {metric} evaluated several times on worker {org}")
        performances[org][metric] = float(value)

    return BenchmarkResults(name="substrafl", exec_time=execution_time, performances=performances)
