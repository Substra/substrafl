import logging

from substrafl.__version__ import __version__  # noqa
from substrafl.compute_plan_builder import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment
from substrafl.experiment import simulate_experiment
from substrafl.index_generator.np_index_generator import NpIndexGenerator
from substrafl.logger import set_logging_level

logger = logging.getLogger(name="substrafl")
if logger.getEffectiveLevel() > logging.INFO:
    set_logging_level(loglevel=logging.INFO)

__all__ = [
    "execute_experiment",
    "simulate_experiment",
    "set_logging_level",
    "NpIndexGenerator",
    "EvaluationStrategy",
    "ComputePlanBuilder",
]
