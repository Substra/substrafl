import logging

from connectlib.__version__ import __version__  # noqa
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.experiment import execute_experiment
from connectlib.index_generator.np_index_generator import NpIndexGenerator
from connectlib.logger import set_logging_level
from connectlib.schemas import StrategyName

set_logging_level(loglevel=logging.INFO)

__all__ = ["execute_experiment", "set_logging_level", "NpIndexGenerator", "EvaluationStrategy", "StrategyName"]
