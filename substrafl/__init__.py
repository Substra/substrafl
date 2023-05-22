import logging

from substrafl.__version__ import __version__  # noqa
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment
from substrafl.index_generator.np_index_generator import NpIndexGenerator
from substrafl.load import download_algo_files
from substrafl.load import load_algo
from substrafl.logger import set_logging_level
from substrafl.schemas import StrategyName

set_logging_level(loglevel=logging.INFO)

__all__ = [
    "execute_experiment",
    "set_logging_level",
    "NpIndexGenerator",
    "EvaluationStrategy",
    "StrategyName",
    "load_algo",
    "download_algo_files",
]
