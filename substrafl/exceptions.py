class ArgumentConflictError(ValueError):
    """Incompatible value in the given arguments."""


class BatchSizeNotFoundError(Exception):
    """No batch size found."""


class SubstraToolsDeprecationWarning(DeprecationWarning):
    """The substratools version used is deprecated."""


class UnsupportedPythonVersionError(Exception):
    """The Python version used is not supported by Substra."""


class UnsupportedClientBackendTypeError(Exception):
    """The given client backend type is not supported by the function."""


class InvalidUserModuleError(Exception):
    """The local folder passed by the user as a dependency is not a valid Python module."""


class InvalidDependenciesError(Exception):
    """The set of constraints given on dependencies cannot be solved or is otherwise invalid (wrong package name)."""


class CriterionReductionError(Exception):
    """The criterion reduction must be set to 'mean' to use the Newton-Raphson strategy."""


class DampingFactorValueError(Exception):
    """The damping factor must be greater than 0 and less than or equal to 1"""


class DatasetSignatureError(Exception):
    """The __init__() function on the given torch Dataset must have data_from_opener and is_inference as parameters."""


class DatasetTypeError(Exception):
    """The given torch Dataset should be a torch.utils.data.Dataset object and not an instance of it."""


class EmptySharedStatesError(Exception):
    """The shared_states is empty. Ensure that the train method of the algorithm returns a
    StrategySharedState object."""


class ExistingRegisteredMetricError(Exception):
    """A metric with the same name is already registered."""


class IncompatibleAlgoStrategyError(Exception):
    """This algo is not compatible with this strategy."""


class IndexGeneratorSampleNoneError(Exception):
    """Try to use the index generator without setting the number of samples."""


class IndexGeneratorUpdateError(Exception):
    """The index generator has not been updated properly."""


class InvalidPathError(Exception):
    """Invalid path."""


class InvalidMetricIdentifierError(Exception):
    """A metric name or identifier cannot be a SubstraFL Outputidentifier."""


class KeyMetadataError(Exception):
    """``substrafl_version``, ``substra_version`` and ``substratools_version`` keys can't be added
    to the experiment metadata."""


class LenMetadataError(Exception):
    """Too long additional metadata passed to the execute_experiment function to be shown on the Substra WebApp."""


class NegativeHessianMatrixError(Exception):
    """Hessian matrix is not positive semi-definite, the problem is not convex"""


class NumUpdatesValueError(Exception):
    """The num_update has been set to an non-authorize value."""


class OptimizerValueError(Exception):
    """The optimizer value is incompatible with the _local_train function."""


class ScaffoldLearningRateError(Exception):
    """When using the :class:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    a learning rate must be passed to the optimizer."""


class SharedStatesError(Exception):
    """Shared states error"""


class TorchScaffoldAlgoParametersUpdateError(Exception):
    """When using  :class:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    :func:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._scaffold_parameters_update` method
    must be called once for each update within the
    :func:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` method.
    """


class TaskNotFoundError(Exception):
    """The provided compute plan must contain a task:

    - hosted by the worker associated to the given client
    - tagged with the given round_idx or rank_idx
    """


class MultipleTaskError(Exception):
    """The experiment from which to get the algo files can't have multiple task tagged with the given round_idx or
    rank_idx hosted on the same organization."""


class UnfinishedTaskError(Exception):
    """The task from which to get the algo files shall be in status ``STATUS_DONE``."""


class LoadMetadataError(Exception):
    """The metadata.json file within the folder given as input must contain a ``model_file``"""


class LoadFileNotFoundError(Exception):
    """The given folder must contains the following files: function.tar.gz, metadata.json, the file entered in the
    ``model_file`` key of the dictionary."""


class LoadLocalDependencyError(Exception):
    """All dependencies from the local input folder should be install by the user."""


class UnsupportedPytorchVersionError(Exception):
    """Unsupported Pytorch version"""


class MetricFunctionSignatureError(Exception):
    """The metric_function() function on the given torch Dataset must ONLY have data_from_opener and
    predictions as parameters."""


class MetricFunctionTypeError(Exception):
    """The metric_function() must be of type function."""


class DependencyCacheNotFoundError(Exception):
    """No cache directory found for the dependencies."""


class WrongWheelNameError(Exception):
    """Error with the extracted wheel name from pip wheel."""
