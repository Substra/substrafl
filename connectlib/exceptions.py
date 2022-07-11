# Alphabetically ordered


class AlgoCategoryError(Exception):
    """The given algo category does not match any substra AlgoCategory."""


class ConnectToolsDeprecationWarning(DeprecationWarning):
    """The connect-tools version used is deprecated."""


class CriterionReductionError(Exception):
    """The criterion reduction must be set to 'mean' to use the Newton-Raphson strategy."""


class DampingFactorValueError(Exception):
    """The damping factor must be greater than 0 and less than or equal to 1"""


class EmptySharedStatesError(Exception):
    """The shared_states is empty. Ensure that the train method of the algorithm returns a
    StrategySharedState object."""


class IncompatibleAlgoStrategyError(Exception):
    """This algo is not compatible with this strategy."""


class IndexGeneratorSampleNoneError(Exception):
    """Try to use the index generator without setting the number of samples."""


class IndexGeneratorUpdateError(Exception):
    """The index generator has not been updated properly."""


class InvalidPathError(Exception):
    """Invalid path."""


class KeyMetadataError(Exception):
    """``connectlib_version``, ``substra_version`` and ``substratools_version`` keys can't be added
    to the experiment metadata."""


class LenMetadataError(Exception):
    """Too long additional metadata passed to the execute_experiment function to be shown on the Connect WebApp."""


class NegativeHessianMatrixError(Exception):
    """Hessian matrix is not positive semi-definite, the problem is not convex"""


class NumUpdatesValueError(Exception):
    """The num_update has been set to an non-authorize value."""


class OptimizerValueError(Exception):
    """The optimizer value is incompatible with the _local_train function."""


class ScaffoldLearningRateError(Exception):
    """When using the :class:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    a learning rate must be passed to the optimizer."""


class SharedStatesError(Exception):
    """Shared states error"""


class TorchScaffoldAlgoParametersUpdateError(Exception):
    """When using  :class:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    :func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._scaffold_parameters_update` method
    must be called once for each update within the
    :func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` method.
    """


class TrainTaskNotFoundError(Exception):
    """When using the :func:`~connectlib.model_loading.download_algo_files` function. The provided compute plan must
    contain a composite train tuple:

        - hosted by the worker associated to the given client
        - tagged with the given round_idx
    """


class MultipleTrainTaskError(Exception):
    """When using the :func:`~connectlib.model_loading.download_algo_files` function in remote mode. The experiment to
    get the algo files from can't have multiple TrainDataNodes hosted on the same organization."""


class UnfinishedTrainTaskError(Exception):
    """When using the :func:`~connectlib.model_loading.download_algo_files` function. The task to get the algo files
    from shall be in status ``STATUS_DONE``."""


class LoadAlgoMetadataError(Exception):
    """When using the :func:`~connectlib.model_loading.load_algo`, the metadata.json file within the folder given as
    input must contain a ``local_state_file``"""


class LoadAlgoFileNotFoundError(Exception):
    """When using the :func:`~connectlib.model_loading.load_algo`, the given folder must contains the following files:
    algo.tar.gz, metadata.json, the file entered in the ``local_state_file`` key of the dictionary."""


class LoadAlgoLocalDependencyError(Exception):
    """When using the :func:`~connectlib.model_loading.load_algo`, all dependencies from the local input folder should be
    install by the user."""
