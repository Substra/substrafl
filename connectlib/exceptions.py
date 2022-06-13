class InvalidPathError(Exception):
    """Invalid path."""

    pass


class IndexGeneratorUpdateError(Exception):
    """The index generator has not been updated properly."""

    pass


class NumUpdatesValueError(Exception):
    """The num_update has been set to an non-authorize value."""

    pass


class ConnectToolsDeprecationWarning(DeprecationWarning):
    """The connect-tools version used is deprecated."""

    pass


class IndexGeneratorSampleNoneError(Exception):
    """Try to use the index generator without setting the number of samples."""

    pass


class LenMetadataError(Exception):
    """Too long additional metadata passed to the execute_experiment function to be shown on the Connect WebApp."""

    pass


class KeyMetadataError(Exception):
    """`connectlib_version`, `substra_version` and `substratools_version` keys can't be added
    to the experiment metadata."""


class IncompatibleAlgoStrategyError(Exception):
    """This algo is not compatible with this strategy."""

    pass


class ScaffoldLearningRateError(Exception):
    """When using the :class:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    a learning rate must be passed to the optimizer."""

    pass


class TorchScaffoldAlgoParametersUpdateError(Exception):
    """When using  :class:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo`,
    :func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._scaffold_parameters_update` method
    must be called once for each update within the
    :func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` method.
    """
