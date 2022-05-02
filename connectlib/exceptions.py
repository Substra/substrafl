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
