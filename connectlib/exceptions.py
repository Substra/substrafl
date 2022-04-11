class InvalidPathError(Exception):
    """Invalid path"""

    pass


class IndexGeneratorUpdateError(Exception):
    """The index generator has not been updated properly"""

    pass


class NumUpdatesValueError(Exception):
    """The num_update has been set to an non-authorize value"""

    pass
