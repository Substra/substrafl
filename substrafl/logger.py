import logging


def set_logging_level(loglevel):
    """Set the logging level for SubstraFL.
    To change it, use the following code:

    .. code-block:: python

        import substrafl
        import logging
        substrafl.set_logging_level(loglevel=logging.DEBUG) # or any other log level

    Args:
        loglevel: log level (e.g. logging.INFO)
    """
    logger = logging.getLogger(name="substrafl")
    logger.setLevel(loglevel)

    # remove precedent handler otherwise it bugs
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
