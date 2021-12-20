import logging


def set_logging_level(loglevel):
    """Set the logging level for connectlib

    Args:
        loglevel: log level (e.g. logging.INFO)
    """
    logger = logging.getLogger(name="connectlib")
    logger.setLevel(loglevel)

    # remove precedent handler otherwise it bugs
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    consoleHandler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
