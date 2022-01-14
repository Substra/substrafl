import logging

from connectlib.logger import set_logging_level


def test_default_logging_level(caplog):
    # check if the logger is at INFO by default
    logger = logging.getLogger(name="connectlib.junior")
    with caplog.at_level(logging.DEBUG):
        logger.info("This should be logged")
        logger.debug("This should not be logged")

        assert logging.getLogger("connectlib.junior").getEffectiveLevel() == logging.INFO
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[0].msg == "This should be logged"


def test_set_logging_level(caplog):
    # check the logs are displayed when I set them at the right level
    logger = logging.getLogger(name="connectlib.junior")
    with caplog.at_level(logging.DEBUG):
        set_logging_level(logging.WARNING)

        logger.warning("This should be logged")
        logger.info("This should not be logged")

        assert logging.getLogger("connectlib.junior").getEffectiveLevel() == logging.WARNING
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert caplog.records[0].msg == "This should be logged"
