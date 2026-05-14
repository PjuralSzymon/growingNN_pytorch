import logging

from .config import ENABLE_LOGGING, LOG_LEVEL

logging.raiseExceptions = False

logger = logging.getLogger("growingnn")

if ENABLE_LOGGING:
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)
else:
    logger.addHandler(logging.NullHandler())
