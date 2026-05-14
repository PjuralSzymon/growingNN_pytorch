import logging
from logging.handlers import RotatingFileHandler

from .config import (
    ENABLE_LOGGING,
    LOG_LEVEL,
    LOG_TO_FILE,
    LOG_FILE_NAME,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUP_COUNT,
)

logging.raiseExceptions = False

logger = logging.getLogger("growingnn")

if ENABLE_LOGGING:
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if LOG_TO_FILE:
            file_handler = RotatingFileHandler(
                LOG_FILE_NAME,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
else:
    logger.addHandler(logging.NullHandler())
