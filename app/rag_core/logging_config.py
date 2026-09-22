"""Application-owned logging configuration.

Chat history is deliberately isolated from the root logger so dependency
traffic (notably httpx request records) cannot pollute the JSON conversation
log.
"""

import logging
from pathlib import Path


CHAT_LOGGER_NAME = "toyota_rag.chat"
_OWNED_HANDLER_MARKER = "_toyota_chat_file_handler"


def configure_chat_logger(log_path):
    """Return an idempotently configured, non-propagating chat logger."""
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove a handler left on the root logger by the previous basicConfig
    # implementation during a Streamlit hot reload. The exact chat-log path is
    # the boundary; unrelated application/Streamlit handlers are untouched.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if Path(handler.baseFilename).resolve() == path:
            root_logger.removeHandler(handler)
            handler.close()

    # These libraries emit every successful Ollama POST at INFO. They remain
    # able to report warnings/errors to the normal terminal logging pipeline.
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger = logging.getLogger(CHAT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if not getattr(handler, _OWNED_HANDLER_MARKER, False):
            continue
        if Path(handler.baseFilename).resolve() == path:
            return logger
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(path, encoding="utf-8")
    setattr(handler, _OWNED_HANDLER_MARKER, True)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
