"""Logging configuration.

Call :func:`setup_logging` exactly once from the process entry point
(``app.py`` or a script). Library modules only ever call
:func:`get_logger`.
"""

from __future__ import annotations

import logging
import os
import sys
import typing
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"
_ROOT_LOGGER_NAME: Final[str] = "ai_trial_room"

#: Third-party loggers that are far too chatty at INFO.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "urllib3",
    "PIL",
    "matplotlib",
    "filelock",
    "asyncio",
    "huggingface_hub",
)

_configured = False


def _safe_stream() -> "typing.TextIO":
    """Return a stdout stream that cannot raise :class:`UnicodeEncodeError`.

    A default Windows console uses cp1252, so a single non-ASCII character in a
    log message - a Devanagari garment name, a curly quote from a filename, an
    emoji - raises mid-handler and takes the request with it. Switching the
    stream to ``errors="replace"`` degrades those characters to ``?`` instead.

    Returns
    -------
    typing.TextIO
        ``sys.stdout``, reconfigured in place when possible.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached/odd stream
            pass
    return stream


def setup_logging(level: str | int = "INFO", *, quiet_third_party: bool = True) -> None:
    """Configure process-wide logging.

    Idempotent: repeated calls only adjust the level, so importing this from a
    notebook cell twice will not duplicate handlers.

    Parameters
    ----------
    level:
        Level name (``"DEBUG"``) or numeric level.
    quiet_third_party:
        Raise noisy dependency loggers to WARNING.
    """
    global _configured

    resolved = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(resolved, int):  # unknown name -> logging returns a str
        resolved = logging.INFO

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(resolved)

    if not _configured:
        handler = logging.StreamHandler(stream=_safe_stream())
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.propagate = False
        _configured = True

    if quiet_third_party:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    # TensorFlow / MediaPipe print C++ log spam unless muted before import.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger.

    Parameters
    ----------
    name:
        Usually ``__name__``. A leading ``ai_trial_room.`` is stripped so the
        logger name stays short in the console.
    """
    short = name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")
