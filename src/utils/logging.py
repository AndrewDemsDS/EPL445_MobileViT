"""Simple logging setup for project-wide use."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "traffic_mobilevit",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically the project name).
    log_file : str, optional
        If provided, also write logs to this file.
    level : int
        Logging level (default ``logging.INFO``).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
