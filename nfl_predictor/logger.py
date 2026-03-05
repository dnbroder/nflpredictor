"""
Centralized logging configuration for NFLPredictor.

All modules import from here:
    from nfl_predictor.logger import get_logger
    logger = get_logger(__name__)

Single logger hierarchy under "nflpredictor" root.
Format: timestamp | level | module | message
"""

import logging
import os
import sys

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root():
    global _configured
    if _configured:
        return

    root = logging.getLogger("nflpredictor")
    root.setLevel(LOG_LEVEL)

    # Remove any existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "botocore", "boto3", "s3transfer", "xgboost", "nfl_data_py"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger under the nflpredictor hierarchy.

    Usage:
        logger = get_logger(__name__)
        # → "nflpredictor.nfl_predictor.feature_engineering"

    Args:
        name: typically __name__ from the calling module

    Returns:
        Logger instance
    """
    _configure_root()

    # Always nest under nflpredictor root
    if name.startswith("nflpredictor"):
        full_name = name
    elif name == "__main__":
        full_name = "nflpredictor.main"
    else:
        # e.g. "nfl_predictor.feature_engineering" → "nflpredictor.feature_engineering"
        clean = name.replace("nfl_predictor.", "").replace("nfl_predictor", "")
        full_name = f"nflpredictor.{clean}".rstrip(".")

    return logging.getLogger(full_name)