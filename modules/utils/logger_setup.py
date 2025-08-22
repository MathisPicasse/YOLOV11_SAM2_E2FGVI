# ========================================
# Author: Mathis Picasse
# Created on: 12th, August 2025
# Description:
# ========================================

import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parents[2] / ".env"
load_dotenv(env_path)


def setup_logger(name):
    """Return a logger with a default ColoredFormatter."""
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  

    if not logger.handlers:  # ⚡ Évite les doublons
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


########################################
# Logger init
########################################
logger = setup_logger(name="logger_pipeline")
