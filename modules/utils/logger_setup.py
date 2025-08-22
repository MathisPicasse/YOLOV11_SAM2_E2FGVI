# ========================================
# Author: Mathis Picasse
# Created on: 12th, August 2025
# Description: to manage the logging setup
# ========================================

import logging
import colorlog
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

env_path = Path(__file__).parents[2] / ".env"
load_dotenv(env_path)


def setup_logger(name: str, log_file_path: Optional[Path] = None):
    """
    Returns a logger with a default ColoredFormatter for the console
    and an optional FileHandler for writing logs to a file.
    """
    # Create a formatter for the console (with color codes).
    console_formatter = colorlog.ColoredFormatter(
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

    # Create a simple formatter for the log file (without color codes).
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Get the logger instance by name.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevents logs from being sent to the root logger, avoiding duplication.
    logger.propagate = False

    # Prevents duplicate handlers if the function is called multiple times.
    if not logger.handlers:
        # Add a handler for logging to the console.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Add a file handler if a log file path is provided.
        if log_file_path:
            # Ensure the directory for the log file exists.
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Delete the log file if it already exists to start fresh.
            if log_file_path.exists():
                log_file_path.unlink()
            
            # Create and configure the file handler.
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger

########################################
# Logger Initialization 
########################################

log_path = Path("./logs/YOLO_SAM2_E2FVGI.log")
logger = setup_logger(name="logger_convert_to_tolo", log_file_path=log_path)
