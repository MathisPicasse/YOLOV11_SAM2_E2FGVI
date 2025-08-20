# ========================================
# Author: Mathis Picasse
# Created on: 2025-07-02
# Last Modified: 2025-08-20
# Description: Script to train YOLO using a pre-trained model
# ========================================


from ultralytics import YOLO
from typing import Dict, Any
from datetime import datetime
from modules.utils.logger_setup import logger
from dotenv import load_dotenv
from config import (
    TRAINING,
    PRETRAINED_MODEL,
    PRETRAINED_MODEL_VERSION,
    PRETRAINED_MODEL_PATH,
    DATASET_YAML_PATH,
)


load_dotenv("./.env")


try:
    model = YOLO(PRETRAINED_MODEL_PATH)
    logger.info(
        f"Successfully loaded/initialized model: {PRETRAINED_MODEL}{PRETRAINED_MODEL_VERSION}")
except Exception as e:
    logger.error(
        f"An unexpected error occurred while loading/initializing the model '{PRETRAINED_MODEL}{PRETRAINED_MODEL_VERSION}': {e}", exc_info=True)

training_parameters: Dict[str, Any] = {
    "data": DATASET_YAML_PATH,
    "epochs": 40,            # Number of training epochs
    "patience": 50,         # Epochs to wait for no observable improvement before early stopping
    # Batch size (adjust based on available GPU memory)
    "batch": 8,
    "imgsz": 864,           # Input image size
    # Save training artifacts (checkpoints, logs, etc.)
    "save": True,
    # Cache images for faster training (RAM-intensive, use cautiously)
    "cache": True,
    # Device to run on: 'cpu', '0' (for GPU 0), '0,1,2,3', etc.
    "device": "cpu",
    # Root directory for saving all related runs
    "project": TRAINING / "run",
    # Vary image size during training for robustness (can improve mAP)
    "name": f"{PRETRAINED_MODEL}{PRETRAINED_MODEL_VERSION}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    "multi_scale": True,
    # If False, errors if 'project/name' already exists, preventing accidental .
    "exist_ok": False,
    # If True, reuses or overwrites the existing directory.
}


# --- Training ---
model.train(**training_parameters)
