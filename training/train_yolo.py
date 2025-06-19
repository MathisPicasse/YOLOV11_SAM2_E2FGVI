"""
File: train_yolo.py
Author: Mathis Picasse
Description: Script to train YOLO using a pre-trained model.
"""
from ultralytics import YOLO
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_MODEL_NAME: str = "yolo11n"
DEFAULT_PATH_TO_MODEL_TEMPLATE: str = f"/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/dectection/YOLO11v/{DEFAULT_MODEL_NAME}.pt"
DEFAULT_DATASET_YAML: str = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/training/dataset.yaml"  
DEFAULT_PROJECT_TEMPLATE: str = f"run/detect/{DEFAULT_MODEL_NAME}"  

try:
    model = YOLO(DEFAULT_PATH_TO_MODEL_TEMPLATE)
    logging.info(f"Successfully loaded/initialized model: {DEFAULT_MODEL_NAME}")
except Exception as e:
    logging.error(f"An unexpected error occurred while loading/initializing the model '{DEFAULT_PATH_TO_MODEL_TEMPLATE}': {e}", exc_info=True)
    raise RuntimeError(f"Failed to load/initialize YOLO model from '{DEFAULT_PATH_TO_MODEL_TEMPLATE}'") from e

training_parameters: Dict[str, Any] = {
        "data": DEFAULT_DATASET_YAML,
        "epochs": 1,            # Number of training epochs
        "patience": 30,         # Epochs to wait for no observable improvement before early stopping
        "batch": 16,            # Batch size (adjust based on available GPU memory)
        "imgsz": 640,           # Input image size (square images, e.g., 640x640)
        "save": True,           # Save training artifacts (checkpoints, logs, etc.)
        "cache": False,         # Cache images for faster training (RAM-intensive, use cautiously)
        "device": "cpu",        # Device to run on: 'cpu', '0' (for GPU 0), '0,1,2,3', etc.
        "project": DEFAULT_PROJECT_TEMPLATE,# Root directory for saving all related runs
        "name": f"{DEFAULT_MODEL_NAME}_run_{1}epochs", # Name for the specific training run directory (e.g., yolov8n_run_100epochs)
        "multi_scale": False,   # Vary image size during training for robustness (can improve mAP)
        "exist_ok": False,      # If False, errors if 'project/name' already exists, preventing accidental .
                                # If True, reuses or overwrites the existing directory.
    }

# --- Training ---
model.train(**training_parameters)

# --- evaluation --
evaluation = model.val()