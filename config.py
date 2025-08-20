# ========================================
# Author: Mathis Picasse
# Created on: 2025-08-15
# Description: Central configuration module for your project
# ========================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

# ========================================
# Paths Configuration
# ========================================


def get_env_variable(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(
            f"La variable d'environnement '{key}' n'est pas définie.")
    return value


try:
    ########################################
    # General Configuaration
    ########################################

    PROJECT_ROOT = Path(__file__).parent

    OUTPUTS_PATH = PROJECT_ROOT / get_env_variable("OUTPUTS_PATH")

    ########################################
    # Data Paths
    ########################################
    DATA_PATH_RAW = PROJECT_ROOT / get_env_variable("DATA_PATH_RAW")
    DATA_PATH_PROCESSED = PROJECT_ROOT / \
        get_env_variable("DATA_PATH_PROCESSED")

    ########################################
    # Models Paths
    ########################################

    # Folder with the different models (detection, masking, inpainting)
    MODELS_PATH = PROJECT_ROOT / get_env_variable("MODELS_PATH")

    # Specific to the detection model
    DETECTION_MODEL = get_env_variable("DETECTION_MODEL")
    DETECTION_MODEL_VERSION = get_env_variable("DETECTION_MODEL_VERSION")
    MODEL_PATH_DETECTION = MODELS_PATH / "detection" / \
        f"{DETECTION_MODEL.upper()}v" / \
        f"{DETECTION_MODEL}{DETECTION_MODEL_VERSION}.pt"

    # Specific to the tracker
    TRACKER_NAME = get_env_variable("TRACKER_NAME")
    TRACKER_PATH = PROJECT_ROOT / get_env_variable("TRACKER_PATH")
    TRACKER_CONFIG_PATH = TRACKER_PATH / TRACKER_NAME

    # Specific to the masking module
    MASK_MODEL = get_env_variable("MASK_MODEL")
    MASK_PATH = MODELS_PATH / "Masks" / MASK_MODEL

    # Specific to the inpainting video model
    INPAINTING_INFERENCE = get_env_variable("INPAINTING_INFERENCE")
    INPAINTING_MODEL = get_env_variable("INPAINTING_MODEL")
    INPAINTING_INFERENCE_PATH = MODELS_PATH / \
        "Inpainting" / "E2FGVI" / INPAINTING_INFERENCE
    INPAINTING_MODEL_PATH = MODELS_PATH / "Inpainting" / \
        "E2FGVI" / "release_model" / INPAINTING_MODEL

    ########################################
    # Training YOLO Paths
    ########################################
    TRAINING = Path(os.getenv("TRAINING"))
    PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL")
    PRETRAINED_MODEL_VERSION = os.getenv("PRETRAINED_MODEL_VERSION")
    DATASET_YAML = os.getenv("DATASET_YAML")

    PRETRAINED_MODEL_PATH = MODELS_PATH / "detection" / f"{PRETRAINED_MODEL.upper()}v" / \
        f"{PRETRAINED_MODEL}{PRETRAINED_MODEL_VERSION}.pt"

    DATASET_YAML_PATH = TRAINING / DATASET_YAML


except ValueError as e:
    raise RuntimeError(f"Erreur de configuration: {e}") from e

# ========================================
# Pipeline Configuration
# ========================================

STEPS = ["detection", "masking", "inpainting"]
NEED_PREPROCESSING = True
VIDEO_NAME = "MOT20-01_edited.mp4"
TARGETS_ENTITIES_IDS = [1, 2]
