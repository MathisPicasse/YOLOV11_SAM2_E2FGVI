

import cv2
from modules.tasks.detector import YoloDetector, Detector
from modules.tasks.tracker import UltralyticsTracker
from modules.tasks.masker import Masker
from modules.utils.visualisation import draw_bbox, draw_mask
from modules.utils.video import create_frames, create_video, get_fps
from modules.data.observations import Observation
from modules.tasks.preprocessing import preprocess_video
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import os
from modules.utils.logger_setup import logger
from dotenv import load_dotenv
from datetime import datetime
load_dotenv(".env")

# ========================================
# Paths Configuration
# ========================================


NEED_PREPROCESSING = True

load_dotenv(".env")
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_PATH = os.getenv("OUTPUTS_PATH")
DATA_PATH_RAW = PROJECT_ROOT / os.getenv("DATA_PATH_RAW")
DATA_PATH_PROCESSED = PROJECT_ROOT / os.getenv("DATA_PATH_PROCESSED")


VIDEO_NAME = "MOT20-01_edited.mp4"

# --- Config Project name and video path ---

datetime_fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if NEED_PREPROCESSING:
    datetime_fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PROJECT_NAME = f"{datetime_fname}_{VIDEO_NAME.replace('.mp4', '')}"
    VIDEO_PATH = DATA_PATH_RAW / VIDEO_NAME
else:
    PROJECT_NAME = input("Project name? ")
    VIDEO_PATH = DATA_PATH_PROCESSED / PROJECT_NAME / VIDEO_NAME

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Vidéo introuvable : {VIDEO_PATH}")


# --- Config détection ---
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")

MODEL_PATH_DETECTION = (
    PROJECT_ROOT / os.getenv("MODELS_PATH") / "detection" /
    f"{MODEL_NAME.upper()}v" / f"{MODEL_NAME}{MODEL_VERSION}.pt"
)

# --- Config tracker ---
TRACKER_NAME = os.getenv("TRACKER_NAME")
TRACKER_CONFIG_PATH = PROJECT_ROOT / os.getenv("TRACKER_PATH") / TRACKER_NAME

# --- Config masker ---
MASK_MODEL = os.getenv("MASK_MODEL")
MASK_PATH = PROJECT_ROOT / os.getenv("MODELS_PATH") / "Masks" / MASK_MODEL

MASKS_OUTPUT = PROJECT_ROOT / \
    os.getenv("OUTPUTS_PATH") / PROJECT_NAME / "masks"

TARGETS_ENTITIES_IDS = [1]  # Example: Process tracker IDs 1, 2, and 3

logger.info("######## VIDEO ANALYSIS PIPELINE #######")

# ========================================
# PREPROCESSING MODULE
# ========================================
if NEED_PREPROCESSING:
    # Defining the path to the folders that will be created to store the
    # processed frames and video
    output_frame_dir = DATA_PATH_PROCESSED / PROJECT_NAME / "frames"
    output_video_path = DATA_PATH_PROCESSED / \
        PROJECT_NAME / f"processed_{VIDEO_NAME}"
    frame_prefix = PROJECT_NAME

    preprocess_video(
        video_path=VIDEO_PATH,
        output_frame_dir=output_frame_dir,
        output_video_path=output_video_path,
        frame_prefix=frame_prefix,
        target_size=(864, 480),
        fps=25,
        codec='mp4v'
    )

    # We need to update the paths because now the video is processed
    # In the next part, we need to use our processed folder
    VIDO_PATH = DATA_PATH_PROCESSED / PROJECT_NAME / "processed_{VIDEO_NAME}"
    VIDEO_NAME = f"processed_{VIDEO_NAME}"

# ========================================
# OBJECT DETECTION + TRACKING
# ========================================

logger.info("Initializing modules...")

detector = YoloDetector(MODEL_PATH_DETECTION)
tracker = UltralyticsTracker(TRACKER_CONFIG_PATH)

# A detection pipeline is the combination of a detector and a tracker
detection_pipeline = Detector(detector=detector, tracker=tracker)

# === Run Detection + Tracking Pipeline ===
output_video_path = PROJECT_ROOT / \
    os.getenv("OUTPUTS_PATH") / PROJECT_NAME / "tracking"
detection_pipeline.track(VIDEO_PATH, output_video_path)

# we create a dictionnary where the key in the entitiy_id and the value is a list with all the observations objects.
# this helps us for the tracking part.
detections_by_entity = detection_pipeline.group_detections_by_entity()


# ========================================
# MASKING MODULE
# ========================================

# === Initialization ===
masker = Masker(MASK_MODEL)
# ===  Masking ===
all_observations_for_targets: List[Observation] = []
for target_entity_id in TARGETS_ENTITIES_IDS:

    if target_entity_id in detections_by_entity:
        all_observations_for_targets.extend(
            detections_by_entity[target_entity_id])
    else:
        logger.warning(
            f"Tracker ID {target_entity_id} not found in tracking results. Skipping for masking.")

    if not all_observations_for_targets:
        logger.info(
            "No observations found for the target tracker IDs. Skipping masking and visualization.")
    else:
        # Ensure MASKS_OUTPUT directory exists
        os.makedirs(MASKS_OUTPUT, exist_ok=True)
        logger.info(
            f"Mask output directory ensured/created at: {MASKS_OUTPUT}")

    if all_observations_for_targets:

        # Getting the list of the processed frames to start masking
        frames_files = sorted(os.listdir(
            DATA_PATH_PROCESSED / PROJECT_NAME / "frames"))
        for frame_idx, frame_file in enumerate(frames_files):
            frame = cv2.imread(DATA_PATH_PROCESSED /
                               PROJECT_NAME / "frames" / frame_file)
            current_frame_observations = [
                obs for obs in all_observations_for_targets if obs.frame_id == frame_idx
            ]

            mask_filename = f"mask_frame{frame_idx:06d}.png"
            mask = masker.create_mask(
                frame,
                current_frame_observations,
                MASKS_OUTPUT,
                mask_filename
            )
            frame_idx += 1
    logger.info("######## PIPELINE FINISHED #######")


# ========================================
# INPAINTING MODULE
# ========================================
