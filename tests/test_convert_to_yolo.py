import os
import sys # For sys.exit
import json
import logging
# display_bbox_on_image is used by create_video_bbox, so direct import isn't strictly needed here
# from modules.utils.image import display_bbox_on_image 
from modules.utils.video import create_video_bbox
from typing import Dict, Tuple


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
JSON_CONFIG_FILE = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/configs/MOT_TO_YOLO.json"

# Define output paths for this test script (can be different from main conversion)
# These are example paths; adjust as needed.
# It's often good to use a dedicated 'test_outputs' directory.
TEST_IMG_DIR = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/training/val/MOT17-09_yolo/images"
TEST_ANNOTATIONS_DIR = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/training/val/MOT17-09_yolo/labels"
TEST_OUTPUT_VIDEO = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/outputs/test_MOT17-09_yolo_annotated.mp4"
VIDEO_PREFIX = "MOT17-09" # This should match the prefix of your images in TEST_IMG_DIR
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v' # Common codec for .mp4 files

def generate_class_colors(class_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """Generates a distinct color for each class ID."""
    colors_list = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0)
    ] 
    
    class_colors: Dict[int, Tuple[int, int, int]] = {}
    for i, class_id in enumerate(sorted(class_names.keys())):
        class_colors[class_id] = colors_list[i % len(colors_list)]
    return class_colors

def main():
    logging.info(f"Attempting to load JSON configuration from: {JSON_CONFIG_FILE}")
    config_data = None
    try:
        with open(JSON_CONFIG_FILE, "r") as f_json:
            config_data = json.load(f_json)
        logging.info(f"Successfully loaded JSON configuration from {JSON_CONFIG_FILE}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {JSON_CONFIG_FILE}")
        sys.exit(1) # Exit script if config is not found
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file: {e}")
        sys.exit(1) # Exit script on bad JSON
    # No _validate_config here, assuming JSON is correct for this test script's needs
    # or that the main script already validated it.

    if not config_data or "Classes" not in config_data:
        logging.error("JSON configuration is missing 'Classes' key or is empty.")
        sys.exit(1)

    
    # Step 1: Create a temporary map from original string ID to name
    original_classes_from_config: Dict[str, str] = config_data["Classes"]
    
    # Step 2: Create a map from new integer ID (0-indexed) to name
    # This mimics what `map_classes`'s `new_id_to_name_map` would produce.
    # The order depends on dict iteration, which is insertion order for Python 3.7+
    yolo_class_id_counter = 0
    yolo_class_dict: Dict[int, str] = {}
    for original_str_id in sorted(original_classes_from_config.keys(), key=lambda x: int(x) if x.isdigit() else x):
        yolo_class_dict[yolo_class_id_counter] = original_classes_from_config[original_str_id]
        yolo_class_id_counter += 1
        
    logging.info(f"YOLO Class Dictionary for video: {yolo_class_dict}")

    # Generate colors for these YOLO class IDs
    class_colors = generate_class_colors(yolo_class_dict)
    logging.info(f"Class colors for video: {class_colors}")

    logging.info("Starting video creation with bounding boxes...")
    try:
        create_video_bbox(
            img_dir=TEST_IMG_DIR,
            annotations_dir=TEST_ANNOTATIONS_DIR,
            output_video=TEST_OUTPUT_VIDEO,
            prefix=VIDEO_PREFIX, # Corrected: comma was missing
            fps=VIDEO_FPS,      # Corrected: convert to int
            codec=VIDEO_CODEC,  # Corrected: 'mp4v' is more common
            class_dict=yolo_class_dict, # Use the remapped YOLO class IDs
            colors=class_colors
        )
        logging.info(f"Video creation complete. Output at: {TEST_OUTPUT_VIDEO}")
    except FileNotFoundError as e:
        logging.error(f"Error during video creation: A required file or directory was not found. {e}")
    except ValueError as e:
        logging.error(f"Error during video creation: Invalid value encountered. {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during video creation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
