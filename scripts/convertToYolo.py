"""
File: convertToYolo.py
Author: Mathis Picasse
Description: Script to convert a MOT dataset to a Yolo dataset.
"""

import json
import csv
import os
import logging
import argparse
from typing import Dict, Tuple, Union, Optional, List, Any


from modules.utils.file import setup_output_dirs
from modules.utils.image import process_image
from modules.utils.geometry import rescale_bbox_coordinates, normalize_bbox_coordinates






logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def map_classes(
    mot_classes: Dict[Union[str, int], str]
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Maps original class IDs to new sequential IDs and provides a name mapping.

    This function takes a dictionary of original class IDs (which can be
    strings or integers representing class identifiers) and their corresponding
    class names. It produces two dictionaries:
    1. A mapping from the original class ID (coerced to an integer) to a new,
       zero-based sequential integer ID.
    2. A mapping from this new sequential ID to the original class name.
    The order of new IDs is based on the iteration order of the input dictionary.

    Args:
        mot_classes (Dict[Union[str, int], str]): A dictionary where keys
            are the original class IDs (e.g., from a MOT dataset) and
            values are the corresponding class names. Keys must be
            convertible to integers.

    Returns:
        Tuple[Dict[int, int], Dict[int, str]]:
            A tuple containing two dictionaries:
            - original_to_new_id_map (Dict[int, int]): Maps original
              integer class IDs to new zero-based sequential IDs.
            - new_id_to_name_map (Dict[int, str]): Maps new zero-based
              sequential IDs to their class names.

    Raises:
        TypeError: If `mot_classes` is not a dictionary.
        ValueError: If any key in `mot_classes` cannot be converted to an
                    integer.

    Example:
        >>> classes1 = {'1': 'car', '7': 'person', 2: 'bicycle'}
        >>> id_map1, name_map1 = map_classes(classes1)
        >>> id_map1 is {1: 0, 7: 1, 2: 2}
        >>> name_map1 is {0: 'car', 1: 'person', 2: 'bicycle'}
    """
    if not isinstance(mot_classes, dict):
        raise TypeError("Input 'mot_classes' must be a dictionary.")

    original_to_new_id_map: Dict[int, int] = {}
    new_id_to_name_map: Dict[int, str] = {}
    
    new_class_id_counter  = 0
    # Iterate through items, providing both original ID key and class name.
    for original_id_key, class_name in mot_classes.items():
        original_id_int = int(original_id_key)
        original_to_new_id_map[original_id_int] = new_class_id_counter
        new_id_to_name_map[new_class_id_counter] = class_name
        new_class_id_counter += 1
        
    return original_to_new_id_map, new_id_to_name_map


def _parse_annotation_row(
    row: List[str],
    mapped_dict: Dict[int, int],
    original_img_width: int,
    original_img_height: int,
    resize_img: bool,
    target_size: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int, float, float, float, float]]:
    """
    Parses a single row from MOT annotation CSV and calculates YOLO coordinates.

    Args:
        row (List[str]): A list of strings representing a row from the CSV.
        mapped_dict (Dict[int, int]): Mapping from original class IDs to new YOLO class IDs.
        original_img_width (int): Width of the original image.
        original_img_height (int): Height of the original image.
        resize_img (bool): Whether the image associated with this annotation will be resized.
        target_size (Optional[Tuple[int, int]]): Target (width, height) if resizing.

    Returns:
        Optional[Tuple[int, int, float, float, float, float]]:
            A tuple (frame, class_id, x_center_norm, y_center_norm, w_norm, h_norm)
            or None if the class ID is not in mapped_dict or row is malformed.

    Raises:
        ValueError: If original image dimensions are not positive when not resizing.
    """
    try:
        frame = int(row[0])
        original_class_id = int(row[7])
        
        if original_class_id not in mapped_dict:
            logging.debug(f"Skipping class ID {original_class_id} not in mapped_dict for frame {frame}.")
            return None
            
        yolo_class_id = mapped_dict[original_class_id]
        x_tl, y_tl, w_box, h_box = map(float, (row[2], row[3], row[4], row[5]))

        current_img_width = original_img_width
        current_img_height = original_img_height
        current_x, current_y, current_w, current_h = x_tl, y_tl, w_box, h_box

        if resize_img and target_size:
            if original_img_width <= 0 or original_img_height <= 0:
                logging.error(f"Frame {frame}: Original image dimensions for rescaling must be positive.")
                return None

            new_img_width, new_img_height = target_size
            current_x, current_y, current_w, current_h = rescale_bbox_coordinates(
                x_tl, y_tl, w_box, h_box,
                original_img_width, original_img_height,
                new_img_width, new_img_height
            )
            current_img_width, current_img_height = new_img_width, new_img_height
        
        # Ensure current dimensions for normalization are positive
        if current_img_width <= 0 or current_img_height <= 0:
            logging.warning(
                f"Frame {frame}, Class {yolo_class_id}: "
                f"Image dimensions ({current_img_width}x{current_img_height}) "
                f"for normalization must be positive. Skipping annotation."
            )
            return None

        x_center_norm, y_center_norm, w_box_norm, h_box_norm = normalize_bbox_coordinates(
            current_x, current_y, current_w, current_h,
            current_img_width, current_img_height
        )
        
        return frame, yolo_class_id, x_center_norm, y_center_norm, w_box_norm, h_box_norm

    except (IndexError, ValueError) as e:
        logging.warning(f"Skipping malformed annotation row: {row}. Error: {e}")
        return None


def convert_to_yolo(
        img_dir: str,
        annotations_file: str,
        prefix: str, 
        output_dir: str,
        img_width: int,
        img_height: int,
        mapped_dict: Dict[int, int],
        resize_img: bool = False,
        target_size: Tuple[int, int] = (640, 640)
):
    # Create the 'images' and 'labels' directories in the output directory
    images_output, labels_output = setup_output_dirs(output_dir, ['images', 'labels'])


    # Create a dictionary where keys are frame numbers and values are lists of annotations (class_id, x, y, w, h)
    annotations_by_frame = {}

    # Read the annotation file
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame = int(row[0])  # Extract frame number
            class_id = int(row[7])  # Extract class ID
            if class_id in mapped_dict:
                x, y, w_box, h_box = map(float, (row[2], row[3], row[4], row[5]))
                class_id = mapped_dict[class_id]  # Map to new class ID

                # Initialize frame entry if not present
                if frame not in annotations_by_frame:
                    annotations_by_frame[frame] = []
                
                # Resize bounding box if resizing is enabled
                if resize_img:
                    new_img_width, new_img_height = target_size
                    x_new, y_new, wbox_new, hbox_new = rescale_bbox_coordinates(
                        x, y, w_box, h_box, img_width, img_height, new_img_width, new_img_height
                    )
                    x_center_norm, y_center_norm, w_box_norm, h_box_norm = normalize_bbox_coordinates(
                        x_new, y_new, wbox_new, hbox_new, new_img_width, new_img_height
                    )
                else:
                    x_center_norm, y_center_norm, w_box_norm, h_box_norm = normalize_bbox_coordinates(
                        x, y, w_box, h_box, img_width, img_height
                    )
                
                annotations_by_frame[frame].append((class_id, x_center_norm, y_center_norm, w_box_norm, h_box_norm))
    
    # Process and save images and annotations
    for img_file in sorted(os.listdir(img_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            new_name = process_image(img_file, img_dir, images_output, resize_img, target_size, prefix)
            frame_number = int(img_file.split('.')[0])  # Extract frame number from filename
            
            if frame_number in annotations_by_frame:
                yolo_annotations = annotations_by_frame[frame_number]
                label_file_path = os.path.join(labels_output, f"{prefix}_{frame_number:06d}.txt")
                with open(label_file_path, 'w') as label_file:
                    for annotation in yolo_annotations:
                        label_file.write(" ".join(map(str, annotation)) + "\n")
def convert_to_yolo(
    img_dir: str,
    annotations_file: str,
    output_prefix: str, # Renamed for clarity to match usage
    output_dir: str,
    original_img_width: int, # Renamed for clarity
    original_img_height: int, # Renamed for clarity
    mapped_dict: Dict[int, int],
    resize_img: bool = False,
    target_size: Optional[Tuple[int, int]] = (640, 640), # Made Optional to align with docstring
    subsample_rate: Optional[int] = None
) -> None:
    """
    Converts a dataset from MOT format to YOLO object detection format.

    This involves reading annotations, normalizing bounding box coordinates,
    optionally resizing images and their corresponding bounding boxes, and
    saving them in the structure expected by YOLO (images and label files).

    Args:
        img_dir (str): Path to the directory containing source images.
        annotations_file (str): Path to the MOT format annotation file (CSV).
        output_prefix (str): A prefix to be added to output image and label filenames.
        output_dir (str): Root directory where the YOLO formatted dataset
                          (images/ and labels/ subdirectories) will be saved.
        original_img_width (int): Width of the original images in `img_dir`.
        original_img_height (int): Height of the original images in `img_dir`.
        mapped_dict (Dict[int, int]): Dictionary mapping original MOT class IDs
                                     to new sequential YOLO class IDs.
        resize_img (bool): If True, images and bounding boxes will be resized
                           to `target_size`. Defaults to False.
        target_size (Optional[Tuple[int, int]]): Target (width, height) for resizing.
                                     Used only if `resize_img` is True. Defaults to (640, 640).
        subsample_rate (Optional[int]): If set to an integer N > 1, only 1 out of every N
                                        frames will be processed. For example, a rate of 2
                                        keeps frames 0, 2, 4, etc. Defaults to None (all frames).

    Raises:
        FileNotFoundError: If `img_dir` or `annotations_file` does not exist.
        IOError: If there's an issue reading files or writing output.
        ValueError: If `original_img_width` or `original_img_height` are not
                    positive when `resize_img` is False or when they are needed
                    for rescaling.
    """
    logging.info(f"Starting YOLO conversion for image directory: {img_dir}")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isfile(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    if not resize_img and (original_img_width <= 0 or original_img_height <= 0):
        raise ValueError(
            "Original image width and height must be positive if not resizing."
        )
    if resize_img and not target_size: # target_size must be valid if resizing
        raise ValueError("target_size must be provided and valid if resize_img is True.")


    # Create the 'images' and 'labels' directories in the output directory
    dir_paths = setup_output_dirs(output_dir, ['images', 'labels'])
    images_output_dir = dir_paths['images']
    labels_output_dir = dir_paths['labels']

    annotations_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    logging.info(f"Reading annotations from: {annotations_file}")
    try:
        with open(annotations_file, 'r', newline='') as f: # Added newline='' for csv
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                parsed_annotation = _parse_annotation_row(
                    row, mapped_dict, original_img_width, original_img_height,
                    resize_img, target_size
                )
                if parsed_annotation:
                    frame, class_id, x_c, y_c, w_n, h_n = parsed_annotation
                    if frame not in annotations_by_frame:
                        annotations_by_frame[frame] = []
                    annotations_by_frame[frame].append((class_id, x_c, y_c, w_n, h_n))
    except IOError as e:
        logging.error(f"Could not read annotations file {annotations_file}: {e}")
        raise
    except Exception as e: # Catch any other unexpected errors during CSV processing
        logging.error(f"Unexpected error processing CSV file {annotations_file}: {e}")
        raise
    
    logging.info(f"Processing images from: {img_dir}")
    processed_image_count = 0
    
    # Get a sorted list of image files to ensure consistent order for subsampling
    image_files = sorted([
        f for f in os.listdir(img_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for i, img_filename in enumerate(image_files):
        # If subsampling is enabled, skip frames that are not on the interval
        if subsample_rate and (i % subsample_rate != 0):
            continue

        full_img_path = os.path.join(img_dir, img_filename)
        effective_target_size = target_size if resize_img else None
        
        try:
            # process_image saves the image with the prefix already.
            # It returns the basename of the saved image.
            _ = process_image( # We don't need the returned basename here
                img_path=full_img_path,
                output_dir=images_output_dir,
                output_prefix=output_prefix,
                target_size=effective_target_size
            )
            processed_image_count +=1
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Failed to process image {full_img_path}: {e}")
            continue # Skip to next image if one fails

        try:
            # Assuming filename format like '000001.jpg'
            frame_number_str = os.path.splitext(img_filename)[0]
            frame_number = int(frame_number_str)
        except ValueError:
            logging.warning(
                f"Could not extract frame number from image filename: {img_filename}. "
                f"Skipping annotation saving for this image."
            )
            continue
            
        if frame_number in annotations_by_frame:
            yolo_annotations = annotations_by_frame[frame_number]
            # Use the original frame number for the label filename, prefixed.
            label_filename = f"{output_prefix}_{frame_number:06d}.txt"
            label_file_path = os.path.join(labels_output_dir, label_filename)
            
            try:
                with open(label_file_path, 'w') as label_file:
                    for annotation_data in yolo_annotations:
                        # Ensure all parts of annotation_data are strings for join
                        label_file.write(" ".join(map(str, annotation_data)) + "\n")
                logging.debug(f"Saved YOLO annotations to: {label_file_path}")
            except IOError as e:
                logging.error(f"Could not write label file {label_file_path}: {e}")
        else:
            logging.info(f"No annotations found for frame {frame_number} (image: {img_filename}).")
            # Optionally, create an empty label file if no annotations exist for an image
            # This is often required by YOLO training scripts.
            label_filename = f"{output_prefix}_{frame_number:06d}.txt"
            label_file_path = os.path.join(labels_output_dir, label_filename)
            try:
                with open(label_file_path, 'w') as label_file:
                    pass # Creates an empty file
                logging.debug(f"Created empty label file for frame {frame_number}: {label_file_path}")
            except IOError as e:
                 logging.error(f"Could not create empty label file {label_file_path}: {e}")


    logging.info(
        f"YOLO conversion completed for {output_prefix}. Processed {processed_image_count} images. "
        f"Output in: {output_dir}"
    )


def _validate_config(config: Dict[str, Any]) -> None:
    """Validates the structure and presence of essential keys in the config."""
    required_top_level_keys = [
        "PathToDataFolders", "Folders", "OutputDir", "Classes",
        "ResizeImages" # TargetSize is optional if ResizeImages is false
    ]
    for key in required_top_level_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: '{key}' in JSON file.")
    
    if not isinstance(config["PathToDataFolders"], str):
        raise ValueError("'PathToDataFolders' must be a string.")
    if not isinstance(config["Folders"], dict):
        raise ValueError("'Folders' must be a dictionary in JSON config.")
    if not isinstance(config["OutputDir"], str):
        raise ValueError("'OutputDir' must be a string.")
    if not isinstance(config["Classes"], dict):
        raise ValueError("'Classes' must be a dictionary in JSON config.")
    if not isinstance(config["ResizeImages"], bool):
        raise ValueError("'ResizeImages' must be a boolean.")

    if config["ResizeImages"]:
        if "TargetSize" not in config:
            raise ValueError("'TargetSize' is required when 'ResizeImages' is true.")
        if not (isinstance(config["TargetSize"], list) and len(config["TargetSize"]) == 2 and
                all(isinstance(dim, int) and dim > 0 for dim in config["TargetSize"])):
            raise ValueError("'TargetSize' must be a list of two positive integers [width, height].")

    if "SubsampleRate" in config:
        if not (isinstance(config["SubsampleRate"], int) and config["SubsampleRate"] > 1):
            raise ValueError("'SubsampleRate' must be an integer greater than 1.")

    for folder_name, dims in config["Folders"].items():
        if not (isinstance(dims, list) and len(dims) == 2 and
                all(isinstance(d, int) and d > 0 for d in dims)):
            raise ValueError(
                f"Dimensions for folder '{folder_name}' must be a list of two positive integers. Got: {dims}"
            )


def main():
    """
    Main function to orchestrate the dataset conversion process.
    Parses command-line arguments for the JSON configuration file,
    loads the configuration, and processes each specified dataset folder.
    """
    parser = argparse.ArgumentParser(
        description="Converts MOT datasets to YOLO format based on a JSON configuration."
    )
    parser.add_argument(
        "json_config_file",
        type=str,
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()
    logging.info(f"Script started with arguments: {args}")

    try:
        with open(args.json_config_file, "r") as f_json:
            config_data = json.load(f_json)
        logging.info(f"Successfully loaded JSON configuration from {args.json_config_file}")
        _validate_config(config_data)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.json_config_file}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file: {e}")
        return
    except ValueError as e: # Catches validation errors from _validate_config
        logging.error(f"Invalid configuration: {e}")
        return
    
    try:
        original_to_yolo_ids, yolo_id_to_names = map_classes(config_data["Classes"])
        logging.info(f"Class mapping successful. YOLO class names: {yolo_id_to_names}")
    except (TypeError, ValueError) as e:
        logging.error(f"Failed to map classes: {e}")
        return

    base_data_path = config_data["PathToDataFolders"]
    yolo_output_root = config_data["OutputDir"]
    resize_images = config_data["ResizeImages"]
    subsample_rate = config_data.get("SubsampleRate") # Can be None if not present

    if subsample_rate:
        logging.info(f"Subsampling enabled: keeping 1 of every {subsample_rate} frames.")
    
    # Prepare target_size_tuple once
    target_size_tuple: Optional[Tuple[int, int]] = None
    if resize_images:
        # Validation ensures TargetSize exists and is correct if ResizeImages is true
        target_size_list = config_data["TargetSize"]
        target_size_tuple = (target_size_list[0], target_size_list[1])

    processed_folders_count = 0
    for folder_name, dimensions in config_data["Folders"].items():
        logging.info(f"--- Processing dataset folder: {folder_name} ---")
        
        img_dir = os.path.join(base_data_path, folder_name, "img1")
        annotations_file = os.path.join(base_data_path, folder_name, "gt", "gt.txt")
        
        original_img_width, original_img_height = dimensions
        
        # Each sub-dataset (folder) will go into its own subdirectory within the main OutputDir
        dataset_specific_output_dir = os.path.join(yolo_output_root, f"{folder_name}_yolo")
        
        logging.info(f"Source image directory: {img_dir}")
        logging.info(f"Source annotations file: {annotations_file}")
        logging.info(f"Output directory for this dataset: {dataset_specific_output_dir}")
        logging.info(f"Original image dimensions: {original_img_width}x{original_img_height}")
        logging.info(f"Resize images: {resize_images}")
        if resize_images:
            logging.info(f"Target size for resize: {target_size_tuple}")
        if subsample_rate:
            logging.info(f"Subsampling rate: {subsample_rate}")
        
        try:
            convert_to_yolo(
                img_dir=img_dir,
                annotations_file=annotations_file,
                output_prefix=folder_name, # Prefix for files within this dataset's output
                output_dir=dataset_specific_output_dir,
                original_img_width=original_img_width,
                original_img_height=original_img_height,
                mapped_dict=original_to_yolo_ids,
                resize_img=resize_images,
                target_size=target_size_tuple,
                subsample_rate=subsample_rate
            )
            processed_folders_count += 1
            logging.info(f"Successfully converted folder '{folder_name}'.")
        except FileNotFoundError as e:
            logging.error(f"Skipping folder '{folder_name}': Required file/directory not found. {e}")
        except ValueError as e:
            logging.error(f"Skipping folder '{folder_name}': Invalid value encountered. {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while converting folder '{folder_name}': {e}", exc_info=True)
        logging.info(f"--- Finished processing dataset folder: {folder_name} ---")

    if processed_folders_count > 0:
        # Create a global data.yaml for the entire converted dataset
        try:
            # List of relative paths for the data.yaml, e.g., MOT20-01_yolo/images
            # These paths are relative to the location of the dataset.yaml file.
            train_image_paths = [
                os.path.join(f"{folder_name}_yolo", "images") for folder_name in config_data["Folders"]
                # Filter out folders that might have failed, if necessary, or assume all are present
            ]
            # For simplicity, using all converted folders for both train and val.
            # A more sophisticated setup might involve splitting.
            
            data_yaml_content = {
                'path': os.path.abspath(yolo_output_root), # Absolute path to the dataset root
                'train': train_image_paths, 
                'val': train_image_paths,   # Or specify a dedicated validation set
                'test': '', # Optional: path to test images
                'nc': len(yolo_id_to_names),
                'names': [yolo_id_to_names[i] for i in sorted(yolo_id_to_names.keys())]
            }
            
            data_yaml_path = os.path.join(yolo_output_root, "dataset.yaml")
            with open(data_yaml_path, 'w') as f_yaml:
                # Using json.dump for simplicity; for true YAML, use PyYAML (import yaml)
                # yaml.dump(data_yaml_content, f_yaml, sort_keys=False, default_flow_style=False)
                json.dump(data_yaml_content, f_yaml, indent=4) # JSON is a subset of YAML
            logging.info(f"Successfully created dataset configuration: {data_yaml_path}")
        except Exception as e:
            logging.error(f"Failed to create dataset.yaml: {e}", exc_info=True)
    else:
        logging.warning("No folders were processed successfully. Skipping dataset.yaml generation.")

    logging.info("Script execution finished.")

if __name__ == "__main__":
    main()
   