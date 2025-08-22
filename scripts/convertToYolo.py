# ========================================
# Author: Mathis Picasse
# Created on: 04-2025
# Last Modified: 22-08-2026
# Description: Script to convert a MOT dataset to a Yolo dataset.
# ========================================

import json
import csv
from pathlib import Path
import yaml
import argparse
from typing import Dict, Tuple, Union, Optional, List, Any
from modules.utils.file import setup_output_dirs
from modules.utils.image import process_image
from modules.utils.geometry import rescale_bbox_coordinates, normalize_bbox_coordinates
from modules.utils.logger_setup import logger


def map_classes(
    mot_classes: Dict[Union[str, int], str]
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Maps original class IDs to new sequential IDs, merging classes with the same name.

    This function takes a dictionary of original class IDs (which can be
    strings or integers representing class identifiers) and their corresponding
    class names. It produces two dictionaries:
    1. A mapping from the original class ID (coerced to an integer) to a new,
       zero-based sequential integer ID. If multiple original IDs share the
       same class name, they will all map to the same new ID.
    2. A mapping from this new sequential ID to the unique class name.

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
              sequential IDs to their unique class names.

    Raises:
        TypeError: If `mot_classes` is not a dictionary.
    """
    if not isinstance(mot_classes, dict):
        raise TypeError("Input 'mot_classes' must be a dictionary.")

    # Get unique class names and sort them for a reproducible order.
    unique_class_names = sorted(list(set(mot_classes.values())))
    
    # Create the mapping from class names to new, sequential IDs.
    # If 'unique_class_names' was ['bicycle', 'car', 'person']
    # 'name_to_new_id_map' will be: {'bicycle': 0, 'car': 1, 'person': 2}
    name_to_new_id_map: Dict[str, int] = {name: i for i, name in enumerate(unique_class_names)}
    
    # Create the mapping from original IDs to new IDs.
    # This dictionary will store the final mapping that the main script needs.
    original_to_new_id_map: Dict[int, int] = {}
    
    for original_id_key, class_name in mot_classes.items():
        # Convert the original ID key to an integer.
        original_id_int = int(original_id_key)
        
        # Look up the new YOLO ID for the current class name and store the mapping.
        original_to_new_id_map[original_id_int] = name_to_new_id_map[class_name]

    # Create the reverse mapping from new IDs to class names.
    new_id_to_name_map: Dict[int, str] = {new_id: name for name, new_id in name_to_new_id_map.items()}

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
        
        # The script skips the annotation if the ID is not in the mapping dictionary.
        if original_class_id not in mapped_dict:
            logger.debug(f"Skipping class ID {original_class_id} not in mapped_dict for frame {frame}.")
            return None
        
        # Getting the new id
        yolo_class_id = mapped_dict[original_class_id]
        
        # Extract and convert the bounding box coordinates.
        # Coordinates are in top-left format (x_tl, y_tl, w_box, h_box).
        x_tl, y_tl, w_box, h_box = map(float, (row[2], row[3], row[4], row[5]))

        # Initialize variables for normalization.
        coords_to_normalize = (x_tl, y_tl, w_box, h_box)
        dims_to_normalize = (original_img_width, original_img_height)
       
        if resize_img and target_size:
            if original_img_width <= 0 or original_img_height <= 0:
                logger.error(f"Frame {frame}: Original image dimensions for rescaling must be positive.")
                return None

            # Coordinates are rescaled based on the new target size.
            coords_to_normalize = rescale_bbox_coordinates(
                x_tl, 
                y_tl, 
                w_box, 
                h_box,
                *dims_to_normalize, 
                *target_size
            )
            
            # The dimensions for normalization become those of the target image.
            dims_to_normalize = target_size
        
        # Normalize the final coordinates.
        x_center_norm, y_center_norm, w_box_norm, h_box_norm = normalize_bbox_coordinates( 
            *coords_to_normalize,
            *dims_to_normalize
        )

        return frame, yolo_class_id, x_center_norm, y_center_norm, w_box_norm, h_box_norm
    
    # Error handling for row parsing
    except (IndexError, ValueError) as e:
        logger.warning(f"Skipping malformed annotation row: {row}. Error: {e}")
        return None

def create_label_file(file_path: str, annotations: Optional[List[Any]] = None):
    try:
        with open(file_path, 'w') as label_file:
            if annotations:
                for annotation_data in annotations:
                    label_file.write(" ".join(map(str, annotation_data)) + "\n")
        logger.debug(f"Saved YOLO annotations to: {file_path}")
        return True
    except IOError as e:
        logger.error(f"Could not write label file {file_path}: {e}")
        return False

def convert_to_yolo(
    img_dir: str,
    annotations_file: str,
    output_prefix: str,
    output_dir: str,
    original_img_width: int,
    original_img_height: int,
    mapped_dict: Dict[int, int],
    resize_img: bool = False,
    target_size: Optional[Tuple[int, int]] = (640, 640),
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
    logger.info(f"Starting YOLO conversion for image directory: {img_dir}")

    img_dir_path = Path(img_dir)
    annotations_file_path = Path(annotations_file)
    
    # 1. Checking parameters and input paths conformity
    if not img_dir_path.exists(): 
        raise FileNotFoundError(f"Image directory not found: {img_dir_path}")
    if not annotations_file_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file_path}")

    if not resize_img and (original_img_width <= 0 or original_img_height <= 0):
        raise ValueError(
            "Original image width and height must be positive if not resizing."
        )
        
    if resize_img and not target_size:
        raise ValueError("target_size must be provided and valid if resize_img is True.")

    # 2. Setting up output directories
    dir_paths = setup_output_dirs(output_dir, ['images', 'labels'])
    images_output_dir = Path(dir_paths['images'])
    labels_output_dir = Path(dir_paths['labels'])

    # 3. Reading and processing annotations
    annotations_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    logger.info(f"Reading annotations from: {annotations_file_path}")
    
    try:
        # Opening the single annotation file with error handling
        with open(annotations_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader: # The index 'i' is not used here, so it can be ignored
                parsed_annotation = _parse_annotation_row(
                    row, 
                    mapped_dict, 
                    original_img_width, 
                    original_img_height,
                    resize_img, 
                    target_size
                )
                # Add the annotation to the list for the corresponding frame
                if parsed_annotation:
                    frame, class_id, x_c, y_c, w_n, h_n = parsed_annotation
                    if frame not in annotations_by_frame:
                        annotations_by_frame[frame] = []
                    annotations_by_frame[frame].append((class_id, x_c, y_c, w_n, h_n))
    except IOError as e:
        logger.error(f"Could not read annotations file {annotations_file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing CSV file {annotations_file_path}: {e}")
        raise
    
    # 4. Processing images and writing label files
    logger.info(f"Processing images from: {img_dir_path}")
    processed_image_count = 0
    
    image_files = sorted([f for f in img_dir_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    for i, img_path in enumerate(image_files):
        # Handling subsampling
        if subsample_rate and (i % subsample_rate != 0):
            continue

        effective_target_size = target_size if resize_img else None
        
        # Processing and saving the image
        try:
            _ = process_image(
                img_path=img_path,
                output_dir=images_output_dir,
                output_prefix=output_prefix,
                target_size=effective_target_size
            )
            processed_image_count +=1
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Failed to process image {img_path}: {e}")
            continue 

        # Extracting the frame number from the filename
        try:
            frame_number_str = img_path.stem
            frame_number = int(frame_number_str)
        except ValueError:
            logger.warning(
                f"Could not extract frame number from image filename: {img_path.name}. "
                f"Skipping annotation saving for this image."
            )
            continue
        
        # Creating the label file path
        label_filename = f"{output_prefix}_{frame_number:06d}.txt"
        label_file_path = labels_output_dir / label_filename
        
        # writing the annotation in the annotation file or create it
        if frame_number in annotations_by_frame:
            yolo_annotations = annotations_by_frame[frame_number]
            create_label_file(label_file_path, yolo_annotations)
        else:
            logger.info(f"No annotations found for frame {frame_number} (image: {img_path.name}). Creating empty label file.")
            create_label_file(label_file_path)

    logger.info(
        f"YOLO conversion completed for {output_prefix}. Processed {processed_image_count} images. "
        f"Output in: {output_dir}"
    )

def validate_config(config: Dict[str, Any]) -> None: 
    """
    Validates the structure and content of a configuration dictionary.

    This function ensures that the JSON configuration dictionary contains
    all required keys and that their values are of the correct type and format.
    It is designed to be called at the beginning of the script to halt
    execution with an explicit error if the configuration is invalid.

    Args:
        config (Dict[str, Any]): The configuration dictionary loaded
                                 from a JSON file.

    Raises:
        ValueError: If a required key is missing, if a value is of the wrong type,
                    or if a validity condition is not met (e.g., non-positive
                    image dimensions or a key is missing when required).
    """
    
    required_top_level_keys = [
        "PathToDataFolders", "OutputDir", "Classes", "ResizeImages"
    ]
    required_folders_keys = ["train_folders"]
    optional_folders_keys = ["val_folders", "test_folders"]
    
    # 1. Validate top-level keys
    for key in required_top_level_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: '{key}' in JSON file.")
    
    if not isinstance(config["PathToDataFolders"], str):
        raise ValueError("'PathToDataFolders' must be a string.")
    if not isinstance(config["OutputDir"], str):
        raise ValueError("'OutputDir' must be a string.")
    if not isinstance(config["Classes"], dict):
        raise ValueError("'Classes' must be a dictionary in JSON config.")
    if not isinstance(config["ResizeImages"], bool):
        raise ValueError("'ResizeImages' must be a boolean.")

    # 2. Validate resizing and subsampling parameters
    if config["ResizeImages"]:
        if "TargetSize" not in config:
            raise ValueError("'TargetSize' is required when 'ResizeImages' is true.")
        if not (isinstance(config["TargetSize"], list) and len(config["TargetSize"]) == 2 and
                all(isinstance(dim, int) and dim > 0 for dim in config["TargetSize"])):
            raise ValueError("'TargetSize' must be a list of two positive integers [width, height].")
    
    if "SubsampleRate" in config:
        if not (isinstance(config["SubsampleRate"], int) and config["SubsampleRate"] > 1):
            raise ValueError("'SubsampleRate' must be an integer greater than 1.")
            
    # 3. Validate folder keys (train, val, test)
    # Check that the required keys are present and are dictionaries
    for key in required_folders_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: '{key}'.")
        if not isinstance(config[key], dict):
            raise ValueError(f"'{key}' must be a dictionary.")

    # Check that optional keys are dictionaries if they are present
    for key in optional_folders_keys:
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"'{key}' must be a dictionary.")
    
    # 4. Validate the dimensions of each folder
    for folder_key in required_folders_keys + [k for k in optional_folders_keys if k in config]:
        for folder_name, dims in config[folder_key].items():
            if not (isinstance(dims, list) and len(dims) == 2 and
                    all(isinstance(d, int) and d > 0 for d in dims)):
                raise ValueError(
                    f"Dimensions for folder '{folder_name}' in '{folder_key}' must be a list of two positive integers. Got: {dims}"
                )



def main():
    """
    Main function to orchestrate the dataset conversion process.
    Parses command-line arguments for the JSON configuration file,
    loads the configuration, and processes each specified dataset folder.
    """
    # 1. Argument Parsing and Configuration Loading
    parser = argparse.ArgumentParser(
        description="Converts MOT datasets to YOLO format based on a JSON configuration."
    )
    parser.add_argument(
        "json_config_file",
        type=str,
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()
    logger.info(f"Script started with arguments: {args}")

    # Load and validate the JSON configuration.
    try:
        with open(args.json_config_file, "r") as f_json:
            config_data = json.load(f_json)
        logger.info(f"Successfully loaded JSON configuration from {args.json_config_file}")
        # Assuming validate_config has been updated to handle the new structure
        validate_config(config_data)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.json_config_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON configuration file: {e}")
        return
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return
    
    # Map original class IDs to new sequential YOLO IDs.
    try:
        original_to_yolo_ids, yolo_id_to_names = map_classes(config_data["Classes"])
        logger.info(f"Class mapping successful. YOLO class names: {yolo_id_to_names}")
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to map classes: {e}")
        return

    # 2. Extract Key Configuration Parameters
    base_data_path = Path(config_data["PathToDataFolders"])
    yolo_output_root = Path(config_data["OutputDir"])
    resize_images = config_data["ResizeImages"]
    subsample_rate = config_data.get("SubsampleRate")

    if subsample_rate:
        logger.info(f"Subsampling enabled: keeping 1 of every {subsample_rate} frames.")
    
    target_size_tuple: Optional[Tuple[int, int]] = None
    if resize_images:
        target_size_list = config_data["TargetSize"]
        target_size_tuple = (target_size_list[0], target_size_list[1])

    # 3. Main Loop: Process Each Dataset Split (train, val, test)
    # Use a dictionary to manage folders by split type. Use .get() for optional keys.
    all_folders = {
        "train": config_data.get("train_folders", {}),
        "val": config_data.get("val_folders", {}),
        "test": config_data.get("test_folders", {})
    }
    
    processed_folders_count = 0
    for split_name, folders in all_folders.items():
        if not folders:
            continue  
            
        logger.info(f"--- Processing {split_name} datasets ---")
        for folder_name, dimensions in folders.items():
            # Construct file paths using the Path object and the split name.
            img_dir = base_data_path / split_name / folder_name / "img1"
            annotations_file = base_data_path / split_name / folder_name / "gt" / "gt.txt"
            original_img_width, original_img_height = dimensions
            
            # Each sub-dataset will go into its own subdirectory within the split's output folder.
            dataset_specific_output_dir = yolo_output_root / split_name / f"{folder_name}_yolo"
                    
            logger.info(f"Source image directory: {img_dir}")
            logger.info(f"Source annotations file: {annotations_file}")
            logger.info(f"Output directory for this dataset: {dataset_specific_output_dir}")
            logger.info(f"Original image dimensions: {original_img_width}x{original_img_height}")
            logger.info(f"Resize images: {resize_images}")
            if resize_images:
                logger.info(f"Target size for resize: {target_size_tuple}")
            if subsample_rate:
                logger.info(f"Subsampling rate: {subsample_rate}")
            
            try:
                # Call the main conversion function. Note: The arguments are converted to strings
                # to match the function signature, as pathlib objects are not always compatible.
                convert_to_yolo(
                    img_dir=str(img_dir),
                    annotations_file=str(annotations_file),
                    output_prefix=folder_name,
                    output_dir=str(dataset_specific_output_dir),
                    original_img_width=original_img_width,
                    original_img_height=original_img_height,
                    mapped_dict=original_to_yolo_ids,
                    resize_img=resize_images,
                    target_size=target_size_tuple,
                    subsample_rate=subsample_rate
                )
                processed_folders_count += 1
                logger.info(f"Successfully converted folder '{folder_name}'.")
            except FileNotFoundError as e:
                logger.error(f"Skipping folder '{folder_name}': Required file/directory not found. {e}")
            except ValueError as e:
                logger.error(f"Skipping folder '{folder_name}': Invalid value encountered. {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while converting folder '{folder_name}': {e}", exc_info=True)
            logger.info(f"--- Finished processing dataset folder: {folder_name} ---")

    # 4. Final step: Create the global dataset.yaml file
    if processed_folders_count > 0:
        try:
            # Build the lists of relative paths for each split
            train_image_paths = [
                str(Path("train") / f"{folder_name}_yolo" / "images") 
                for folder_name in config_data.get("train_folders", {})
            ]
            val_image_paths = [
                str(Path("val") / f"{folder_name}_yolo" / "images") 
                for folder_name in config_data.get("val_folders", {})
            ]
            test_image_paths = [
                str(Path("test") / f"{folder_name}_yolo" / "images") 
                for folder_name in config_data.get("test_folders", {})
            ]
            
            data_yaml_content = {
                'path': str(yolo_output_root.resolve()),
                'train': train_image_paths, 
                'val': val_image_paths,
                'test': test_image_paths,
                'nc': len(yolo_id_to_names),
                'names': [yolo_id_to_names[i] for i in sorted(yolo_id_to_names.keys())]
            }
            
            data_yaml_path = yolo_output_root / "dataset.yaml"
            with open(data_yaml_path, 'w') as f_yaml:
                yaml.dump(data_yaml_content, f_yaml, sort_keys=False)
            logger.info(f"Successfully created dataset configuration: {data_yaml_path}")
        except Exception as e:
            logger.error(f"Failed to create dataset.yaml: {e}", exc_info=True)
    else:
        logger.warning("No folders were processed successfully. Skipping dataset.yaml generation.")

    logger.info("Script execution finished.")

if __name__ == "__main__":
    main()
    
    