"""
File: images.py
Author: Mathis Picasse
Description: useful functions to work with images.
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict, List
import cv2

from modules.utils.logger_setup import logger


def process_image(
    img_path: str,
    output_dir: str,
    output_prefix: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> str:
    """
    Loads an image, optionally resizes it, and saves it to a specified directory.

    This function provides a robust way to handle individual image files,
    ensuring that the source image exists and that the output directory is
    available before processing. It returns the full path to the newly created image.

    Args:
        img_path (str): The full path to the source image file.
        output_dir (str): The directory path where the processed image will be saved.
            The directory will be created if it does not exist.
        output_prefix (Optional[str]): A prefix to add to the new image filename.
            If None, the original filename is used. Defaults to None.
        target_size (Optional[Tuple[int, int]]): The target (width, height) to
            resize the image to. If None, the image is not resized. Defaults to None.

    Returns:
        str: The full path to the newly saved image file.

    Raises:
        FileNotFoundError: If the source image at `img_path` does not exist.
        IOError: If the image cannot be read by OpenCV or if the processed
            image cannot be written to the output directory.
    """

    # --- 1. Validate Inputs and Pre-conditions ---

    # Ensure the source image file exists before trying to read it.
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Source image not found at: {img_path}")

    # Ensure the output directory exists. This prevents errors during saving.
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Process the Image ---

    # Load the image from the specified path.
    img = cv2.imread(img_path)
    if img is None:
        # This occurs if the file is not a valid image format or is corrupted.
        raise IOError(f"Failed to read or decode the image at: {img_path}")

    # Resize the image only if a target size is provided.
    if target_size:
        logger.info(f"Resizing image to {target_size[0]}x{target_size[1]}...")
        try:
            img = cv2.resize(img, target_size)
        except cv2.error as e:
            # Catch potential OpenCV errors during resizing (e.g., invalid size).
            raise IOError(f"OpenCV failed to resize image {img_path}: {e}")

    # --- 3. Save the Result ---

    # Generate a robust and descriptive output filename.
    # Using os.path.splitext is safer than string splitting.
    original_basename = os.path.basename(img_path)
    base, ext = os.path.splitext(original_basename)
    
    if output_prefix:
        new_filename = f"{output_prefix}_{base}.jpg"
    else:
        new_filename = f"{base}.jpg"
    
    # Always save as .jpg for consistency, but this could be a parameter.
    new_img_path = os.path.join(output_dir, new_filename)

    # Save the processed image to the new path.
    success = cv2.imwrite(new_img_path, img)
    if not success:
        # This handles potential filesystem errors (e.g., permissions).
        raise IOError(f"Failed to write processed image to: {new_img_path}")

    logger.info(f"Successfully processed image and saved to: {new_img_path}")

    return new_img_path



def display_bbox_on_image(
    image: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    class_dict: Dict[int, str],
    class_objects: List[int],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    thickness: int = 2
) -> np.ndarray:
    
    """
    Displays bounding boxes on an image with corresponding class labels.

    This function draws rectangles for each bounding box and annotates them
    with class labels. It allows for custom colors per class. The input image
    is copied, so the original image is not modified.

    Args:
        image (np.ndarray): The image (OpenCV format, BGR) on which to draw.
        boxes (List[Tuple[float, float, float, float]]): A list of bounding boxes.
            Each box is (x_top_left, y_top_left, width, height) in float coordinates.
        class_dict (Dict[int, str]): Maps class IDs to class names.
            Example: {0: 'person', 1: 'car'}.
        class_objects (List[int]): List of class IDs, one for each box.
        colors (Optional[Dict[int, Tuple[int, int, int]]]): Maps class IDs
            to (B, G, R) color tuples. If None, or if a class_id is not in
            this map, a default color (green) is used.
        thickness (int): Thickness of the bounding box lines. Default is 2.

    Returns:
        np.ndarray: The image with bounding boxes and labels drawn.

    Raises:
        ValueError: If the number of boxes does not match the number of
                    class_objects.
        TypeError: If the image is not a numpy ndarray.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> bboxes = [(50.0, 50.0, 100.0, 150.0)]
        >>> cls_dict = {0: "cat"}
        >>> cls_objects = [0]
        >>> custom_colors = {0: (255, 0, 0)} # Blue for cat
        >>> annotated_img = display_bbox_on_image(img, bboxes, cls_dict, cls_objects, custom_colors)
        # annotated_img will have a blue bounding box with "cat" label.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy ndarray.")
    if len(boxes) != len(class_objects):
        raise ValueError("The number of boxes must match the number of class_objects.")

    # Create a mutable copy of the image to draw on, preventing modification of the original.
    output_image = image.copy()

    effective_colors: Dict[int, Tuple[int, int, int]] = {}
    if colors is not None:
        effective_colors.update(colors)

    default_color = (0, 255, 0) # Green (B, G, R)

    for (x_f, y_f, w_f, h_f), class_id in zip(boxes, class_objects):
        # Convert float coordinates to integers for OpenCV drawing functions
        x, y, w, h = int(x_f), int(y_f), int(w_f), int(h_f)

        if class_id in class_dict:
            color = effective_colors.get(class_id, default_color)
            class_label = class_dict[class_id]

            # Draw the bounding box
            # The bottom-right corner is (x + width, y + height)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)

            # Prepare text properties
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size to position it nicely
            (text_width, text_height), baseline = cv2.getTextSize(class_label, font, font_scale, font_thickness)

            # Position text background just above the bounding box
            # Ensure text_bg_y1 is not less than 0 (top of the image)
            text_bg_y1 = max(0, y - text_height - baseline - 2) # Small padding
            text_bg_y2 = y - baseline + 2

            # Adjust if text background would have been above the image or if there's no space
            if y - text_height - baseline - 2 < 0 : # If text would go off screen at the top
                # Place text background inside the top of the bounding box
                text_bg_y1 = y + baseline + 2
                text_bg_y2 = y + text_height + baseline * 2 + 2 # Adjusted for better padding
                text_y_pos = y + text_height + baseline # Text y position
            else:
                text_y_pos = y - baseline - 2 # Text y position

            # Draw background for the text
            cv2.rectangle(output_image, (x, text_bg_y1), (x + text_width, text_bg_y2), color, cv2.FILLED)
            # Draw the text (black text on colored background for contrast)
            cv2.putText(output_image, class_label, (x, text_y_pos), font, font_scale, (0,0,0), font_thickness)
        else:
            # Optionally, log a warning if a class_id is not found in class_dict
            logger.warning(f"Class ID {class_id} not found in class_dict. Skipping label for this box.")
    return output_image
         