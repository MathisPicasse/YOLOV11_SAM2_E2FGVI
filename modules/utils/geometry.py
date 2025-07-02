"""
File: file.py
Author: Mathis Picasse
Description: useful functions linked to geometry and especially bounding boxes.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rescale_bbox_coordinates(
    x: float,
    y: float,
    w_box: float,
    h_box: float,
    img_width: int,
    img_height: int,
    new_img_width: int,
    new_img_height: int
) -> Tuple[float, float, float, float]:
    """
    Resizes bounding box coordinates and dimensions to a new image size.

    Calculates the scaling factors based on the old and new image dimensions
    and applies them to the bounding box's top-left coordinates (x, y)
    and its width and height.

    Args:
        x (float): X-coordinate of the top-left corner of the bounding box
                   in the original image.
        y (float): Y-coordinate of the top-left corner of the bounding box
                   in the original image.
        w_box (float): Width of the bounding box in the original image.
        h_box (float): Height of the bounding box in the original image.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.
        new_img_width (int): Width of the resized image.
        new_img_height (int): Height of the resized image.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the rescaled
        (x_new, y_new, w_box_new, h_box_new).

    Raises:
        ValueError: If original image width or height is negative, as dimensions
                    cannot be negative.
        ValueError: If new image width or height is negative, as dimensions
                    cannot be negative.
    """
    if img_width < 0 or img_height < 0:
        raise ValueError("Original image width and height must be non-zero.")
    if new_img_width < 0 or new_img_height <0:
        raise ValueError("New image width and height must be non-zero.")

    x_scale: float = new_img_width / img_width
    y_scale: float = new_img_height / img_height

    x_new: float = x * x_scale
    y_new: float = y * y_scale
    w_box_new: float = w_box * x_scale
    h_box_new: float = h_box * y_scale

    return x_new, y_new, w_box_new, h_box_new


def normalize_bbox_coordinates(
    x: float,
    y: float,
    w_box: float,
    h_box: float,
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Normalizes bounding box coordinates to be relative to image size.

    The normalization converts the bounding box (defined by its top-left corner
    and dimensions) into a format where coordinates and dimensions are
    relative to the image dimensions (values between 0 and 1).
    The output x and y are the center of the bounding box.

    Args:
        x (float): X-coordinate of the top-left corner of the bounding box.
        y (float): Y-coordinate of the top-left corner of the bounding box.
        w_box (float): Width of the bounding box.
        h_box (float): Height of the bounding box.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the normalized
        (x_center_norm, y_center_norm, w_box_norm, h_box_norm).

    Raises:
        ValueError: If original image width or height is negative, as dimensions
                    cannot be negative.
        
    """
    if img_width == 0 or img_height == 0:
        raise ValueError("Image width and height must be non-zero for normalization.")

    # Calculate the center of the bounding box
    x_center: float = x + (w_box / 2)
    y_center: float = y + (h_box / 2)

    # Normalize coordinates and dimensions
    x_center_norm: float = x_center / img_width
    y_center_norm: float = y_center / img_height
    w_box_norm: float = w_box / img_width
    h_box_norm: float = h_box / img_height

    # return x_center_norm, y_center_norm, w_box_norm, h_box_norm

    x_center_norm_clamped = max(0.0, min(1.0, x_center_norm))
    y_center_norm_clamped = max(0.0, min(1.0, y_center_norm))
    w_box_norm_clamped = max(0.0, min(1.0, w_box_norm))
    h_box_norm_clamped = max(0.0, min(1.0, h_box_norm))

    return x_center_norm_clamped, y_center_norm_clamped, w_box_norm_clamped, h_box_norm_clamped

def denormalize_bbox_coordinates(
    x_center_norm: float,
    y_center_norm: float,
    w_box_norm: float,
    h_box_norm: float,
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Converts normalized bounding box coordinates back to absolute image coordinates.

    This function takes normalized bounding box center coordinates and dimensions
    (values between 0 and 1) and converts them back to absolute pixel values
    for the top-left corner (x, y), width, and height.

    Args:
        x_center_norm (float): Normalized X-coordinate of the bounding box center.
        y_center_norm (float): Normalized Y-coordinate of the bounding box center.
        w_box_norm (float): Normalized width of the bounding box.
        h_box_norm (float): Normalized height of the bounding box.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the denormalized
        top-left (x, y) coordinates, width (w_box), and height (h_box).

    Raises:
        ValueError: If original image width or height is negative, as dimensions
                    cannot be negative.
        
    """
    if img_width < 0 or img_height < 0:
        raise ValueError("Image width and height must be non-negative.")

    # Denormalize width and height
    w_box: float = w_box_norm * img_width
    h_box: float = h_box_norm * img_height

    # Denormalize center coordinates
    x_center: float = x_center_norm * img_width
    y_center: float = y_center_norm * img_height

    # Convert center coordinates to top-left coordinates
    x: float = x_center - (w_box / 2)
    y: float = y_center - (h_box / 2)

    return x, y, w_box, h_box

