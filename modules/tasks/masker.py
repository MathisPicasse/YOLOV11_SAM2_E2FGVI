"""
File: mask.py
Author: Mathis Picasse
Created: 2025-06-02
Last Modified: 2025-06-02
Description: Implements the Masker class to generate masks for detected objects (after tracking)
"""

import os
import logging
import cv2
import numpy as np
from typing import List
from ultralytics import SAM
from modules.data.observations import Observation

# === Logger configuration ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Masker:
    """
    Masker class to handle loading of a SAM model and creating segmentation masks for detected objects.
    """
    models_directory = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/Masks"

    def __init__(self, model_name: str):
        """
        Initialize the masker by loading the specified SAM model.
        """
        self._model = None
        self.model = model_name  # triggers the setter

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_name: str):
        """
        Loads the SAM model from disk.
        """
        path_to_model = os.path.join(self.models_directory, model_name)
        if not os.path.isfile(path_to_model):
            logger.error(f"Model file not found at: {path_to_model}")
            raise FileNotFoundError(f"Semantic model file does not exist: {path_to_model}")
        
        try:
            logger.info(f"Loading SAM model from: {path_to_model}")
            self._model = SAM(path_to_model)
        except Exception as e:
            logger.exception("Error while loading SAM model")
            raise RuntimeError("Failed to load semantic model") from e

    def info(self) -> str:
        """
        Returns model information.
        """
        return self._model.info()

    def create_mask(self, frame: np.ndarray, observations: List[Observation], output_path: str, mask_name) -> np.ndarray:
        """
        Generates a binary mask for the given observation (bounding box).
        
        Args:
            frame: The original video frame (H, W, 3).
            obs: The Observation object with bbox and frame_id.

        Returns:
            mask: Binary mask as a NumPy array (H, W) with values 0 or 1.
        """
        bboxes = [obs.bbox.tolist() for obs in observations]   
        print(bboxes)

        # Run SAM model on the frame with the provided bbox
        try:
            results = list(self._model(frame, bboxes=bboxes))
        except Exception as e:
            logger.error("Error during SAM inference")
            raise RuntimeError("Failed to compute segmentation mask") from e
        
        if not results:
            logger.warning("SAM model returned no results list.")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        mask_obj = results[0].masks

        if mask_obj is None or mask_obj.data is None:
            frame_id_info = f"frame {observations[0].frame_id}" if observations else "unknown frame"
            logger.warning(f"No mask data found by SAM for {frame_id_info} and bboxes {bboxes}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        mask_np = mask_obj.data.cpu().numpy() # Shape (N, H, W), values are float (0.0 to 1.0)
        
        if mask_np.ndim != 3 or mask_np.shape[0] == 0:
            frame_id_info = f"frame {observations[0].frame_id}" if observations else "unknown frame"
            logger.warning(
                f"Empty or malformed mask tensor returned by SAM for {frame_id_info} and bboxes {bboxes}. Shape: {mask_np.shape}"
            )
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Combine all masks by taking the maximum intensity at each pixel (union of masks)
        combined_mask_float = np.max(mask_np, axis=0) # Result is (H, W)
        # Convert to binary mask (0 or 255) and uint8 type
        mask = (combined_mask_float * 255).astype(np.uint8)




        # mask_obj = results[0].masks
        # mask_np = mask_obj.data.cpu().numpy()
        
        # if mask_np.shape[0] == 0:
        #     logger.warning("Empty mask returned")
        #     return np.zeros(frame.shape[:2], dtype=np.uint8)

        # # Take the first mask in the batch
        # #mask = (mask_np[0] * 255).astype(np.uint8)
        print(mask)
        if not os.path.exists(output_path):
            print("created the folder")
            os.makedirs(output_path)
        save = cv2.imwrite(f"{output_path}/{mask_name}", mask)
        print("Mask saved:", save)
        if not save:
            print("Failed to save mask.")
        else:
            print("mask saved")
        return mask
        
    
    @classmethod
    def convert_to_color_mask(cls, mask):
        logger.debug(f"Converting mask with shape {mask.shape}")
        height, width = mask.shape

        mask_color = np.zeros((height, width, 3))
        logger.debug(f"mask color with 0 : {mask_color}")
        for y in range(height):
            for x in range(width):
                if mask[y, x]==255:
                    logger.debug("modified color")
                    mask_color[y, x,1]=128
        return mask_color
    