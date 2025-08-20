# ========================================
# Author: Mathis Picasse
# Created on: 2025-06-02
# Last Modified: 15-08-2025
# Description: Implement masking module
# ========================================

from modules.data.observations import Observation
import os
from modules.utils.logger_setup import logger
import cv2
import numpy as np
from typing import List
from ultralytics import SAM
from pathlib import Path
from config import MASK_PATH


class Masker:
    """
    Masker class to handle loading of a SAM model and creating segmentation masks for detected objects.
    
    Attributes:
        models_directory (str): Directory where SAM model files are stored.
        _model (SAM | None): Private attribute holding the loaded SAM model instance.
    """
    
    models_directory = MASK_PATH

    def __init__(self, model_name: str):
        """
        Initialize the masker by loading the specified SAM model.
    
        Args:
            model_name (str): File name of the SAM model to load from `models_directory`.
        """
        
        self._model = None
        self.model = model_name  # triggers the setter

    @property
    def model(self):
        """Return the loaded SAM model instance."""
        
        return self._model

    @model.setter
    def model(self, model_name: str):
        """
        Loads the SAM model from disk.
        
        Args:
            model_name (str): File name of the SAM model to load.
        
        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model fails to load.
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


    def create_mask(
        self, 
        frame: np.ndarray, 
        observations: List[Observation], 
        output_path: str, 
        mask_name: str
    ) -> np.ndarray: 
        """
        Generate a binary mask for a frame given detected observations.

        Args:
            frame: Original video frame (H, W, 3).
            observations: List of Observation objects containing bbox and frame_id.
            output_path: Directory where the mask will be saved.
            mask_name: File name for the saved mask.

        Returns:
            Binary mask as a NumPy array (H, W) with values 0 or 255.
        """

        # Extract bounding boxes from observations as a list of lists.
        bboxes = [obs.bbox.tolist() for obs in observations]
        
        try:
            results = list(self._model(frame, bboxes=bboxes))
        except Exception as e:
            logger.error("Error during SAM inference")
            raise RuntimeError("Failed to compute segmentation mask") from e

        # Handle case where SAM returns an empty results list.
        if not results:
            logger.warning("SAM model returned no results list.")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Container returned by SAM with differents mask representation
        masks_container = results[0].masks
        
        frame_id_info = f"frame {observations[0].frame_id}" if observations else "unknown frame"
    
        if masks_container is None or masks_container.data is None:
            logger.warning(f"No mask data found by SAM for {frame_id_info} and bboxes {bboxes}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Convert boolean mask tensor (N instances, H, W) to NumPy array
        mask_np = masks_container.data.cpu().numpy()
        
        # Validate mask dimensions and check for empty or malformed tensors.
        if mask_np.ndim != 3 or mask_np.shape[0] == 0:
            logger.warning(
                f"Empty or malformed mask tensor returned by SAM for {frame_id_info} and bboxes {bboxes}. Shape: {mask_np.shape}"
            )
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Combine all instance masks into a single mask by taking the maximum value per pixel (True > False)
        combined_mask_bool = np.max(mask_np, axis=0) 
        
        # Convert boolean mask to uint8 binary mask (0 for background, 255 for object)
        mask = (combined_mask_bool * 255).astype(np.uint8)

        logger.debug(f"Mask generated for {frame_id_info}: shape={mask.shape}, unique values={np.unique(mask)}")
        
        # Save the mask as an image file and confirm success.
        output_path = Path(output_path)
        success = cv2.imwrite(str(output_path / mask_name), mask)
        if not success:
            logger.error(f"Failed to save mask at {output_path / mask_name}")
        else:
            logger.info(f"Mask saved: {output_path / mask_name}")

        return mask
    
