"""
File: observations.py
Author: Mathis Picasse
Created: 2025-05-31
Last Modified: 2025-05-31
Description: Defines the Observation class representing a single detection in a video frame.
"""

from typing import Tuple

class Observation:
    """Represents a single detection of an entity in a video frame.

    Attributes:
        frame_id (int): Index of the video frame where the detection occurred.
        class_id (int): Identifier of the detected class.
        bbox (Tuple[float, float, float, float]): Bounding box (x, y, width, height) of the detection.
        confidence (float): Confidence score of the detection.
    """

    def __init__(
        self, 
        frame_id: int, 
        class_id: int, 
        bbox: Tuple[float, float, float, float], 
        confidence: float
    ):
        self._frame_id = frame_id
        self._class_id = class_id
        self._bbox = bbox
        self._confidence = confidence

    @property 
    def frame_id(self) -> int:
        return self._frame_id

    @property 
    def class_id(self) -> int:
        return self._class_id

    @property 
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._bbox

    @property 
    def confidence(self) -> float:
        return self._confidence 
    
    def __repr__(self) -> str:
        return (
            f"Observation("
            f"frame_id={self._frame_id}, "
            f"class_id={self._class_id}, "
            f"bbox={self._bbox}, "
            f"confidence={self._confidence:.2f}"
            f")"
        )
