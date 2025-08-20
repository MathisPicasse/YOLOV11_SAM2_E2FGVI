# ========================================
# Author: Mathis Picasse
# Created on: 2025-06-01
# Last Modified: 2025-06-01
# Description: Combines YOLO object detection and tracking into a single class.
# ========================================

from ultralytics import YOLO
from typing import Dict, List, Optional
from modules.data.observations import Observation
from modules.utils.logger_setup import logger

class DetectTrack:
    """
    High-level wrapper class combining YOLO object detection and tracking.

    This class allows running object detection on videos/images and tracks
    detected objects across frames, providing a structured output with
    observations grouped by entity ID.

    Attributes:
        _model (YOLO): YOLO model instance for detection and tracking.
        _tracker: Optional tracker configuration or tracker object.
        _raw_detections (list, optional): Stores raw tracking results per frame.
    """

    def __init__(self, detector: str, tracker: str):
        """
        Initializes the DetectTrack class with a YOLO model and tracker.

        Args:
            detector (str): Path to YOLO model weights (.pt file) or model identifier.
            tracker (str): Tracker configuration or tracker name (handled internally by YOLO).
        """
        self._model = YOLO(detector)
        self._tracker = tracker
        self._raw_detections = None

    @property
    def tracker(self) -> str:
        """Returns the tracker configuration used for tracking."""
        return self._tracker

    @property
    def raw_detections(self) -> Optional[list]:
        """Returns the raw detection/tracking results after processing a video."""
        return self._raw_detections

    def track(self, video: str, project: str) -> list:
        """
        Runs YOLO detection and tracking on the input video.

        Args:
            video (str): Path to the video file or stream.
            project (str): Directory path where tracking video will be saved.

        Returns:
            list: Raw detection/tracking results from YOLO.
        """
        self._raw_detections = self._model.track(
            source=video,
            project=project,
            show=True,  # Display the tracking live
            save=True,  # Save output video/images
        )
        return self._raw_detections

    def group_detections_by_entity(self) -> Dict[int, List[Observation]]:
        """
        Organizes raw detection results into a dictionary mapping entity IDs
        to their observations across frames.

        Each observation stores:
            - Frame index
            - Class ID
            - Bounding box coordinates
            - Confidence score

        Returns:
            Dict[int, List[Observation]]: 
                Key: entity_id (int) – Unique ID for each tracked object.
                Value: List of Observation objects for that entity across frames.

        Logging:
            - Logs when a new entity is detected.
            - Logs when an existing entity is detected again in subsequent frames.
        """
        detections_by_entity: Dict[int, List[Observation]] = dict()  # Initialize storage

        # Iterate frame by frame
        for frame_id, results in enumerate(self._raw_detections or []):
            # Iterate over all detected entities in the current frame
            for entity_id, class_id, bbox, confidence in zip(
                results.boxes.id,
                results.boxes.cls,
                results.boxes.xyxy,
                results.boxes.conf
            ):
                # Create a new Observation for this detection
                new_observation: Observation = Observation(
                    frame_id, class_id, bbox, confidence
                )

                entity_key: int = entity_id.item()  # Convert tensor to int
                if entity_key in detections_by_entity:
                    logger.info(f"entity {entity_id} in frame {frame_id}")
                    detections_by_entity[entity_key].append(new_observation)
                else:
                    logger.info(f"new entity (id: {entity_id}) in frame {frame_id}")
                    detections_by_entity[entity_key] = [new_observation]

        return detections_by_entity