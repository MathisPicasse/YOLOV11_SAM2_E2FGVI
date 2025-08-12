"""
File: entity.py
Author: Mathis Picasse
Created: 2025-06-01
Last Modified: 2025-06-01
Description: Defines the detector and the combination with tracker.
"""

from ultralytics import YOLO
from abc import ABC, abstractmethod
from .tracker import BaseTracker, UltralyticsTracker
from typing import Optional, Dict, List
from modules.data.observations import Observation
from modules.utils.logger_setup import logger


class BaseDetector(ABC):
    """Abstract base class for all object detectors.

    All detectors must implement the `detect` method and expose a `model` property.
    """
    @abstractmethod
    def detect(self, source):
        pass

    @property
    @abstractmethod
    def model(self):
        pass


class YoloDetector(BaseDetector):
    """Concrete implementation of BaseDetector using the YOLO model.

    Attributes:
        _model (YOLO): An instance of the Ultralytics YOLO detector.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the YOLO model weights (.pt file).
        """
        self._model = YOLO(model_path)

    def detect(self, source):
        """Runs YOLO detection on the input source.

        Args:
            source: Input for detection (image path, frame, video stream, etc.).

        Returns:
            Detection results.
        """
        return self._model(source)

    @property
    def model(self):
        """Returns the YOLO model instance."""
        return self._model


class Detector:
    """Encapsulates a detector and a tracker to perform video object tracking.

    Attributes:
        _detector (BaseDetector): The detector used to extract object detections.
        _tracker (BaseTracker): The tracker responsible for associating detections over time.
        _raw_detections (Optional[list]): Raw tracking results after processing.
    """

    def __init__(self, detector: BaseDetector, tracker: BaseTracker):
        """
        Args:
            detector (BaseDetector): A detection model (e.g., YOLO).
            tracker (BaseTracker): A tracking algorithm (e.g., ByteTrack).
        """
        self._detector = detector
        self._tracker = tracker
        self._raw_detections = None

    @property
    def detector(self) -> BaseDetector:
        """Returns the detector instance."""
        return self._detector

    @property
    def tracker(self) -> BaseTracker:
        """Returns the tracker instance."""
        return self._tracker

    @property
    def raw_detections(self) -> Optional[list]:
        """Returns the raw tracking results."""
        return self._raw_detections

    def track(self, video, project):
        """
        Runs detection and tracking on the input video.

        Args:
            video: Input video file path or frame stream.

        Returns:
            List of tracking results per frame.
        """
        self._raw_detections = self.tracker.track(
            video, self.detector.model, project)
        return self._raw_detections

    def group_detections_by_entity(self) -> Dict[int, Observation]:
        """
        Formats raw object detection results into a dictionary mapping entity IDs to their observations.

        This function processes the raw detection results stored in `self._raw_results`.
        For each frame, it iterates over all detected entities, extracting their ID,
        class, bounding box coordinates, and confidence score. Each detection is
        converted into an `Observation` object and added to the results dictionary.

        If an entity ID has been detected in previous frames, the new observation
        is appended to the existing list. Otherwise, a new list is created for that entity.

        Args:
            self: The class instance containing `_raw_results` as an iterable of detection outputs.
                Each detection output must have `boxes.id`, `boxes.cls`, `boxes.xyxy`, and `boxes.conf`.

        Returns:
            Dict[int, List[Observation]]:  
                A dictionary where:
                - Key: `entity_id` (int) – Unique identifier of the tracked object.
                - Value: List of `Observation` objects containing detection details for each frame.

        Logging:
            - Logs when an entity is detected for the first time in a frame.
            - Logs when an entity is detected again in subsequent frames.
        """

        # Store the observations with a list by entity_id -> Dict[int, List[Observation]]
        detections_by_entity = dict()

        # Iterating through the results frame by frame
        for frame_id, results in enumerate(self._raw_detections):
            # Iterating through the detected entities in the frame
            for entity_id, class_id, bbox, confidence in zip(
                results.boxes.id,
                results.boxes.cls,
                results.boxes.xyxy,
                results.boxes.conf
            ):
                # Create a new Observation object
                new_observation = Observation(
                    frame_id, class_id, bbox, confidence)

                # If the entity has been detected before, append the new observation to its existing list
                if entity_id.item() in detections_by_entity:
                    logger.info(f"entity {entity_id} in frame {frame_id}")
                    detections_by_entity[entity_id.item()].append(
                        new_observation)
                else:
                    # If this is a new entity, create a new entry in the dictionary with its first observation
                    logger.info(
                        f"new entity (id: {entity_id}) in frame {frame_id}"
                    )
                    detections_by_entity[entity_id.item()] = [new_observation]
        return detections_by_entity
