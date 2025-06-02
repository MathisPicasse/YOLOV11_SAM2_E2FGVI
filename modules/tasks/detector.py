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
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        _raw_results (Optional[list]): Raw tracking results after processing.
    """
    def __init__(self, detector: BaseDetector, tracker: BaseTracker):
        """
        Args:
            detector (BaseDetector): A detection model (e.g., YOLO).
            tracker (BaseTracker): A tracking algorithm (e.g., ByteTrack).
        """
        self._detector = detector
        self._tracker = tracker
        self._raw_results = None

    @property
    def detector(self) -> BaseDetector:
        """Returns the detector instance."""
        return self._detector

    @property
    def tracker(self) -> BaseTracker:
        """Returns the tracker instance."""
        return self._tracker

    @property
    def raw_results(self) -> Optional[list]:
        """Returns the raw tracking results."""
        return self._raw_results

    def track(self, video):
        """Runs detection and tracking on the input video.

        Args:
            video: Input video file path or frame stream.

        Returns:
            List of tracking results per frame.
        """
        self._raw_results = self.tracker.track(video, self.detector.model)
        return self._raw_results

    
    def format_results(self)->Dict[int, Observation]:
        formatted_results = dict()
        for frame_id, results in enumerate(self._raw_results): 
            for entity_id, class_id, bbox, confidence in zip(results.boxes.id, results.boxes.cls, results.boxes.xyxy, results.boxes.conf):
                new_observation = Observation(frame_id, class_id, bbox, confidence)
                if entity_id.item() in formatted_results:
                    logging.info(f"entity {entity_id} detected again in frame {frame_id}")
                    formatted_results[entity_id.item()].append(new_observation)
                else:
                    logging.info(f"new entity (id: {entity_id}) detected in frame {frame_id}")
                    formatted_results[entity_id.item()] = [new_observation]
        return formatted_results