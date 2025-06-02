"""
File: tracker.py
Author: Mathis Picasse
Created: 2025-05-31
Last Modified: 2025-05-31
Description: Define the tracker classes with base abstract class and implementation for Ultralytics tracker.
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseTracker(ABC):
    """Abstract base class for tracker implementations."""

    @abstractmethod
    def track(self, source: Any, detections: Any) -> Any:
        """Run tracking on the given source using provided detections.

        Args:
            source: Video source or data stream.
            detections: Detection results from detector.

        Returns:
            Tracking results, type depends on implementation.
        """
        pass


class UltralyticsTracker(BaseTracker):
    """Tracker implementation using Ultralytics tracker.

    Attributes:
        _tracker_name (str): Name/configuration of the tracker.
    """

    def __init__(self, tracker_name: str):
        self._tracker_name = tracker_name

    @property
    def tracker_name(self) -> str:
        return self._tracker_name

    def track(self, source: Any, detector_model: Any) -> Any:
        """Track objects on the source video using the Ultralytics model and tracker.

        Args:
            source: Path or stream of video input.
            detector_model: An instance of YOLO model (or compatible) with `track` method.

        Returns:
            Tracking results object returned by `detector_model.track`.
        """
        return detector_model.track(source=source, tracker=self._tracker_name, show=False)

    def __repr__(self) -> str:
        return f"UltralyticsTracker(tracker_name='{self._tracker_name}')"
