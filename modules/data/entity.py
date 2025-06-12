"""
File: entity.py
Author: Mathis Picasse
Created: 2025-05-31
Last Modified: 2025-05-31
Description: Defines the types of entities (e.g., human, dog, bicycle) detectable in the video.
"""

from .observations import Observation
from typing import List, Dict, Any

class EntityType:
    """Represents a type/category of entity detectable in a video stream.

    Attributes:
        entity_name (str): Name of the entity type (e.g., "human", "dog").
        class_id (int): Numeric identifier for the entity type.
        expected_attributes (List[str]): List of expected attribute names for this entity.
    """
    def __init__(self, entity_name: str, class_id: int, expected_attributes: List[str]):
        self._entity_name = entity_name
        self._class_id = class_id
        self._expected_attributes = expected_attributes

    @property
    def entity_name(self) -> str:
        return self._entity_name

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def expected_attributes(self) -> List[str]:
        return self._expected_attributes


class Entity:
    """Represents an instance of an entity detected in video.

    Attributes:
        entity_id (int): Unique identifier for the entity instance.
        entity_type (EntityType): The type/category of the entity.
        tracks (List[Observation]): List of observations (detections) over time.
        attributes (Dict[str, Any]): Additional attributes describing the entity.
    """
    def __init__(self, entity_id: int, 
                 entity_type: EntityType, 
                 tracks: List[Observation] = None, 
                 attributes: Dict[str, Any] = None):
        self._entity_id = entity_id
        self._entity_type = entity_type
        self._tracks = tracks if tracks is not None else []
        self._attributes = attributes if attributes is not None else {}

    @property
    def entity_id(self) -> int:
        return self._entity_id

    @property
    def entity_type(self) -> EntityType:
        return self._entity_type

    @property
    def tracks(self) -> List[Observation]:
        return self._tracks

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes
    
    def __repr__(self) -> str:
        return (
            f"Entity(\n"
            f"  id={self._entity_id},\n"
            f"  type={self._entity_type.entity_name},\n"
            f"  tracks={len(self._tracks)},\n"
            f"  attributes={list(self._attributes.keys())}\n"
            f")"
        )


    def add_observation(self, observation: Observation, verbose: bool = False) -> None:
        """Adds an observation to the entity's track list.

        Args:
            observation (Observation): Observation instance to add.
            verbose (bool): If True, prints debug information.

        Returns:
            None
        """

        if not isinstance(observation, Observation):
            raise TypeError("Expected an Observation instance.")
        
        self._tracks.append(observation)

        if verbose:
            print(f"Added observation to entity {self._entity_id}: {observation}")
        
        