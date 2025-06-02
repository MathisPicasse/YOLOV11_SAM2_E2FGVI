"""
File: entity.py
Author: Mathis Picasse
Created: 2025-06-01
Last Modified: 2025-06-01
Description: Define the functions useful for the visualization on the video
"""

import cv2
import numpy as np
from modules.data.observations import Observation

def draw_bbox(frame, obs): 
    x_min, y_min, x_max, y_max = obs.bbox.tolist()
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.putText(frame, f"ID {obs.class_id}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_mask(frame, mask): 
    # Masque coloré vert semi-transparent
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 1] = mask * 255
    return cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
