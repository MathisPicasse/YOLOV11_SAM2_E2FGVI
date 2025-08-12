# ========================================
# Author: Mathis Picasse
# Created on: 12th, August 2025
# Description: Contains the functions related to the preprocessing module.
# ========================================

from typing import Optional, Tuple
from modules.utils.logger_setup import logger
from modules.utils.video import create_frames, create_video, get_fps
import os


def preprocess_video(
    video_path: str,
    output_frame_dir: str,
    output_video_path: str,
    frame_prefix: str,
    target_size: Optional[Tuple[int, int]] = None,
    fps: Optional[int] = None,
    codec: str = 'mp4v'
) -> None:
    """
    Preprocesses a video by extracting frames and optionally resizing them,
    then creates a new video from the processed frames.

    This function first extracts all frames from the input video, saving them
    as individual image files with a specified prefix. It can optionally resize
    these frames to a target size during extraction. Finally, it compiles
    these processed frames back into a new video file.

    Args:
        video_path (str): The full path to the input video file.
        output_frame_dir (str): The directory where the extracted and
                                processed frames will be saved.
        output_video_path (str): The full path (including filename) for the
                                 output video created from the frames.
        frame_prefix (str): A string prefix to use for naming the output
                            frame files (e.g., 'frame').
        target_size (Optional[Tuple[int, int]]): An optional tuple (width, height)
            to resize the frames to during extraction. If None, frames are
            saved with their original dimensions. Defaults to None.
        fps (Optional[int]): The frame rate (frames per second) for the output video.
            If None, the original video's FPS is used. Defaults to None.
        codec (str): The four-character code (FourCC) for the video codec
                     (e.g., 'mp4v', 'XVID'). Defaults to 'mp4v'.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        IOError: If the video file cannot be opened or read by OpenCV,
                 or if there are issues during frame extraction or video creation.
        ValueError: If no frames are found after extraction (indicating an issue
                    with the input video or extraction process).
    """
    logger.info(f"Starting video preprocessing for: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # ========================================
    # Create frames from the video and resize them
    # ========================================
    logger.info(
        f"Extracting frames to: {output_frame_dir} with prefix '{frame_prefix}' and target size: {target_size}")

    try:
        create_frames(output_frame_dir, frame_prefix,
                      video_path, resize_dim=target_size)
        logger.info("Frame extraction completed.")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed during frame extraction: {e}")
        raise IOError(f"Failed to extract frames from {video_path}") from e

    # ========================================
    # Get fps from the original video
    # ========================================
    output_fps = fps or get_fps(video_path)
    logger.info(f"Using FPS for output video: {output_fps}")

    # ========================================
    # Create a new video from the resized frames
    # ========================================
    logger.info(
        f"Creating output video at: {output_video_path} with codec '{codec}'")
    try:
        create_video(output_frame_dir, output_video_path,
                     frame_prefix, fps=output_fps, codec=codec)
        logger.info("Output video creation completed.")
    except (ValueError, IOError) as e:
        logger.error(f"Failed during output video creation: {e}")
        raise IOError(
            f"Failed to create output video at {output_video_path}") from e

    logger.info(f"Video preprocessing finished for: {video_path}")
