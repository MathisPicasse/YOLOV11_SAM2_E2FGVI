"""
File: video.py
Author: Mathis Picasse
Description: useful functions to work with videos.
"""


import os
import cv2
from typing import Optional, Tuple, Dict
from modules.utils.geometry import denormalize_bbox_coordinates
from modules.utils.image import display_bbox_on_image
from modules.utils.logger_setup import logger

def create_frames(
    output_dir: str,
    frame_prefix: str,
    path_to_video: str,
    resize_dim: Optional[Tuple[int, int]] = None,
    target_fps: Optional[int] = None
) -> None:

    """Extracts frames from a video file and saves them as images.

    This function reads a video from the specified path, extracts each frame,
    optionally resizes it, and saves it as a JPEG image in the output
    directory. If `target_fps` is provided, it will subsample frames to
    approximate the target frame rate.

    Args:
        output_dir: The directory where the extracted frames will be saved.
            The directory will be created if it does not exist.
        frame_prefix: A string prefix for naming the output frame files (e.g., 'frame').
        path_to_video: The full path to the input video file.
        resize_dim: An optional tuple (width, height) to resize the frames.
            If None, frames are saved with their original dimensions.
        target_fps (Optional[int]): An optional integer for the desired frames per second.
            If set, frames will be skipped to match this rate. If None, all
            frames are extracted. Defaults to None.

    Raises:
        FileNotFoundError: If the specified `path_to_video` does not exist.
        IOError: If the video file cannot be opened or read by OpenCV.

    """

    if not os.path.exists(path_to_video):
        # Raise an exception for a non-existent file for clearer error handling.
        raise FileNotFoundError(f"The specified video path does not exist: {path_to_video}")

    # Ensure the output directory exists before saving frames.
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open the video file at: {path_to_video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        logger.warning(
            f"Video {path_to_video} has an invalid source FPS ({source_fps}). "
            "Cannot use target_fps for subsampling. Extracting all frames."
        )
        target_fps = None  # Disable subsampling if source FPS is invalid

    frame_skip_interval = 1
    if target_fps and target_fps > 0:
        if target_fps > source_fps:
            cap.release()  # Release resource before raising
            raise ValueError(f"Target FPS ({target_fps}) cannot be higher than source FPS ({source_fps:.2f}).")
        
        # Calculate how many frames to skip.
        # e.g., source=30, target=15 -> interval=2. Keep every 2nd frame.
        frame_skip_interval = round(source_fps / target_fps)
        logger.info(
            f"Source FPS: {source_fps:.2f}, Target FPS: {target_fps}. "
            f"Extracting every {frame_skip_interval} frames."
        )

    frame_index = 0
    saved_frame_count = 0
    while True:
        # Read the next frame from the video.
        is_frame_read, frame = cap.read()
        if not is_frame_read:
            # The video has ended.
            break

        # Check if this frame should be saved based on the calculated interval.
        if frame_index % frame_skip_interval == 0:
            # If resize dimensions are provided, resize the frame.
            if resize_dim:
                try:
                    frame = cv2.resize(frame, resize_dim)
                except cv2.error as e:
                    logger.error(f"Failed to resize frame {frame_index}: {e}")
                    frame_index += 1
                    continue  # Skip this frame

            # Use saved_frame_count for sequential numbering of output files.
            frame_filename = os.path.join(output_dir, f"{frame_prefix}_{saved_frame_count:05d}.jpg")
            if not cv2.imwrite(frame_filename, frame):
                # Log a warning if a frame fails to save, but continue.
                logger.warning(f"Could not write frame to {frame_filename}")
            else:
                saved_frame_count += 1
        
        frame_index += 1
    
    # Release the video capture object.
    cap.release()
    logger.info(f"Successfully extracted {saved_frame_count} frames to '{output_dir}'")


def create_video(img_dir: str, output_video: str, prefix: str, fps: int, codec: str) -> None:

    """Creates a video from a sequence of images.

    This function scans a directory for images with a specific prefix, sorts them
    alphanumerically, and compiles them into a video file using the specified
    codec and frame rate.

    Args:
        img_dir: The path to the directory containing the input images.
        output_video: The path and filename for the output video file.
        prefix: The prefix used to filter and select the image files.
        fps: The frame rate (frames per second) for the output video.
        codec: The four-character code (FourCC) for the video codec (e.g., 'mp4v', 'XVID').
    
    Raises:
        ValueError: If no images with the specified prefix are found in the
            directory.
    """

    # Find and sort all image files in the directory that match the prefix. 
    images = sorted(
        [img for img in os.listdir(img_dir) if img.startswith(prefix) and img.endswith((".png", ".jpg"))]
    )

    if not images:
        raise ValueError(f"No images found with prefix '{prefix}' in {img_dir}")
    
    # Read the first image to determine video dimensions.
    first_frame_path = os.path.join(img_dir, images[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise IOError(f"Could not read the first image: {first_frame_path}")
    height, width, _ = first_frame.shape

    # Initialize the video writer object.
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write each frame to the video.
    for image_name in images:
        image_path = os.path.join(img_dir, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)
    
    # Release the video writer and close any OpenCV windows.
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Successfully created video at '{output_video}' from {len(images)} frames.")



def get_fps(video_path: str) -> int:
    """
    Retrieves the frames per second (FPS) of a video file.

    This function opens the specified video file, queries its FPS property,
    and returns it as an integer. It includes error handling for file
    access, video opening, and invalid FPS values.

    Args:
        video_path (str): The full path to the video file.

    Returns:
        int: The frames per second of the video, rounded down to the nearest integer.

    Raises:
        FileNotFoundError: If the `video_path` does not exist.
        ValueError: If the video reports an FPS of zero or less, which is
                    considered invalid.
        RuntimeError: For any other unexpected errors during the process.
    """
    
    if not os.path.exists(video_path): 
            raise FileNotFoundError(f"Input video file not found: {video_path}")
    cap = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            raise IOError(f"Could not open video file {video_path} to read FPS.")
        output_fps = cap.get(cv2.CAP_PROP_FPS)

        if output_fps <= 0:
            output_fps = 30
            raise ValueError(
                f"Video file {video_path} reported an invalid FPS: {output_fps}. "
                "The file might be corrupted or metadata might be missing."
                "Defaulting to 30 FPS."
            )
        return output_fps
    
    except Exception as e_unknown: # Why: Catch any other unexpected errors and wrap them.
        logger.error(f"An unexpected error occurred while getting FPS from {video_path}: {e_unknown}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred while getting FPS from {video_path}.") from e_unknown
    
    finally:
        if cap is not None and cap.isOpened:
            cap.release()

    
    

#### This function is useful for working with YOLO ####
def create_video_bbox(
    img_dir: str,
    annotations_dir: str,
    output_video: str,
    prefix: str,
    fps: int,
    codec: str,
    class_dict: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]]
) -> None:

    """Creates a video by overlaying bounding boxes from annotations on images.

    This function reads a sequence of images, finds the corresponding YOLO
    annotation files (.txt), draws the bounding boxes and labels on each
    image, and then compiles the annotated images into a video file.

    Args:
        img_dir (str):
            The path to the directory containing the input images.
        annotations_dir (str):
            The path to the directory containing the annotation files in
            YOLO (.txt) format.
        output_video (str):
            The full path (including filename) for the output video file.
        prefix (str):
            The prefix used to filter the image filenames to include.
        fps (int):
            The frame rate (frames per second) for the output video.
        codec (str):
            The four-character code (FourCC) specifying the video codec
            (e.g., 'mp4v', 'XVID').
        class_dict (dict):
            A dictionary that maps class IDs (int) to their corresponding
            class names (str) for labeling.
        colors (dict):
            A dictionary that maps class IDs to color tuples (BGR) for
            drawing the bounding boxes.

    Raises:
        ValueError: If no images matching the prefix are found in `img_dir`.
        FileNotFoundError: If an expected annotation file cannot be found.
    """

    # Find and sort all image files in the directory that match the prefix. 
    images = sorted(
        [img for img in os.listdir(img_dir) if img.startswith(prefix) and img.endswith((".png", ".jpg"))]
    )

    # Raise an exception if no images were found to prevent downstream errors.
    if not images:
        raise ValueError(f"No images found with prefix '{prefix}' in {img_dir}")
    
    # Read the first image to determine the video's dimensions.
    # This assumes all images in the sequence are the same size.
    first_frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Prepare a list to store all annotated images in memory.
    images_annotated = []
    for image_name in images:
        # Load the current image into memory.
        image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(image_path)

        # Construct the annotation filename based on the image name.
        # e.g., 'frame_001.jpg' becomes 'frame_001.txt'.
        annotations_file = image_name.rsplit('.', 1)[0] + ".txt"
        annotations_path = os.path.join(annotations_dir, annotations_file)

        # Open and read the corresponding annotation file.
        boxes = []
        class_objects = []
        with open(annotations_path, "r") as f:
            lines = f.readlines()
            # Iterate over each line in the annotation file (each line is one object).
            for line in lines:
                parts = line.strip().split()
                class_object = int(parts[0])  # The first element is the class ID.
                
                # The subsequent elements are the normalized bounding box coordinates.
                x, y, w, h = map(float, parts[1:])

                # Convert normalized (0-1) coordinates to absolute pixel coordinates.
                x, y, w, h = denormalize_bbox_coordinates(x, y, w, h, width, height)
                
                # Convert float coordinates to integers for OpenCV's drawing functions.
                x, y, w, h = map(int, [x, y, w, h])
                
                boxes.append((x, y, w, h))
                class_objects.append(class_object)

            # Draw all boxes and labels on the image using an external helper function.
            image_annotated = display_bbox_on_image(image, boxes, class_dict, class_objects, colors)
            images_annotated.append(image_annotated)
    
    # After processing all images, write each annotated frame to the video file.
    for a_img in images_annotated:
        video_writer.write(a_img)
    
    # Finalize the video by releasing the VideoWriter object.
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Successfully created video at '{output_video}' from {len(images)} frames.")
