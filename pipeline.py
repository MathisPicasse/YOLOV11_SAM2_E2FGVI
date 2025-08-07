import cv2
from modules.tasks.detector import YoloDetector, Detector
from modules.tasks.tracker import UltralyticsTracker
from modules.tasks.masker import Masker
from modules.utils.visualisation import draw_bbox, draw_mask  
from modules.utils.video import create_frames, create_video, get_fps
from modules.data.observations import Observation
from typing import Optional, Tuple, Dict, List
import logging
import os
from colorama import Fore, Style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a logger instance

# === Paths Configuration ===

#Workspace Configuration
WORKSPACE = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI"
DATA_PATH = f"{WORKSPACE}/dataset/raw"

#Poject configuation
PROJECT_NAME = "moi"
VIDEO_NAME = "video_moi.mp4"
VIDEO_PATH = f"{DATA_PATH}/{PROJECT_NAME}/{VIDEO_NAME}"

#Detection configuration
MODEL_NAME = "yolo11"
MODEL_VERSION = "n"
MODEL_PATH = f"{WORKSPACE}/models/detection/{MODEL_NAME.upper()}v/{MODEL_NAME}{MODEL_VERSION}.pt"

#Tracker configuration 
TRACKER_NAME = "botsort"
TRACKER_CONFIG_PATH = f"{WORKSPACE}/configs/trackers/{TRACKER_NAME}.yaml"

#Masker configuration
MASK_MODEL = "sam2.1_b.pt"
MASKS_OUTPUT = f"{WORKSPACE}/outputs/masks_{PROJECT_NAME}"

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
    
    # 1. Creation of the frames
    logger.info(Fore.YELLOW + f"Extracting frames to: {output_frame_dir} with prefix '{frame_prefix}' and target size: {target_size}" + Style.RESET_ALL)

    try:
        create_frames(output_frame_dir, frame_prefix, video_path, resize_dim=target_size)
        logger.info(Fore.GREEN + "Frame extraction completed." + Style.RESET_ALL)
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed during frame extraction: {e}")
        raise IOError(f"Failed to extract frames from {video_path}") from e 


    # 2. Determine FPS for the output video
    output_fps = fps
    if output_fps is None:
        logger.info(Fore.YELLOW + "Determining FPS from original video..." + Style.RESET_ALL)
        output_fps =get_fps(video_path)
    else: 
        logger.info(f"Using specified FPS for output video: {output_fps}")
    
    # 3. Creation of the output video from processed frames
    logger.info(Fore.YELLOW + f"Creating output video at: {output_video_path} with FPS {output_fps} and codec '{codec}'" + Style.RESET_ALL)
    try: 
        create_video(output_frame_dir, output_video_path, frame_prefix, fps=output_fps, codec=codec)
        logger.info(Fore.GREEN + "Output video creation completed." + Style.RESET_ALL)
    except (ValueError, IOError) as e: 
        logger.error(f"Failed during output video creation: {e}")
        raise IOError(f"Failed to create output video at {output_video_path}") from e # Re-raise as IOError
    logger.info(f"Video preprocessing finished for: {video_path}")


NEED_PREPROCESSING = True

def main():

    logger.info(Style.BRIGHT+"######## VIDEO ANALYSIS PIPELINE #######" + Style.RESET_ALL)

    # Use local variables for the video path and name, which might change after preprocessing
    processed_video_path = VIDEO_PATH
    processed_video_name = VIDEO_NAME

    # === Preprocessing ===
    if NEED_PREPROCESSING:
        output_frame_dir = f"{DATA_PATH}/processed/{PROJECT_NAME}/frames"
        output_video_path = f"{DATA_PATH}/processed/{PROJECT_NAME}/processed_{VIDEO_NAME}"
        frame_prefix = PROJECT_NAME 

        try:
            preprocess_video(
                video_path=VIDEO_PATH,
                output_frame_dir=output_frame_dir,
                output_video_path=output_video_path,
                frame_prefix=frame_prefix,
                target_size=(864, 480),
                fps=25, # Now this FPS will be used for the output video if not None
                codec='mp4v'
            )
            # Update the local variables to point to the processed video
            processed_video_path = output_video_path
            processed_video_name = os.path.basename(output_video_path) # Get just the filename
        except (FileNotFoundError, IOError, ValueError) as e:
            logger.error(f"Preprocessing failed: {e}")
            # Exit the script if preprocessing fails as subsequent steps depend on it
            exit(1)

    # === Module Initialization ===
    logger.info(Fore.YELLOW+"Initializing modules..." + Style.RESET_ALL)
    detector = YoloDetector(MODEL_PATH)
    tracker = UltralyticsTracker(TRACKER_CONFIG_PATH)
    # A detection pipeline is the combination of a detector and a tracker
    detection_pipeline = Detector(detector=detector, tracker=tracker)
    masker = Masker(MASK_MODEL)
    logger.info(Fore.GREEN+"Modules initialized" + Style.RESET_ALL)

    # === Run Detection + Tracking Pipeline ===
    try:
        detection_pipeline.track(processed_video_path)
        #prepare for working with inpainting
        results_formatted = detection_pipeline.format_results()
    except Exception as e: # Catch potential errors in the tracking pipeline
        logger.error(f"Detection and tracking pipeline failed: {e}", exc_info=True)
        exit(1)

    # ===  Masking ===
    target_tracker_ids = [1] # Example: Process tracker IDs 1, 2, and 3
    all_observations_for_targets: List[Observation] = []
    for tracker_id in target_tracker_ids:
        if tracker_id in results_formatted: # Corrected check
            all_observations_for_targets.extend(results_formatted[tracker_id])
        else:
            logger.warning(f"Tracker ID {tracker_id} not found in tracking results. Skipping for masking.")
    
    if not all_observations_for_targets:
        logger.info("No observations found for the target tracker IDs. Skipping masking and visualization.")
    else:
        # Ensure MASKS_OUTPUT directory exists
        try:
            os.makedirs(MASKS_OUTPUT, exist_ok=True)
            logger.info(f"Mask output directory ensured/created at: {MASKS_OUTPUT}")
        except OSError as e:
            logger.error(f"Could not create mask output directory {MASKS_OUTPUT}: {e}. Skipping masking.")
            all_observations_for_targets.clear() # Prevent further processing if dir fails

    if all_observations_for_targets:
        cap = None
        try:
            cap = cv2.VideoCapture(processed_video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video for visualization: {processed_video_path}")
                # Consider raising an error or exiting more gracefully
                exit(1) 

            logger.info(Fore.YELLOW + f"Starting visualization and mask creation for target tracker IDs: {target_tracker_ids}" + Style.RESET_ALL)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream for masking.")
                    break

                current_frame_observations = [
                    obs for obs in all_observations_for_targets if obs.frame_id == frame_idx
                ]
                
                for obs in current_frame_observations: # obs is an Observation object
                    #drawing the bbox on the frame
                    draw_bbox(frame, obs)
                try:
                    mask_filename = f"mask_frame{frame_idx:06d}.png" 
                    mask = masker.create_mask(frame, current_frame_observations, MASKS_OUTPUT, mask_filename)
                    if mask is not None:
                        frame = draw_mask(frame, mask) 
                except Exception as e: 
                    logger.error(f"Error processing observation {obs} for frame {frame_idx}: {e}", exc_info=True)   

                cv2.imshow("Video with bbox + mask", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit visualization.")
                    break
                frame_idx += 1
        except Exception as e: # Catch any other unexpected errors in the loop
            logger.error(f"An error occurred during the masking/visualization loop: {e}", exc_info=True)
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Video visualization and masking finished.")

    logger.info(Style.BRIGHT+"######## PIPELINE FINISHED #######" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
