import os
import cv2
import logging

def create_frames(output_dir, frame_prefix, path_to_video, resize_dim=None):
    if not os.path.exists(path_to_video):
        logging.error("The specified video path does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        logging.error(f"Cannot open the video located at {path_to_video}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame if resize dimensions are provided
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)

        # Construct frame filename with prefix and zero-padded frame count
        frame_filename = os.path.join(output_dir, f"{frame_prefix}_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"{frame_count} frames extracted to folder '{output_dir}'")


def create_video(img_dir, output_video, prefix, fps, codec):
    """
    Create a video by overlaying bounding boxes from annotations on images.

    Parameters:
    -----------
    img_dir : str
        Path to the directory containing the input images.

    annotations_dir : str
        Path to the directory containing the annotation files (.txt format, YOLO format).

    output_video : str
        Path (including filename) for the output video file to be created.

    prefix : str
        Prefix used to filter the image filenames to include in the video.

    fps : int
        Frames per second for the output video.

    codec : str
        Four-character code (e.g., 'mp4v', 'XVID') specifying the codec to use for encoding the video.


    Returns:
    --------
    None
        The function writes the video file to disk and does not return anything.
    """
    # Get the sorted images list using the prefix parameter
    images = sorted(
        [img for img in os.listdir(img_dir) if img.startswith(prefix) and img.endswith((".png", ".jpg"))]
    )

    # Check if any images were found
    if not images:
        raise ValueError(f"No images found with prefix '{prefix}' in {img_dir}")
    
    # Load the first image to get its size
    first_frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, _ = first_frame.shape

    # Initialize the video writer using the first image to get dimensions
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    
    for image_name in images:
        # Read the image
        image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # Finalize the video
    video_writer.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
