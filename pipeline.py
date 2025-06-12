import cv2
from modules.tasks.detector import YoloDetector, Detector
from modules.tasks.tracker import UltralyticsTracker
from modules.tasks.masker import Masker
from modules.utils.visualisation import draw_bbox, draw_mask  
from modules.utils.video import create_frames, create_video



# === Paths Configuration ===
VIDEO_PATH = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_edited_resize.mp4"
YOLO_MODEL_PATH = "models/YOLO11v/yolo11n.pt"
TRACKER_CONFIG_PATH = "configs/trackers/botsort.yaml"
MASK_MODEL = "sam2.1_b.pt"
MASKS_OUTPUT = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/mask_people_resize"

# === Preprocessing ===
#The aim of this part is to downsize images and to create a new video from the newly created downsized images.
#This will help to speed up the computation ie the detection and the segmentation.
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
create_frames("/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_frame", "people", "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_edited.mp4", resize_dim=(512,512))
create_video("/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_frame", "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_edited_resize.mp4", "people", fps=fps, codec='mp4v')

# === Module Initialization ===
detector = YoloDetector(YOLO_MODEL_PATH)
tracker = UltralyticsTracker(TRACKER_CONFIG_PATH)
detection_pipeline = Detector(detector=detector, tracker=tracker)
masker = Masker(MASK_MODEL)
masker.info()  



# === Run Detection + Tracking Pipeline ===
detection_pipeline.track(VIDEO_PATH)
results_formatted = detection_pipeline.format_results()

# === Select tracked person ID and retrieve observations ===
PERSON_ID = 1
person_observations = results_formatted[PERSON_ID]

# === Open video for reading ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    for obs in person_observations:
        if obs.frame_id == frame_idx:
            draw_bbox(frame, obs)  
            #Handling mask creation for the inpiainting part
            mask_name = f"mask_{frame_idx}.png"
            mask = masker.create_mask(frame, obs, MASKS_OUTPUT, mask_name)  
            frame = draw_mask(frame, mask)  

    cv2.imshow("Video with bbox + mask", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
