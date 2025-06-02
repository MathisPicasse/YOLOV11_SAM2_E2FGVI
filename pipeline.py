import cv2
from modules.tasks.detector import YoloDetector, Detector
from modules.tasks.tracker import UltralyticsTracker
from modules.tasks.masker import Masker
from modules.utils.visualisation import draw_bbox, draw_mask  

# === Paths Configuration ===
VIDEO_PATH = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/people_edited.mp4"
YOLO_MODEL_PATH = "models/YOLO11v/yolo11n.pt"
TRACKER_CONFIG_PATH = "configs/trackers/botsort.yaml"
MASK_MODEL = "sam2.1_b.pt"

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
            mask = masker.create_mask(frame, obs)  
            frame = draw_mask(frame, mask)  

    cv2.imshow("Video with bbox + mask", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
