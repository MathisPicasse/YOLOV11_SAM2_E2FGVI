from modules.detector import Detector
from ultralytics import SAM 
import cv2
import numpy as np

model_path = 'models/YOLO11v/yolo11n.pt'
tracker = 'configs/trackers/botsort.yaml'

new_detector = Detector(model_path, tracker)
results = new_detector.track('dataset/raw/people_edited.mp4')

result_transform = new_detector.results_format()
print(result_transform[1])


model = SAM("sam2.1_b.pt")
# Display model information (optional)
model.info()

cap = cv2.VideoCapture('dataset/raw/people_edited.mp4')

frame_idx = 0
id = 1
person = result_transform[1]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pour chaque détection de cet ID, appliquer si frame correspond
    for frame_id, boxe in person:
        if frame_idx == frame_id:
            x_min, y_min, x_max, y_max = boxe.astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calcul masque SAM
            results = model(frame, bboxes=boxe)
            mask_obj = results[0].masks
            mask_np = mask_obj.data.cpu().numpy()  # shape (1, H, W)
            mask = mask_np[0].astype(np.uint8)

            # Masque coloré vert semi-transparent
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 1] = mask * 255

            # Superposition masque + frame
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    cv2.imshow("Video with bbox + mask", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    frame_idx += 1
