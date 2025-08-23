# 🎬 Video Inpainting Pipeline + MOT → YOLO Converter

## 🚀 End-to-End Pipeline for **Video People Removal**

This project implements a complete **video inpainting pipeline** to automatically remove people from videos.  
It combines **state-of-the-art models** in detection, segmentation and inpainting to achieve robust results on challenging videos (e.g. surveillance footage with multiple people).

---

## 🔍 Models & Pipeline

| Step                  | Model/Tool | Description |
|-----------------------|------------|-------------|
| **Detection + Tracking** | [YOLO](https://github.com/ultralytics/ultralytics) + BoT-SORT | Detect and track people across frames |
| **Segmentation**      | [Segment Anything Model (SAM)](https://segment-anything.com/) | Generate binary masks for the detected persons |
| **Video Inpainting**  | [E2FGVI](https://github.com/MCG-NKU/E2FGVI) | Remove masked objects while maintaining temporal consistency |

---

## 📂 Dataset Engineering

Most of the work focused on building a **robust dataset** to train YOLO for people detection on challenging videos.  

- Datasets used: [MOT17](https://motchallenge.net/data/MOT17/) + [MOT20](https://motchallenge.net/data/MOT20/) from the **MOTChallenge**.  
- Custom script to:
  - **Merge datasets**  
  - **Convert MOT format → YOLO format**  

👉 The script is available in [`scripts/convertToYolo.py`](scripts/convertToYolo.py.py).

---

## 🏗️ Project Context

This project was carried out during my **6-month internship at the Computer Vision Laboratory, Linköping University**.  
It allowed me to explore **Computer Vision, dataset engineering, and model training** in depth.  

📄 You can read more about the challenges and results in my [internship report](assets/internship_report.pdf).

---

## 📊 Results

- Custom training of YOLO on MOT17+MOT20 significantly improved **mAP** on dense, multi-person videos.  
- Example (before vs after inpainting):  

>>>>>>> 5b3568b (readme.md)
