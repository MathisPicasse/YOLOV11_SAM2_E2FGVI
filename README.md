# 🎬 Video Inpainting Pipeline + MOT → YOLO Converter

## 🚀 End-to-End Pipeline for **Video People Removal**

This project implements a complete **video inpainting pipeline** to automatically remove people from videos.  
It combines **state-of-the-art models** in detection, segmentation and inpainting to achieve robust results on challenging videos (e.g. surveillance footage with multiple people).

[Example of result](assets/output.gif)

---
## Models & Pipeline

| Step                  | Model/Tool | Description |
|-----------------------|------------|-------------|
| **Detection + Tracking** | [YOLO](https://github.com/ultralytics/ultralytics) + BoT-SORT | Detect and track people across frames |
| **Segmentation**      | [Segment Anything Model (SAM)](https://segment-anything.com/) | Generate binary masks for the detected persons |
| **Video Inpainting**  | [E2FGVI](https://github.com/MCG-NKU/E2FGVI) | Remove masked objects while maintaining temporal consistency |

<p align="center">
  <img src="assets/pipeline_overview.png" alt="Initial dataset structure" width=600/>
</p>


---

## Project Context

This project was carried out during my **6-month internship** at the <a href="https://liu.se/en/organisation/liu/isy/cvl">Computer Vision Laboratory</a>, <a href="https://liu.se/en">Linköping University</a>.  
It allowed me to explore **Computer Vision** and **dataset engineering** in depth.  

📄 You can read more about the challenges and results in my [internship report](assets/internship_report.pdf).

---

## Setup installation 
1. Clone this repository to your local machine (or download the ZIP and extract it to any desired location):

   ```bash
   git clone https://github.com/MathisPicasse/YOLOV11_SAM2_E2FGVI.git
   cd YOLOV11_SAM2_E2FGVI
   ```

2. Install Python 

   #### Linux/Unix
   <details>
  
   We recommend using <a href="https://github.com/pyenv/pyenv">pyenv</a> to manage Python versions. This project uses **Python 3.10.13**:

   ```bash
   
   pyenv install 3.10.13
   ```
   </details>
  
   #### Windows
   <details>
   pyenv is not available natively. 
   <ul>Install Python 3.10.13 directly
   <li>Use <a href="https://github.com/pyenv-win/pyenv-win">pyenv-win</a></li>
   </ul>
   </details>
  

3. Create and Configure the Virtual Environment

   #### Linux/Unix
   <details>

    ```bash
   
    pyenv virtualenv 3.10.13 your_env_name
    ```

    Associate this environment with the project folder (auto-activated)
    
    ```bash
   
    pyenv local your_env_name
    ```
   </details>
  
   #### Windows
   <details>

    ```bash
   
    python -m venv .venv
    .\.venv\Scripts\Activate.bat
    ```
   </details>


4. Install Dependencies
   ```bash
   
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
5. Download models <br>
   You can fin in the folder `02_weights/` on [google drive](https://drive.google.com/drive/folders/1bgOJGK5JYptOmDVDqoLaYZq2qA8GkE5T?usp=drive_link) the differents models and weights for each module:
   <ul>
     <li>best.pt → models/detection </li>
     <li>E2FGVI-HQ-CVPR22.pth -> models/Inpaiting/E2FGVI/release_model</li>
     <li>sam2.1_b.pt -> models/Masks</li>
   </ul>

## 📂 Script to convert from MOT dataset format to YOLO format

Most of the work focused on building a **robust dataset** to train YOLO for people detection on challenging videos to get bounding boxes as first step of the pipeline .

- Datasets used: combined [MOT17](https://motchallenge.net/data/MOT17/) + [MOT20](https://motchallenge.net/data/MOT20/) from the **MOTChallenge**.  

The script allow to resize images, compute bounding boxes in the right format, downsample the number of frames and saving this in some 
specific folder to then train YOLO. 

You can find the dataset already converted to a yolo format on [google drive](https://drive.google.com/drive/folders/1bgOJGK5JYptOmDVDqoLaYZq2qA8GkE5T?usp=drive_link) in `01_dataset/`
### 🔧 How to use the script to convert MOT → YOLO

👉 The script is available in [`scripts/convertToYolo.py`](scripts/convertToYolo.py).

In the folder `configs/`, you will find the file **`MOT_TO_YOLO.json`** to define all parameters for converting the **MOT dataset → YOLO format**.  
Here is the configuration I used adapted to the dataset.

```json
{
    "PathToDataFolders": "/path/to/MOT/data/",
    "OutputDir": "/path/to/output/",
    "Classes": {
        "1": "person",
        "2": "person",
        "7": "person"
    },
    "ResizeImages": true,
    "TargetSize": [864,480],
    "train_folders": {
        "MOT17-02": [1920, 1080],
        "MOT17-05": [640, 480],
        "MOT17-09": [1920, 1080],
        "MOT17-13": [1920, 1080],
        "MOT20-01": [1920, 1080],
        "MOT20-02": [1920, 1080],
        "MOT20-05": [1654, 1080]
    },
    "val_folders": {
        "MOT17-04": [1920, 1080],
        "MOT17-10": [1920, 1080]
    },
    "test_folders": {
        "MOT17-11": [1920, 1080],
        "MOT20-03": [1173, 880]
    },
    "SubsampleRate": 7
}
```
🔑 Parameters Details  

- **PathToDataFolders**: Path to the main folder containing the raw MOT sequences. It corresponds to the folder **`training`** in the dataset structure below.
- **OutputDir**: Path to the output folder. For each sequence, two subfolders will be created:  
  - `img/` → resized frames  
  - `annotations/` → YOLO label files (one per frame).  
- **Classes**: Mapping between MOT class IDs and YOLO class names. In this example, IDs `1`, `2`, and `7` are all grouped under the class `person`.  
- **ResizeImages**: Boolean flag to enable or disable resizing of images.  
- **TargetSize**: Target resolution `[width, height]` for resizing.  
- **train_folders / val_folders / test_folders**: Sequences used for training, validation, and testing. The values `[W, H]` correspond to the original resolution of each video (important for annotation conversion).  
- **SubsampleRate**: Frame subsampling rate. For example, `7` means every 7th frame is kept (0, 7, 14, …).  


The combined MOT dataset must have the following structure:  

<p align="center">
  <img src="assets/initial_dataset_structure.svg" alt="Initial dataset structure"/>
</p>

To run the conversion script using the default configuration file: 
```bash
cd YOLO11_SAM_E2FGVI
python3 -m scripts.convertToYolo configs/MOT_TO_YOLO.json
```



## 🚀 How to Run the Pipeline

Follow these steps to run the complete video inpainting pipeline.

---

### ⚠️ Important Note
Before you begin, ensure you have completed the **Setup installation steps**, especially **Step 5: Download models**.  
The following models must be placed in their correct directories:

- `best.pt` → `models/detection`  
- `E2FGVI-HQ-CVPR22.pth` → `models/Inpaiting/E2FGVI/release_model`  
- `sam2.1_b.pt` → `models/Masks`



### Step 1: Configure the Pipeline
All key parameters for the pipeline are managed within the `config.py` file.  
Open this file and adjust the following variables according to your needs:


**STEPS**
- A list of strings defining which parts of the pipeline to execute.  
- Useful if you want to run only a specific part (e.g., skip detection and go straight to inpainting).
- **Available steps**:  
  - `"detection"`  
  - `"masking"`  
  - `"inpainting"`
- **Example**:  
  ```python
  STEPS = ["masking", "inpainting"]
  ```
  This will skip the detection step.


**NEED_PREPROCESSING**
- Controls whether the input video needs to be pre-processed.
- Set to True if you are providing a raw video. The script will automatically extract and resize the frames to 864x480. A processed video will be saved in dataset/processed, and the resized frames will also be stored there.
- Set to False if you have already pre-processed the video.

**VIDEO_NAME**
- If `NEED_PREPROCESSING = True` → set this to the name of your video file located in `dataset/raw`.
- If `NEED_PREPROCESSING = False` → set this to name of your video in the project folder name inside `dataset/processed`.

**TARGET_ENTITIES_IDS**
A list of the IDs of the people you want to remove from the video.

Workflow suggestion:

<ol>
  <li>Run the pipeline once to get the IDs of all detected people.</li>
  <li>Re-run it with only the masking and inpainting steps to remove selected IDs.
</ol>

### Step 2: run the pipeline
```bash
  cd YOLO11_SAM2_E2FGVI
  python3 pipeline.py
```

you will find the results of the tracking video in the folder `outputs` as well as the masks.
You will see the result of the inpaiting video directly at the end of the run. 


