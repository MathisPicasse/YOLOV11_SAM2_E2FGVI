import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import subprocess
#from modules.utils.video import extract_frames


mask_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw/mask_venice"
video_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/raw"
video_name = "venice2_resized.mp4"
script_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/Inpainting/E2FGVI/test.py"
model = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/Inpainting/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"

#output_frames = f"{video_path}/people_frame"
#extract_frames(f"{video_path}/{video_name}", output_frames)

result = subprocess.run(f"python {script_path} --model e2fgvi_hq --video {video_path}/venice_frame --mask {mask_path}  --ckpt {model} --set_size --width 512 --height 512", shell=True)