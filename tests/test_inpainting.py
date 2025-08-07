import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import subprocess
#from modules.utils.video import extract_frames


mask_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/outputs/masks_moi"
video_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/dataset/processed/moi_project"
video_name = "processed_video_moi.mp4"
script_path = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/Inpainting/E2FGVI/test.py"
model = "/home/mapicasse/Documents/02_Academic/Internship/YOLO11_SAM_E2FGVI/models/Inpainting/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"

#output_frames = f"{video_path}/frames"
#extract_frames(f"{video_path}/{video_name}", output_frames)

result = subprocess.run(f"python {script_path} --model e2fgvi_hq --video {video_path}/frames --mask {mask_path}  --ckpt {model} --set_size --width 864 --height 480", shell=True)