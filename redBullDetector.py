"""
File: redBullDetector.py
Description: This script is responsible for detecting segments of vidoes containing Red Bull cans
Example usage: python redBullDetector.py --input_file_path <input_file_path> --output_file_path <output_file_path> --model_path <model_path>
Data: 03-11-2025
Author: Tom Davies
"""

###############################################
# IMPORTS
###############################################

import argparse
import ultralytics
import cv2
import os

###############################################
# CONSTANTS
###############################################

TARGET_FPS = 3  # Frames per second for processing
MINIMUM_CONSECUTIVE = 3  # Minimum consecutive frames with detection to save

###############################################
# GET COMMAND LINE ARGUMENTS
###############################################

parser = argparse.ArgumentParser(description="Red Bull Can Detector")
    
parser.add_argument("--input_file_path", type=str, default="input/", help="Path to the folder containing input videos")
parser.add_argument("--output_file_path", type=str, default="output/", help="Path to save output clips")
parser.add_argument("--model_path", type=str, default="runs/train/red_bull_yolo_model/weights/best.pt", help="Path to the trained YOLO model")

args = parser.parse_args()

###############################################
# LOAD INPUT DATA
###############################################

input_videos = []

# Make the output folder
if not os.path.exists(args.output_file_path):
    os.makedirs(args.output_file_path)

###############################################
# LOAD MODEL
###############################################

model = ultralytics.YOLO(args.model_path)

###############################################
# LOOP THROUGH VIDEOS
###############################################

# Loop through all video files in the input directory
for video_file in os.listdir(args.input_file_path):
    if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')): # Skip non-video files
        continue

    video_path = os.path.join(args.input_file_path, video_file)
    cap = cv2.VideoCapture(video_path) # Open the video file

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output video writer
    output_path = os.path.join(args.output_file_path, f"detected_{video_file}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Used to limit fps
    mod = max(1, round(fps / TARGET_FPS))


    frame_count = 0
    consecutive = 0

    ###############################################
    # LOOP THROUGH FRAMES
    ###############################################

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1

        # Limit fps
        if frame_count % mod != 0:
            continue

        results = model(frame) # Perform detection

        boxes = results[0].boxes

        if boxes.shape[0] == 0:
            consecutive = 0
            continue # Skip if no Red Bull cans detected

        ###############################################
        # IF A RED BULL CAN IS DETECTED
        ###############################################

        consecutive += 1

        if consecutive < MINIMUM_CONSECUTIVE:
            continue # Require minimum number of consecutive freames with detection

        annotated_frame = cv2.resize(results[0].plot(), (width, height))

        out.write(annotated_frame)

    cap.release()
    out.release()