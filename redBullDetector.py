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
# FUCNTION TO CREATE CLIPS GIVE START AND END FRAMES
###############################################
def create_clip(start_frame, end_frame, video_path, file_name, output_file_path):

    # Create output path
    clip_name = "clip_{}_{}_{}.mp4".format(file_name, start_frame, end_frame)
    output_path = os.path.join(output_file_path, clip_name)

    # Read frames from the original video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_to_write = end_frame - start_frame + 1

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to new clip
    while frames_to_write > 0:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_to_write -= 1
    
    cap.release()
    out.release()

###############################################
# CONSTANTS
###############################################

TARGET_FPS = 10  # Frames per second for processing
MINIMUM_CONSECUTIVE = 5  # Minimum consecutive frames with detection to save
LEEWAY_FRAMES = 2 # Leeway given for missed Red Bull detections
BORDER_FRAMES = 30 # Number of frames added before and after clip

###############################################
# GET COMMAND LINE ARGUMENTS
###############################################

parser = argparse.ArgumentParser(description="Red Bull Can Detector")
    
parser.add_argument("--input_file_path", type=str, default="input/", help="Path to the folder containing input videos")
parser.add_argument("--output_file_path", type=str, default="output/", help="Path to save output clips")
parser.add_argument("--model_path", type=str, default="runs/train/red_bull_yolo_model/weights/best.pt", help="Path to the trained YOLO model")

args = parser.parse_args()

###############################################
# CREATE OUTPUT FOLDER
###############################################

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

    # Used to limit fps
    mod = max(1, round(fps / TARGET_FPS))


    frame_count = 0
    start_frame = 0
    consecutive = 0
    leeway = LEEWAY_FRAMES
    clips = []

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

            # Provides leeway for missed detections
            if consecutive > 0:
                if leeway > 0:
                    leeway -= 1
                    continue
                
            # Add clip if minimum consecutive frames met
            if consecutive >= MINIMUM_CONSECUTIVE:
                start = max(0, start_frame - BORDER_FRAMES)
                end = min(frame_count + BORDER_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                create_clip(start, end, video_path, video_file.split('.')[0], args.output_file_path)

            consecutive = 0
            leeway = LEEWAY_FRAMES

        ###############################################
        # IF A RED BULL CAN IS DETECTED
        ###############################################
        else:
            if consecutive == 0:
                start_frame = frame_count

            leeway = LEEWAY_FRAMES
            consecutive += 1

    cap.release()