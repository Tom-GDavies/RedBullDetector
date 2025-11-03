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

###############################################
# GET COMMAND LINE ARGUMENTS
###############################################

parser = argparse.ArguementParser(description="Red Bull Can Detector")
    
parser.add_argument("--input_file_path", type=str, default="input/", help="Path to the file containing input videos")
parser.add_argument("--output_file_path", type=str, default="output/", help="Path to save output clips")
parser.add_argument("--model_path", type=str, default="runs/train/red_bull_yolo_model/weights/best.pt", help="Path to the trained YOLO model")

args = parser.parse_args()

###############################################
# LOAD INPUT DATA
###############################################

input_videos = []

###############################################
# GENERATE FRAMES TO PROCESS
###############################################



###############################################
# DETECT REDBULL CANS IN FRAMES
###############################################



###############################################
# DETECT IF REDBULL CAN IS BEING DRUNK
###############################################



###############################################
# EXTRACT CLIPS WITH REDBULL CANS BEING DRUNK
###############################################