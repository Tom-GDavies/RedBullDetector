"""
File: modelTraner.py
Description: This script is responsible for training a YOLO model with the purpose of detecting Red Bull cans in videos
Data: 03-11-2025
Author: Tom Davies
"""


###############################################
# IMPORTS
###############################################

from ultralytics import YOLO
import os
import wandb

###############################################
# Training function
###############################################
def train_YOLO_model():
    # Initialise wandb
    wandb.init(project="RedBullDetector", entity="tom-davies")

    # Define data and pretrained model paths
    data_path = "Dataset/data.yaml"
    pretrained_model_path = "yolov8n.pt"
    
    # Load model
    model = YOLO(pretrained_model_path)

    # Train model
    model.train(
        data=data_path,
        epochs=300,
        imgsz=960,
        batch=16,
        name="red_bull_yolo_model",
        patience=10,
        device=0,
        save=True,
        project="runs/train",
        logger="wandb"
    )

    # Save model

    # Perform evaluation
    metrics = model.val()
    print("Evaluation Metrics:", metrics)

    wandb.finish()

if __name__ == "__main__":
    train_YOLO_model()