# Red Bull Detector

## Purpose

The purpose of this application is to crop clips out of vidoes, where there is someone drinking a can of Red Bull.
It is designed to accept a folder containing live stream vods and makes use of a YOLOv8 model to detect Red Bull Cans, and then saves short clips.

## Model

The object detection model selected was YOLOv8 which was trained on a dataset found at https://app.roboflow.com/redbulldetector/redbulldetector-ph8bk/upload.

## Detection Process
