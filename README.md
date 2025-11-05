# Red Bull Detector
The purpose of this application is to crop clips out of vidoes, where there is someone drinking a can of Red Bull.
It is designed to accept a folder containing live stream vods and makes use of a YOLOv8 model to detect Red Bull Cans, and then saves short clips.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model](#model)
4. [Detection Process](#detection-process)


## Installation
1. Clone the repository:
```bash
git clone https://github.com/Tom-GDavies/RedBullDetector.git
cd RedBullDetector
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate    # On macOS/Linux
venv/Scripts/activate       # On Windows
 ```

3. Install dependencies:
```bash
 pip install -r requirements.txt
 ```

## Usage
To run the project, use the following command:
```bash
python redBullDetector.py --input_file_path <input_file_path> --output_file_path <output_file_path> --model_path <model_path>
```


## Model

The object detection model selected was YOLOv8 which was trained on a dataset found at https://app.roboflow.com/redbulldetector/redbulldetector-ph8bk/upload.

## Detection Process
