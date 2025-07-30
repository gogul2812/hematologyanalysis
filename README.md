Blood Cell Detection & Classification with YOLOv10
Project Objective:
Automatically detect and count different blood cell types (RBCs, WBCs, Platelets) in microscopy images using YOLOv10 object detection.

Table of Contents
Project Overview

System Requirements

Setup Instructions

Dataset

Model Training & Validation

Results & Metrics

Inference & Visualization

Gradio App Deployment

Directory Structure

Citation

License

Contact

Project Overview
This repository contains all code, scripts, and deployment tools for segmenting and classifying blood cells in high-res images. The project uses:

YOLOv10n (nano): Lightweight, high-performance detector.

Ultralytics YOLO & Gradio: For training, evaluation, and web app deployment.

Roboflow: Easy dataset management & augmentation.

System Requirements
Python 3.8+

CUDA-enabled GPU (recommended)

pip

Key packages:

torch (>=1.8.0)

ultralytics

opencv-python

matplotlib

roboflow

gradio

...see requirements.txt

Setup Instructions
Clone the repository:

bash
git clone https://github.com/yourusername/blood-cell-detection-yolov10.git
cd blood-cell-detection-yolov10
Install dependencies:

bash
pip install -r requirements.txt
Customize paths in notebooks/scripts as needed.

Dataset
Source: Roboflow project: blood-cell-detection-bsbvn (v3)

Classes: RBC, WBC, Platelets

Format: Images (.jpg) and YOLO-format labels

Download using Roboflow:

python
from roboflow import Roboflow
rf = Roboflow('<your-roboflow-api-key>')
project = rf.workspace("clg-vtj9f").project("blood-cell-detection-bsbvn")
dataset = project.version(3).download("yolov8")
Replace <your-roboflow-api-key> with your key.

Model Training & Validation
Model:
YOLOv10n, 3 output classes, 25 epochs, batch size 32, augmentations enabled.

Training:

bash
yolo task=detect mode=train epochs=25 batch=32 plots=True \
  model=<path_to_yolov10n.pt> \
  data=<path_to_data.yaml>
Validation:

bash
yolo task=detect mode=val model=<path_to_trained_model.pt> \
  data=<path_to_data.yaml>
Results & Metrics
Class	Precision	Recall	mAP50	mAP50-95
RBC	0.825	0.828	0.887	0.624
WBC	0.929	0.968	0.972	0.625
Platelets	0.684	0.751	0.734	0.380
All	0.812	0.849	0.864	0.543
Speed: ~6ms/image for inference

Outputs in: /runs/detect/

Inference & Visualization
Predict and visualize detections on validation images:

python
from ultralytics import YOLO
model = YOLO(<path_to_best_weights.pt>)
model(source=<path_to_images>, conf=0.25, save=True)

# Visualization
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
images = glob.glob('<predict_folder_path>/*.jpg')[:10]
fig, axes = plt.subplots(2, 5, figsize=(20,10))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(mpimg.imread(images[i]))
        ax.axis("off")
    else:
        ax.axis("off")
plt.tight_layout()
plt.show()
Sample output counts:

text
RBC: 26, WBC: 1
Gradio App Deployment
Simple web app for live demo:

python
import gradio as gr
import cv2
import numpy as np
from collections import Counter

def predict(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model.predict(source=image_rgb, imgsz=640, conf=0.25)
    annotated_img = result[0].plot()
    detections = result[0].boxes.data
    class_names = [model.names[int(cls)] for cls in detections[:,5]]
    count = Counter(class_names)
    detection_str = ', '.join([f"{name}:{count}" for name, count in count.items()])
    annotated_img = annotated_img[:, :, ::-1]
    return annotated_img, detection_str

app = gr.Interface(
    predict,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[gr.Image(type="numpy", label="Annotated Image"), gr.Textbox(label="Detection Counts")],
    title="Blood Cell Detection Using YOLOV10",
    description="Upload an image and YOLOv10 will detect blood cells."
)

app.launch()
Directory Structure
text
blood-cell-detection-yolov10/
├── notebooks/
│   └── Hematology_Analysis.ipynb
├── src/
├── data/
│   └── (README or scripts; dataset not included)
├── app/
│   └── gradio_app.py
├── runs/
│   └── detect/
├── requirements.txt
├── README.md
└── .gitignore
Citation
YOLOv10: THU-MIG/yolov10 on GitHub

Roboflow platform

License
Distributed under the MIT License. See LICENSE for details.

Contact
Feel free to contact me anytime if you have questions, feedback, or want to collaborate!
Ping me here, create an issue, or reach out via my profile. Glad to help
