# 🧬 Blood Cell Detection Using YOLOv10

AI-powered, image-based blood cell detection system using the latest YOLOv10 object detection algorithm  
Built with Python, Ultralytics YOLOv10, Roboflow, OpenCV, and Gradio

---

## 🚀 Project Overview

This project applies real-time object detection to microscopic blood smear images to identify and count **Red Blood Cells (RBCs)**, **White Blood Cells (WBCs)**, and **Platelets**.  
It utilizes the newly released **YOLOv10** architecture for enhanced accuracy and speed. The model was trained using a custom-labeled dataset prepared via Roboflow and deployed with an interactive **Gradio web interface** for live predictions.

---

## ✨ Features

- **Advanced Object Detection:** Powered by YOLOv10 for efficient and accurate cell detection  
- **Custom Dataset:** Trained on a labeled dataset with RBC, WBC, and Platelet annotations  
- **Gradio Web Interface:** Upload images and get annotated results with detection counts  
- **Small Object Friendly:** Designed to accurately detect small, overlapping cells  
- **Class-wise Count Display:** Uses Python `Counter` to display detected class frequencies  
- **Visualization Support:** Outputs saved and visualized using Matplotlib  

---

## 🧑💻 Technology Stack

- **Python 3**
- **Ultralytics YOLOv10** – Model training and inference
- **OpenCV** – Image conversion and preprocessing
- **Gradio** – Web-based user interface for interactive prediction
- **Matplotlib** – Displaying output images
- **Roboflow** – Dataset annotation and export in YOLO format
- **NumPy & Collections** – For image processing and detection counting

---


---

## 🗂️ Dataset Information

- **Source:** [Roboflow Blood Cell Detection Dataset](https://universe.roboflow.com)  
- **Classes:** RBC, WBC, Platelets  
- **Images:** Microscopic blood smear images with bounding box annotations  
- **Annotations Format:** YOLO-compatible `.txt` files with class indices and coordinates  
- **Preprocessing:** Performed resizing, augmentation (e.g., exposure, flip) in Roboflow  

---

## ⚙️ How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/blood-cell-detection-yolov10.git
   cd blood-cell-detection-yolov10

---


