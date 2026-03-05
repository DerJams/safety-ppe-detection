# Construction Site Safety: PPE Detection System

This project implements a real-time computer vision system to detect personal protective equipment (PPE) on construction sites. It utilizes the **YOLOv8 (You Only Look Once)** architecture, fine-tuned on a custom dataset to identify "Hard Hat", "Mask", "Vest", and "Person" classes.

## 🎯 Objective
To automate safety compliance monitoring by instantly identifying workers who are missing required safety gear.

## 🛠️ Tech Stack
* **Model Architecture:** YOLOv8 (Nano)
* **Framework:** PyTorch & Ultralytics
* **Data Management:** Roboflow
* **Training Environment:** Google Colab (T4 GPU)

## 📊 Performance
The model was trained for 15 epochs with the following results:
* **mAP@0.5:** 0.92
* **Precision:** 0.89
* **Recall:** 0.85

## 🚀 Usage

### 1. Install Dependencies
pip install ultralytics

### 2. Run Inference
from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Predict on an image
model.predict("test_image.jpg", save=True)

## 📂 Dataset
The dataset was sourced from Roboflow Universe and contains annotated images of construction workers. It includes varying lighting conditions and angles to ensure model robustness.
