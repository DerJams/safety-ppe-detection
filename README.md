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
The model was trained for 15 epochs on a T4 GPU. The final evaluation on the validation set yielded the following metrics:

| Class | Images | Instances | Precision (P) | Recall (R) | mAP@0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Classes** | 603 | 4668 | **0.890** | **0.838** | **0.888** |
| Helmet | 555 | 1174 | 0.852 | 0.714 | 0.798 |
| Person | 593 | 1759 | 0.903 | 0.895 | 0.925 |
| Vest | 601 | 1735 | 0.915 | 0.904 | 0.942 |

### Key Insights
* **High Accuracy on Safety Gear:** The model performs exceptionally well at detecting safety vests, achieving a **0.942 mAP**.
* **Reliable Detection:** With a precision of **0.89**, the system maintains a low false-positive rate, which is critical for automated safety monitoring.
* **Room for Improvement:** Detection of helmets (**0.798 mAP**) was slightly lower than other classes, likely due to their smaller scale in long-distance site photos.

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
