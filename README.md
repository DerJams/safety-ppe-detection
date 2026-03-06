# Construction Site Safety: Real-Time PPE Detection
### *Automated Safety Compliance using YOLOv8*

This project implements a real-time computer vision system to automate safety inspections. By identifying "Hard Hat," "Vest," and "Person" classes, the system helps safety managers reduce manual oversight and improve site compliance instantly.

## 🚀 Business Value

* **Insurance Compliance:** Provides a verifiable log of safety gear usage.
* **Risk Mitigation:** Real-time alerts for workers missing critical PPE.
* **Scalability:** Designed to integrate with existing CCTV infrastructure via a modular Python API.

---

## 📊 Performance & Insights

The model was fine-tuned on a custom dataset for 15 epochs using a T4 GPU. 

| Class | Images | Precision (P) | Recall (R) | mAP@0.5 |
| :--- | :--- | :--- | :--- | :--- |
| **All Classes** | 603 | **0.890** | **0.838** | **0.888** |
| **Safety Vest** | 601 | 0.915 | 0.904 | **0.942** |
| **Helmet** | 555 | 0.852 | 0.714 | 0.798 |

> **Analysis:** The system is exceptionally reliable at detecting safety vests (0.94 mAP). While helmet detection is slightly lower (0.79 mAP) due to smaller object scales in wide-angle site photos, the overall precision of 0.89 ensures a low false-alarm rate for automated monitoring.

---

## 🛠️ Tech Stack

* **Architecture:** YOLOv8 (Nano) — Optimized for edge deployment.
* **Framework:** PyTorch & Ultralytics.
* **Environment:** Data managed via Roboflow; trained on Google Colab.

---

## 📂 Implementation & Usage

The system is designed to be "plug-and-play" for developers or site engineers.

### 1. Installation

```bash
pip install ultralytics

from ultralytics import YOLO

# Load the production-ready weights
model = YOLO("best.pt")

# Predict and save results
model.predict("construction_site.jpg", save=True, conf=0.5)

