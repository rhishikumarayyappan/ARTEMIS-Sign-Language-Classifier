# ARTEMIS: Portable Sign Language & Emotion Recognition for Edge Devices

![Live demo snapshot](Final_Demo_UI.png)  
*Real-time demonstration: portable sign language and emotion recognition from webcam, deployable on any standard computer (CPU-only).*

---

## 🚀 Overview

**ARTEMIS** is an MSc AI capstone project for practical accessibility:  
A privacy-focused, reproducible, and portable system for **American Sign Language (ASL) word recognition** and **emotion analysis**—requiring only a webcam and CPU.

- **Keypoint-based sign recognition:** Fast, accurate using hand, body, and face landmarks (no raw video needed).
- **Optional emotion inference:** Real-time emotion context from user expressions.

---

## 🛠️ Technologies & Architecture

- **[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)**: Extracts pose, hand, and face keypoints live.
- **Bidirectional GRU classifier**: Calibrated, lightweight, exported to ONNX (CPU).
- **Confidence calibration & abstention**: Reliable predictions using temperature scaling and threshold logic.
- **Emotion model**: Frame-level, robust to linguistic facial signals.

---

## 🎯 Results

- **Sign accuracy**: Top-1: 0.692, Top-3: 0.923 — signer-independent (WLASL, 10-class slice).
- **Reliability**: Abstention system—100% correct for kept-predictions at 54% coverage.
- **Performance**: ~27ms model inference, 149ms end-to-end per clip (CPU-only).
- **Privacy**: Only keypoints processed—no video stored/transferred.

---

## 📦 Quick Start

1. **Clone this repository**
2. **Install dependencies**  
poetry install --no-root


3. **Run calibrated demo UI**  
poetry run python scripts/live_demo_stable.py



---

## 📝 Design Notes

- Early **YOLO + MMPose** (offline use only)—final runtime is keypoint/GRU/ONNX pipeline.
- **No GPU needed**—CPU performance tested on everyday ultrabook (Core i5, 16GB RAM).
- **Calibrated abstention**—"Not Sure" output for low-confidence, maximizing trustworthiness.
- **Dataset**: WLASL (10-class, signer-independent selection)—documented for reproducibility.

---

## 📖 Reference

For full methodology and evaluation:
**A Portable Sign Language Recognizer with Emotion Analysis for Edge Devices**  
*MSc Thesis, Rhishi Kumar Ayyappan, University of Galway (2025)*

---

## 💡 About the Author

**Rhishi Kumar Ayyappan**  
MSc Computer Science (AI) | University of Galway  
Passionate about edge AI, accessibility, and machine learning for real-world impact.





