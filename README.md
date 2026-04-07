# Custom Object Detection & Counting using YOLOv8

## Overview
This project implements a custom object detection system using YOLOv8 to detect and count bottles in real time through a webcam feed. It even helps in estimating the distance of objects from the camera.

The system is trained on a custom dataset and demonstrates real-time inference along with multi-object counting, making it suitable for applications like inventory monitoring and smart surveillance.

---

## Features
- Real-time object detection using webcam
- Multi-object detection ,counting and distance estimation
- Bounding box visualization with confidence scores
- Lightweight and runs on CPU

---

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch

---

## Results
- Custom dataset (40+ images)
- Achieved high detection accuracy (~98% mAP)
- Real-time performance on CPU

---

## Demo Video
[Watch Demo Video](https://drive.google.com/file/d/1tDKeNTTc_e1Loc9TLOcBoU2YRT9KnL5i/view?usp=drive_link)

---

## Project Structure
bottle-detection-yolov8/
├── dataset_sample/
├── count.py
├── distance.py   
├── test.py
├── README.md

---