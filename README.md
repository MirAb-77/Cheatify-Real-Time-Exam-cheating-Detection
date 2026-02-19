# Cheatify: Real Time Exam Cheating Surveillance System

## 📌 Overview

The **Cheating Surveillance System** is an AI-powered real-time proctoring solution designed to detect suspicious behavior during **online interviews and examinations**.

The system combines:

* 👁️ **Facial Landmark Detection** using dlib’s 68-point Shape Predictor
* 📱 **Object Detection** using YOLO
* 🧠 Intelligent behavior analysis for identifying potential cheating actions

It monitors:

* Head pose direction
* Eye gaze movement
* Unauthorized mobile phone usage

All detections are processed in real time using live video feed.

---

# 🚀 Key Features

### 👤 Head Pose Detection

* Tracks head orientation (left, right, up, down)
* Flags excessive or prolonged abnormal head movement

### 👁️ Eye & Pupil Tracking

* Uses 68 facial landmarks to estimate gaze direction
* Detects suspicious repeated gaze shifts

### 📱 Mobile Phone Detection

* Detects presence of mobile phones in the camera frame
* Model trained on the **Roboflow Cellphone Detection Dataset**

### ⚡ Real-Time Processing

* Works on live webcam feed
* Instant alert triggering system

### 🚨 Smart Alert System

Flags cheating when:

* Head turns beyond allowed duration
* Repeated gaze diversion
* Mobile phone appears in frame

---

# 🛠 Technologies Used

* **Python 3.8+**
* OpenCV – Video capture & processing
* dlib – Facial landmark detection
* YOLO – Real-time object detection
* Roboflow – Dataset & model training
* PyTorch – Model inference backend

---

# 📂 Project Structure

```
cheating-surveillance/
│
├── models/
│   ├── best_yolov8.pt
│   ├── best_yolov12.pt
│   └── shape_predictor_68_face_landmarks.dat
│
├── logs/                  # Saved detection screenshots
├── Demo_vid/              # (Optional: internal use only)
│
├── main.py                # Entry point
├── head_pose.py           # Head pose estimation logic
├── eye_movement.py        # Eye gaze tracking
├── mobile_detection.py    # YOLO phone detection
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 🔹 Prerequisites

Make sure you have installed:

* Python 3.8+
* pip
* Virtual environment (recommended)

Required libraries:

* OpenCV
* dlib
* torch
* numpy

---

## 🔹 Setup Steps

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sania-hasann/Cheating-Surveillance-System.git
cd Cheating-Surveillance-System
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Shape Predictor Model

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

Move the `.dat` file to the `models/` directory.

### 4️⃣ Add YOLO Weights

Download your trained YOLO weights from Roboflow and place them inside:

```
models/
```

---

# ▶️ Usage

Run the system:

```bash
python main.py
```

The webcam will start automatically, and real-time monitoring will begin.

---

# 🧠 System Workflow

## 🔄 Detection Pipeline

Below is the complete system workflow:

```
        Webcam Input
              │
              ▼
      Frame Extraction
              │
              ▼
 ┌────────────────────────────┐
 │  Facial Landmark Detection │
 │   (dlib - 68 landmarks)    │
 └────────────────────────────┘
              │
              ├──────────► Head Pose Estimation
              │
              └──────────► Eye Gaze Detection
              │
              ▼
 ┌────────────────────────────┐
 │     YOLO Object Detection  │
 │   (Mobile Phone Detection) │
 └────────────────────────────┘
              │
              ▼
      Behavior Analysis Engine
              │
              ▼
          Alert System
              │
              ▼
        Log Screenshot
```

---

## 📊 Logical Flow Diagram (Conceptual)

```
Input → Face Detection → Landmark Mapping → 
Head/Gaze Tracking → Object Detection → 
Suspicious Pattern Analysis → Alert
```

---

# 📦 Dataset

The mobile detection model was trained using the:

**Roboflow Cellphone Detection Dataset**

Access it here:
[https://universe.roboflow.com/d1156414/cellphone-0aodn](https://universe.roboflow.com/d1156414/cellphone-0aodn)

---

# 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a new branch

   ```
   git checkout -b feature-name
   ```
3. Commit changes

   ```
   git commit -m "Add feature"
   ```
4. Push changes

   ```
   git push origin feature-name
   ```
5. Open a Pull Request

---

# 🔐 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgments

* dlib
* OpenCV
* YOLO
* Roboflow

