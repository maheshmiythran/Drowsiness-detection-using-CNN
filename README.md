🚗 Driver Drowsiness Detection System

A real-time computer vision system that monitors a driver’s eye state using a webcam and alerts them when signs of drowsiness are detected.

This project combines OpenCV, Haar Cascade classifiers, and a Convolutional Neural Network (CNN) to classify eye states (Open/Closed) and trigger an alert when prolonged eye closure is detected.

📌 Table of Contents

Overview

System Pipeline

CNN Architecture

Project Structure

Installation

How to Run

How the Algorithm Works

Future Improvements

🧠 Overview

The system continuously captures video frames from a webcam and processes them through the following stages:

Face detection

Eye detection

Eye state classification (Open/Closed)

Drowsiness scoring

Alarm triggering

If both eyes remain closed beyond a predefined threshold, an alarm sound is played to alert the driver.

🔁 System Pipeline
1️⃣ Frame Capture

Video frames are captured in real time using:

cv2.VideoCapture(0)

2️⃣ Face Detection

Frame converted to grayscale

Haar Cascade classifier applied

Region of Interest (ROI) extracted

3️⃣ Eye Detection

Eyes detected within face ROI

Eye regions cropped for classification

4️⃣ Eye Classification

Each eye image is:

Resized to 24 × 24 pixels

Normalized

Passed to trained CNN model (cnnCat2.h5)

Model predicts:

Open

Closed

5️⃣ Drowsiness Score Logic

Score increases when both eyes are closed

Score decreases when eyes are open

Alarm triggers when score exceeds threshold

🧩 CNN Architecture
🔹 Convolutional Layers

Conv2D – 32 filters (3×3)

Conv2D – 32 filters (3×3)

Conv2D – 64 filters (3×3)

🔹 Fully Connected Layers

Dense – 128 neurons

Output – 2 neurons (Softmax)

🔹 Activation Functions

ReLU – hidden layers

Softmax – output layer

📂 Project Structure
Driver-Drowsiness-Detection/
│
├── Drowsiness detection.py     # Main execution script
├── Model.py                    # CNN model definition and training
│
├── models/
│   └── cnnCat2.h5              # Pre-trained model
│
├── haar cascade files/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
│
└── alarm.wav                   # Alert sound

⚙️ Installation
🔹 Prerequisites

Python 3.6 (recommended)

Webcam

🔹 Install Dependencies
pip install opencv-python
pip install tensorflow
pip install keras
pip install pygame

▶️ How to Run

Navigate to the project directory and execute:

python drowsiness detection.py


The webcam window will open and real-time detection status will be displayed.

Press q to exit.

🔍 How the Algorithm Works

Capture real-time video frames.

Convert frame to grayscale.

Detect face using Haar Cascade.

Extract eye regions.

Preprocess eye images (resize + normalize).

Predict eye state using CNN.

Update drowsiness score.

Trigger alarm if threshold is crossed.

The system runs continuously until manually stopped.

🚀 Future Improvements

Train on a larger and more diverse dataset to improve generalization.

Replace Haar Cascades with deep-learning-based face detection.

Add:

Head tilt detection

Yawn detection

Multi-person monitoring

Optimize for deployment on embedded devices.

🛠 Tech Stack

Python

OpenCV

TensorFlow

Keras

Pygame

📚 References

OpenCV Documentation

TensorFlow Documentation

Keras Documentation
