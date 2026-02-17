<<<<<<< HEAD
Driver Drowsiness Detection System
Overview

This project implements a real-time Driver Drowsiness Detection System that monitors a driver’s eye state through a webcam feed. The system detects facial features using OpenCV and classifies eye status (Open/Closed) using a Convolutional Neural Network (CNN). If the eyes remain closed beyond a defined threshold, an audible alarm is triggered to alert the driver.

The objective is to reduce accident risks by identifying early signs of fatigue using computer vision and deep learning.

System Architecture

The detection pipeline operates in five sequential stages:

1. Frame Acquisition

Live video frames are captured using:

cv2.VideoCapture(0)


Each frame is processed individually in real time.

2. Face Localization

Frames are converted to grayscale to improve detection efficiency. A Haar Cascade Classifier is applied using detectMultiScale() to locate faces. Bounding boxes define the Region of Interest (ROI).

3. Eye Extraction

Within the detected face ROI, a second Haar Cascade model identifies eye regions. These eye crops are isolated and prepared for classification.

4. Eye State Classification

Each detected eye image undergoes preprocessing:

Resize to 24 × 24 pixels

Normalize pixel values

Reshape for CNN input

The processed image is passed into the trained CNN model (cnnCat2.h5) which predicts:

Open

Closed

5. Drowsiness Scoring Mechanism

A running score tracks how long both eyes remain closed.

Score increases when eyes are detected as closed.

Score decreases when eyes are open.

If the score exceeds a predefined threshold, an alarm (alarm.wav) is triggered using the Pygame library.

CNN Model Architecture

The Convolutional Neural Network is structured as follows:

Convolutional Layers

32 filters, kernel size 3 × 3

32 filters, kernel size 3 × 3

64 filters, kernel size 3 × 3

Fully Connected Layers

Dense layer with 128 neurons

Output layer with 2 neurons (Softmax activation)

Activation Functions

ReLU: Used in all hidden layers

Softmax: Used in the output layer for binary classification

Project Requirements
Hardware

Webcam for real-time image capture

Software

Python 3.6 (recommended)

Install required libraries:

=======
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
>>>>>>> c776dbe71738f04e1727b9e6d93f6598b0fddf0f
pip install opencv-python
pip install tensorflow
pip install keras
pip install pygame

<<<<<<< HEAD
Project Structure

Model.py
Contains CNN architecture definition and training logic.

Drowsiness detection.py
Main script that runs the detection pipeline.

models/cnnCat2.h5
Pre-trained CNN model.

haar cascade files/
XML files for face and eye detection.

alarm.wav
Alert sound triggered during drowsiness detection.

How It Works (Algorithm Summary)

Capture live video frames.

Detect faces using Haar Cascade.

Extract eye regions from detected face.

Classify eye state using trained CNN.

Update drowsiness score.

Trigger alarm when threshold is exceeded.

The system runs continuously until manually terminated.

Running the Application

Navigate to the project directory and execute:

python drowsiness detection.py


The webcam feed will open and real-time detection status will be displayed.

Future Enhancements

Improve model generalization using larger and more diverse datasets.

Integrate additional fatigue indicators such as:

Head tilt detection

Yawning detection

Extend support for multi-person monitoring in shared environments.

Optimize inference speed for embedded deployment in vehicles.

Acknowledgments

OpenCV Documentation

Keras Documentation

TensorFlow Documentation
=======
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
>>>>>>> c776dbe71738f04e1727b9e6d93f6598b0fddf0f
