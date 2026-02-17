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

pip install opencv-python
pip install tensorflow
pip install keras
pip install pygame

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
