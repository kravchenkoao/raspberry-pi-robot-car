# Raspberry Pi Robot Car with Camera Streaming and Autopilot

A Raspberry Pi-based robot car project focused on camera streaming, FPV manual control, OpenCV-based autonomous driving, and machine learning autopilot experiments.

The project combines camera input, browser-based monitoring, Python control logic, computer vision, dataset collection, model training, and motor/servo control.

## Main Project Parts

The project includes three main software parts:

1. **FPV Manual Control**  
   Manual control of the robot car with live camera video.  
   Main file: `src/fpv_record.py`

2. **OpenCV Autopilot**  
   Computer vision-based autopilot that detects one or two lane lines, calculates the driving path, and controls steering.  
   Main file: `src/opencv_autopilot_record.py`

3. **ML Autopilot**  
   Machine learning-based autopilot that includes dataset collection, model training, and running the trained model on Raspberry Pi.  
   Main files: `src/data_recorder.py`, `src/train_model.py`, `src/ml_autopilot.py`

## Key Features

- Raspberry Pi camera capture using Picamera2
- MJPEG video streaming to a browser
- FPV manual control
- OpenCV-based line and lane detection
- Dataset collection for machine learning
- Custom model training with TensorFlow / Keras
- TensorFlow Lite model execution on Raspberry Pi
- Motor and servo control integration

## Technologies and Tools

- Raspberry Pi
- Python
- Picamera2
- OpenCV
- MJPEG streaming
- TensorFlow / Keras
- TensorFlow Lite
- Motor control
- Servo control

## System Architecture

```text
Pi Camera → Raspberry Pi → MJPEG Stream → Browser / Control Logic → Motor & Servo
```

![System block diagram](images/block-diagram.png)

## Project Structure

```text
src/
├── fpv_record.py
├── opencv_autopilot_record.py
├── data_recorder.py
├── train_model.py
├── ml_autopilot.py
├── prepSteps/
├── dataset/
└── models/

images/
├── robot-car-photo.png
├── camera-stream.png
└── block-diagram.png
```

## Program Descriptions

### `src/fpv_record.py`

Final FPV manual control program.  
It handles camera input, live video stream, manual driving control, and recording of demonstration videos.

### `src/opencv_autopilot_record.py`

Final OpenCV autopilot program.  
It processes camera frames, detects lane lines, calculates the center of the path, and converts the result into steering commands.

### ML Autopilot Files

- `src/data_recorder.py` — collects camera frames and corresponding manual control commands.
- `src/train_model.py` — trains a custom autopilot model using the collected dataset.
- `src/ml_autopilot.py` — runs the trained model on Raspberry Pi and controls the robot car based on model predictions.

## Development Files

The `src/prepSteps/` folder contains test scripts used during the development of the OpenCV autopilot, including line detection, steering logic, ROI testing, and two-line detection experiments.

## Dataset and Models

The `src/dataset/` folder contains training images and control data.  
The `src/models/` folder contains trained `.tflite` and `.keras` models, training graphs, and preprocessing configuration files.

## Demonstrations

Demo videos and screenshots are available here:

[Open demonstration materials](https://drive.google.com/drive/folders/1VzoefyW_gy8uRJTU7gW_DrXXvEAA0x7D?usp=sharing)

The demonstrations show FPV manual control, OpenCV autopilot, and ML autopilot testing.

## Images

Robot Car setup:

![Robot car setup](images/robot-car-photo.png)

Camera Stream Output:
![Camera stream output](images/camera-stream.png)

## What This Project Demonstrates

This project demonstrates practical experience with Raspberry Pi-based robotics, camera streaming, Python scripting, OpenCV image processing, machine learning workflow, dataset collection, model training, and integration between software and physical motor/servo control.

## Project Materials

Full project files, screenshots, videos, and documentation are available here:

[Open project folder](https://drive.google.com/drive/folders/1dkqSVrvkGmvXaaQFwxH7gmkEv91YAHWD?usp=sharing)
