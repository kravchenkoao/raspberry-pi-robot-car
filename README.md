\# Raspberry Pi Robot Car with Camera Streaming and Autopilot



A Raspberry Pi-based robot car project focused on camera streaming, FPV manual control, OpenCV-based autonomous driving, and machine learning autopilot experiments.



The project combines camera input, browser-based video monitoring, Python control logic, computer vision, dataset collection, model training, and motor/servo control.



\## Overview



This repository contains project materials for a Raspberry Pi robot car setup.



The project includes three main software parts:



1\. \*\*FPV Manual Control\*\* — manual control of the robot car with live camera video.

2\. \*\*OpenCV Autopilot\*\* — computer vision-based driving using lane detection.

3\. \*\*ML Autopilot\*\* — dataset collection, model training, and running a trained model on Raspberry Pi.



\## Key Features



\- Raspberry Pi camera capture using Picamera2

\- MJPEG video streaming to a browser

\- FPV manual control

\- OpenCV-based line and lane detection

\- Steering control based on camera image processing

\- Dataset collection for machine learning

\- Training a custom autopilot model

\- Running a trained model on Raspberry Pi

\- Motor and servo control integration

\- Demonstration videos for FPV, OpenCV autopilot, and ML autopilot



\## Technologies and Tools



\- Raspberry Pi

\- Python

\- Picamera2

\- OpenCV

\- MJPEG streaming

\- TensorFlow / Keras

\- TensorFlow Lite

\- Browser-based video stream

\- Motor control

\- Servo control

\- Remote control logic



\## System Architecture



```text

Pi Camera → Raspberry Pi → MJPEG Stream → Browser / Control Logic → Motor \& Servo

```



\## Project Structure



```text

donkeyCar\_app\_materials/

├── prepSteps/

├── finalStep/

│   ├── dataset/

│   └── models/

│

donkeyCar\_demonstrations/

├── FPV/

├── openCV/

└── ml/

```



\## Main Software Parts



\### 1. FPV Manual Control



The FPV part is used for manual control of the robot car with live camera preview.



Main file:



```text

donkeyCar\_app\_materials/finalStep/fpv\_record.py

```



This program handles:



\- camera input

\- live video stream

\- manual driving control

\- recording of demonstration videos



This part was used to test the basic camera stream, control behavior, and manual driving of the robot car.



\---



\### 2. OpenCV Autopilot



The OpenCV autopilot uses camera frames to detect lane lines and control the steering direction.



Main file:



```text

donkeyCar\_app\_materials/finalStep/opencv\_autopilot\_record.py

```



This program handles:



\- receiving frames from the camera

\- detecting one or two lane lines

\- calculating the center of the driving path

\- converting image processing results into steering commands

\- recording OpenCV autopilot demonstrations



The OpenCV part includes both single-line following and two-line lane following experiments.



\---



\### 3. ML Autopilot



The machine learning autopilot is based on collecting training data, training a custom model, and running this model on the Raspberry Pi.



Main files:



```text

donkeyCar\_app\_materials/finalStep/data\_recorder.py

donkeyCar\_app\_materials/finalStep/train\_model.py

donkeyCar\_app\_materials/finalStep/ml\_autopilot.py

```



File purposes:



\- `data\_recorder.py` — collects camera frames and corresponding manual control commands.

\- `train\_model.py` — trains a custom autopilot model using the collected dataset.

\- `ml\_autopilot.py` — runs the trained model on Raspberry Pi and controls the robot car based on model predictions.



The trained models are stored in:



```text

donkeyCar\_app\_materials/finalStep/models/

```



The dataset is stored in:



```text

donkeyCar\_app\_materials/finalStep/dataset/

```



\## Development and Test Files



The `prepSteps/` folder contains test scripts used during the gradual development of the OpenCV autopilot.



```text

donkeyCar\_app\_materials/prepSteps/

```



Main test files:



\- `orange\_debug.py` — tests orange line detection from the camera.

\- `orange\_steering\_test.py` — tests conversion of detected line position into steering commands.

\- `lane\_steering\_roi\_test.py` — tests steering based on a selected region of interest.

\- `lane\_two\_contours\_debug.py` — tests detection of two lane lines.

\- `autopilot\_follow.py` — intermediate OpenCV autopilot version for following one line.



These files are not the final programs, but they show the development and debugging process.



\## Dataset and Models



The `dataset/` folder contains training images and control data used for model training.



```text

donkeyCar\_app\_materials/finalStep/dataset/

```



The `models/` folder contains trained models and training results.



```text

donkeyCar\_app\_materials/finalStep/models/

```



Model files include:



\- `.tflite` models for running on Raspberry Pi

\- `.keras` models for full Keras versions

\- saved best model versions

\- training loss and MAE plots

\- preprocessing configuration files



Example model groups:



```text

manual\_model.\*

lane\_model.\*

```



\## Demonstrations



Demo materials are stored in:



```text

donkeyCar\_demonstrations/

```



The demonstrations show the work of all three main project parts:



\- FPV manual control

\- OpenCV autopilot

\- ML autopilot



\### FPV Demonstrations



```text

donkeyCar\_demonstrations/FPV/

```



Includes:



\- final FPV control video

\- web interface test recordings

\- manual driving demonstrations



\### OpenCV Autopilot Demonstrations



```text

donkeyCar\_demonstrations/openCV/

```



Includes:



\- one-line following demonstration

\- two-line lane following demonstration

\- web interface captures

\- OpenCV debugging screenshots



\### ML Autopilot Demonstrations



```text

donkeyCar\_demonstrations/ml/

```



Includes:



\- ML autopilot trained on manual control data

\- ML autopilot trained on OpenCV-generated data

\- web interface recording of ML autopilot launch



\## Demo



A short demo video is available here:



\[Open demo video](ADD\_YOUR\_DEMO\_VIDEO\_LINK\_HERE)



The demo shows:



\- Raspberry Pi robot car hardware setup

\- camera stream from the Pi camera

\- browser-based monitoring

\- FPV manual control

\- OpenCV autopilot behavior

\- ML autopilot testing



\## Images



Add screenshots and diagrams here:



```markdown

!\[Robot car setup](images/robot-car-photo.png)



!\[Camera stream output](images/camera-stream.png)



!\[System block diagram](images/block-diagram.png)

```



\## What This Project Demonstrates



This project demonstrates practical experience with Raspberry Pi-based robotics, camera streaming, Python scripting, OpenCV image processing, machine learning workflow, dataset collection, model training, and integration between software and physical motor/servo control.



\## Project Materials



Full project files, screenshots, videos and documentation are available here:



\[Open project folder](ADD\_YOUR\_GOOGLE\_DRIVE\_LINK\_HERE)

