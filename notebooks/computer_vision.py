"""
Computer Vision in Python

This script covers various essential computer vision techniques, including:
- Image preprocessing
- Object detection using OpenCV and YOLO
- Image classification using CNNs
- Transfer learning for computer vision tasks
"""

import cv2
import numpy as np

# ----------------------------
# 1. Image Preprocessing
# ----------------------------

# Load image and convert to grayscale
image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized_image = cv2.resize(image, (224, 224))

# ----------------------------
# 2. Object Detection using OpenCV
# ----------------------------

# Load a pre-trained object detection model (YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Perform object detection
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
detections = net.forward(output_layers)

# ----------------------------
# 3. Image Classification using CNNs
# ----------------------------

from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights="imagenet")

# ----------------------------
# 4. Transfer Learning for Computer Vision
# ----------------------------

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
