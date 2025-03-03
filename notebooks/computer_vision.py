"""
Comprehensive Computer Vision in Python

This script covers various advanced computer vision techniques, including:
- Image preprocessing
- Edge detection and segmentation
- Object detection (YOLO, Haar Cascades, Faster R-CNN)
- Image classification using CNNs and transfer learning
- Image augmentation techniques
- Optical character recognition (OCR)
"""

import cv2
import numpy as np

# ----------------------------
# 1. Image Preprocessing
# ----------------------------

# Load an image and convert to grayscale
image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Resize image
resized_image = cv2.resize(image, (224, 224))

# ----------------------------
# 2. Edge Detection & Segmentation
# ----------------------------

# Canny edge detection
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

# Adaptive thresholding for segmentation
thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# ----------------------------
# 3. Object Detection using YOLO and Haar Cascades
# ----------------------------

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
detections = net.forward(output_layers)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# ----------------------------
# 4. Image Classification using CNNs and Transfer Learning
# ----------------------------

from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights="imagenet")

# Transfer Learning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ----------------------------
# 5. Image Augmentation
# ----------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
augmented_image = datagen.random_transform(image)

# ----------------------------
# 6. Optical Character Recognition (OCR)
# ----------------------------

import pytesseract

text = pytesseract.image_to_string(gray_image)
print("Extracted Text:", text)
