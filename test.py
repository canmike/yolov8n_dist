import ultralytics
from ultralytics import YOLO
import cv2
import torch
import numpy as np


model = YOLO("yolov8n.yaml")
model.load("yolov8n.pt")

# model = YOLO("yolov8n.pt")  # or yolov8s.pt

path = r"C:\Users\can.michael\Downloads\car.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x = (
    torch.from_numpy(img).unsqueeze(0).float().permute(0, 3, 1, 2)
)  # Convert to CHW format
x /= 255.0  # Normalize to [0, 1]
x.shape
# resize to (640, 640) if needed
if x.shape[2] != 640 or x.shape[3] != 640:
    x = torch.nn.functional.interpolate(x, size=(640, 640), mode="bilinear")
x.shape
y = model(x)
y[0]
y[0].boxes
