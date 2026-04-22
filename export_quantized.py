#!/usr/bin/env python3
import sys
from pathlib import Path
from ultralytics import YOLO

print("Exporting INT8 quantized models...")

# Export detection model with INT8 quantization
print("\n=== Exporting YOLOv8n detection (INT8) ===")
model = YOLO('yolov8n.pt')
model.export(format='onnx', dynamic=True, simplify=True)
model.export(format='openvino', int8=True)
print("Done: yolov8n_openvino_model_int8/")

# Export pose model with INT8 quantization  
print("\n=== Exporting YOLOv8n-pose (INT8) ===")
model = YOLO('yolov8n-pose.pt')
model.export(format='onnx', dynamic=True, simplify=True)
model.export(format='openvino', int8=True)
print("Done: yolov8n-pose_openvino_model_int8/")

print("\n=== Quantized models ready! ===")
