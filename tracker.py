import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Check for OpenVINO availability
try:
    import openvino as ov
    print("✅ OpenVINO Python package available")
except ImportError:
    print("⚠️ OpenVINO not found, using PyTorch")
    ov = None

# Model configuration - use INT8 quantized for faster inference
MODEL_PATH = 'yolov8n_int8_openvino_model/'

# Load OpenVINO model - Ultralytics auto-detects format from path
print(f"Loading YOLOv8 detection model with OpenVINO INT8...")
model = YOLO(MODEL_PATH, task='detect')

# Tracking history for trajectory
track_history = defaultdict(lambda: [])

def get_tracked_persons(frame):
    """
    Returns list of dicts: [{'id': int, 'bbox': (x1,y1,x2,y2)}, ...]
    Track ID is persistent across frames — same person = same ID
    
    Optimized inference loop similar to LearnOpenCV article
    """
    global track_history
    
    start_time = time.time()
    
    # Inference - using optimized settings
    # Ultralytics handles device selection internally
    results = model.track(
        frame,
        classes=[0],          # 0 = person only
        conf=0.5,             
        tracker='bytetrack.yaml',
        persist=True,         
        verbose=False
    )
    
    persons = []
    if results[0].boxes.id is not None:
        for box, tid in zip(results[0].boxes.xyxy,
                            results[0].boxes.id.int().tolist()):
            x1, y1, x2, y2 = map(int, box)
            persons.append({'id': tid, 'bbox': (x1, y1, x2, y2)})
            
            # Update trajectory history
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 30: # Limit history length
                track_history[tid].pop(0)
    
    # Calculate FPS
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    
    return persons

def get_trajectory(track_id):
    """Get trajectory points for a specific track ID"""
    return track_history.get(track_id, [])

def get_fps():
    """Get current model FPS"""
    return 0 # Placeholder