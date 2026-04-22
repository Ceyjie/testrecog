import os
import cv2
import numpy as np
from ultralytics import YOLO

# Model configuration - use INT8 quantized for faster inference
MODEL_PATH = 'yolov8n-pose_int8_openvino_model/'

# Load OpenVINO model - Ultralytics auto-detects format from path
print(f"Loading YOLOv8 pose model with OpenVINO INT8...")
_model = YOLO(MODEL_PATH, task='pose')

_db = {}  # { name: signature }
_last_keypoints = None  # Cache for keypoints

def _get_signature(frame, get_kps=False):
    global _last_keypoints
    # Optimized inference
    res = _model(frame, verbose=False)
    if not res[0].keypoints or len(res[0].keypoints.xy) == 0:
        return None
    kp = res[0].keypoints.xy[0].numpy()
    if get_kps:
        _last_keypoints = kp
    if len(kp) < 13:
        return None
    # Body proportions: shoulder width, torso length, left arm length
    sh_w  = abs(kp[5][0] - kp[6][0])
    torso = abs(kp[5][1] - kp[11][1])
    arm   = abs(kp[5][1] - kp[9][1])
    sig = np.array([sh_w, torso, arm], dtype=np.float32)
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig

def register(name, frame):
    sig = _get_signature(frame)
    if sig is not None:
        _db[name] = sig
        return True
    return False

def recognize(frame, threshold=0.88):
    sig = _get_signature(frame, get_kps=True)
    if sig is None:
        return None
    for name, stored in _db.items():
        if np.dot(sig, stored) > threshold:
            return name
    return None

def get_db():
    return _db

def load_db(data):
    global _db
    _db = {k: np.array(v) for k, v in data.items()}

def get_keypoints(frame):
    """Get all keypoints in the frame for visualization"""
    res = _model(frame, verbose=False)
    if not res[0].keypoints or len(res[0].keypoints.xy) == 0:
        return None
    return res[0].keypoints.xy[0].numpy()