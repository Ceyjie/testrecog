import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession('osnet_x0_25_msmt17.onnx')
_db = {}  # { name: embedding_vector }

def _get_embedding(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (128, 256))
    crop = crop.astype(np.float32) / 255.0
    crop = np.transpose(crop, (2, 0, 1))[np.newaxis]  # NCHW
    emb = session.run(None, {'input': crop})[0][0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

def register(name, frame, bbox):
    emb = _get_embedding(frame, bbox)
    if emb is not None:
        _db[name] = emb

def recognize(frame, bbox, threshold=0.7):
    emb = _get_embedding(frame, bbox)
    if emb is None:
        return None
    best_name, best_score = None, 0
    for name, stored in _db.items():
        score = float(np.dot(emb, stored))
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score > threshold else None

def get_db():
    return _db

def load_db(data):
    global _db
    _db = {k: np.array(v) for k, v in data.items()}
