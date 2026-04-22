import numpy as np
from insightface.app import FaceAnalysis

_app = FaceAnalysis(name='buffalo_sc')
_app.prepare(ctx_id=-1)  # -1 = CPU
_db = {}  # { name: embedding }

def register(name, frame):
    faces = _app.get(frame)
    if faces:
        emb = faces[0].embedding
        _db[name] = emb / np.linalg.norm(emb)
        return True
    return False  # no face detected

def recognize(frame, bbox=None, threshold=0.35):
    # If bbox is provided, crop the frame to the region of interest
    if bbox:
        x1, y1, x2, y2 = bbox
        # Ensure bounds are within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        frame = frame[y1:y2, x1:x2]
        
    faces = _app.get(frame)
    if not faces:
        return None
    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)
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
