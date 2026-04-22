import json
import os
import numpy as np
import cv2
import body_reid
import face_reid
import pose_reid
import tracker

PERSONS_FILE = 'persons.json'

def register_person(name, cap):
    """
    Captures 10 frames from cap, builds all 3 embeddings.
    Saves to persons.json on disk.
    """
    print(f"Registering {name} — stand in front of camera for 3 seconds...")
    frames = []
    for _ in range(15):  # skip first 15 frames (camera warm-up)
        cap.read()
    for _ in range(10):  # Capture 10 frames
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # Get persons from all frames and use the one with best detection
    best_bbox = None
    best_person = None
    for f in frames:
        persons = tracker.get_tracked_persons(f)
        if persons:
            best_person = persons[0]
            best_bbox = best_person['bbox']
            break
    
    if not best_bbox:
        print("No person detected — try again")
        return False
    bbox = best_bbox

    # Build all three embeddings - average over multiple frames for robustness
    body_embs = []
    face_registered = False
    pose_registered = False
    
    for f in frames:
        # Body embedding
        emb = body_reid._get_embedding(f, bbox)
        if emb is not None:
            body_embs.append(emb)
        
        # Face and pose - use best frame
        if not face_registered:
            face_registered = face_reid.register(name, f)
        if not pose_registered:
            pose_registered = pose_reid.register(name, f)
    
    if body_embs:
        body_reid._db[name] = np.mean(body_embs, axis=0)

    # Save to disk
    db = {}
    if os.path.exists(PERSONS_FILE):
        with open(PERSONS_FILE, 'r') as f:
            db = json.load(f)
    db[name] = {
        'body': body_reid._db.get(name, np.array([])).tolist(),
        'face': face_reid._db.get(name, np.array([])).tolist(),
        'pose': pose_reid._db.get(name, np.array([])).tolist(),
    }
    with open(PERSONS_FILE, 'w') as f:
        json.dump(db, f)
    print(f"Registered {name}: body={bool(body_embs)}, face={face_registered}, pose={pose_registered}")
    return True

def load_all():
    """Load saved person database into all three ReID modules."""
    if not os.path.exists(PERSONS_FILE):
        return
    with open(PERSONS_FILE, 'r') as f:
        db = json.load(f)
    body_reid.load_db({k: v['body'] for k, v in db.items() if v['body']})
    face_reid.load_db({k: v['face'] for k, v in db.items() if v['face']})
    pose_reid.load_db({k: v['pose'] for k, v in db.items() if v['pose']})
    print(f"Loaded {len(db)} registered persons: {list(db.keys())}")
