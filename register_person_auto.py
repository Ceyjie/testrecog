#!/usr/bin/env python3
# register_person_auto.py — GUI‑free face registration

import cv2, os, argparse, json, time

PERSONS_DIR = "/home/medpal/MedPalRobotV2/data/persons"
REGISTRY    = os.path.join(PERSONS_DIR, "registry.json")
HAAR_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
CAPTURES    = 5
DELAY_SEC   = 1.0   # pause between captures

os.makedirs(PERSONS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Register a person for MedPal tracking")
parser.add_argument("--name", required=True, help="Person name (spaces allowed)")
args = parser.parse_args()
name = args.name.strip().replace(" ", "_")   # replace spaces with underscores
print(f"\n=== Registering: {name} ===")
print("Look at the camera. The script will capture 5 face images automatically.")
print("Move your head slightly between captures.\n")

haar = cv2.CascadeClassifier(HAAR_PATH)
if haar.empty():
    print(f"ERROR: Could not load Haar cascade from {HAAR_PATH}")
    exit(1)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

saved = []
count = 0

while count < CAPTURES:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 5, 0, (80, 80))

    if len(faces) == 1:
        x, y, w, h = faces[0]
        # Add padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 128))
        path = os.path.join(PERSONS_DIR, f"{name}_{count}.jpg")
        cv2.imwrite(path, crop)
        saved.append(path)
        count += 1
        print(f"  Captured {count}/{CAPTURES} – saved to {path}")
        time.sleep(DELAY_SEC)   # allow user to move
    else:
        print(f"  Waiting for face... (detected {len(faces)} faces)", end="\r")

cap.release()
print("\n")

if count < CAPTURES:
    print(f"Only captured {count}/{CAPTURES}. Registration incomplete.")
    exit(1)

# Update registry
reg = {}
if os.path.exists(REGISTRY):
    with open(REGISTRY) as f:
        reg = json.load(f)

reg[name] = {
    "name":   name,
    "images": saved,
    "count":  count
}

with open(REGISTRY, "w") as f:
    json.dump(reg, f, indent=2)

print(f"✓ Registered '{name}' with {count} captures.")
print(f"\nNow you can make MedPal follow {name}:")
print(f"  Web UI: click the '{name}' pill")
print(f"  Voice:  say 'MedPal follow me' (while being the target)")
print(f"  API:    curl http://localhost:5000/set_target/{name}")
