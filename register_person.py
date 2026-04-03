#!/usr/bin/env python3
# register_person.py — Better face registration with more samples
import cv2, os, argparse, json, time

PERSONS_DIR = "/home/medpal/MedPalRobotV2/data/persons"
REGISTRY    = os.path.join(PERSONS_DIR, "registry.json")
HAAR_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--samples", type=int, default=15)
args = parser.parse_args()

name = args.name.strip().replace(" ", "_")
num_samples = args.samples

print(f"\n=== Registering: {name} ===")
print(f"Capturing {num_samples} samples with variety...")
print("- Look straight, left, right, up, down")
print("- Vary distance (close/far)")
print("- Different lighting if possible\n")

os.makedirs(PERSONS_DIR, exist_ok=True)

haar = cv2.CascadeClassifier(HAAR_PATH)
if haar.empty():
    print("ERROR: Cannot load face cascade")
    exit(1)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

saved = []
count = 0
wait = 0

print("Look at camera and move your face slightly...\n")

try:
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5, 0, (60, 60), (200, 200))

        display = frame.copy()
        
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        if len(faces) == 1 and wait <= 0:
            x, y, w, h = faces[0]
            pad = 15
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad)
            y2 = min(frame.shape[0], y+h+pad)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (200, 200))
            path = os.path.join(PERSONS_DIR, f"{name}_{count}.jpg")
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(path)
            count += 1
            wait = 25
            print(f"  ✓ {count}/{num_samples}: captured (face size: {w}x{h})")

        wait -= 1
        
        status = f"Captures: {count}/{num_samples}"
        if len(faces) == 0:
            hint = "No face - look at camera"
            cv2.putText(display, hint, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(faces) > 1:
            hint = "Multiple faces - only you"
            cv2.putText(display, hint, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            hint = f"Next in {max(0,wait)}..."
            cv2.putText(display, hint, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Register", display)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nCancelled.")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

cap.release()
cv2.destroyAllWindows()

if count < num_samples:
    print(f"\nOnly captured {count}/{num_samples}")
    if count < 3:
        print("Need at least 3 samples. Try again.")
        exit(1)

reg = {}
if os.path.exists(REGISTRY):
    with open(REGISTRY) as f:
        reg = json.load(f)

reg[name] = {"name": name, "images": saved, "count": count}

with open(REGISTRY, "w") as f:
    json.dump(reg, f, indent=2)

print(f"\n✓ Registered '{name}' with {count} samples")

os.remove(os.path.join(PERSONS_DIR, "recognizer.yml")) if os.path.exists(os.path.join(PERSONS_DIR, "recognizer.yml")) else None

print("\nRe-training recognizer...")
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))
    import face_recognizer
    face_recognizer.get_recognizer().retrain()
    print("✓ Recognizer trained!")
except Exception as e:
    print(f"Note: {e}")

print(f"\nTo start robot: python3 python/main.py")
