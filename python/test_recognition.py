#!/usr/bin/env python3
# test_recognition.py — Test face recognition
import cv2, os, json, numpy as np

PERSONS_DIR = "/home/medpal/MedPalRobotV2/data/persons"
HAAR_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

print("=== Face Recognition Test ===\n")

# Load Haar
haar = cv2.CascadeClassifier(HAAR_PATH)
if haar.empty():
    print("ERROR: Cannot load face cascade")
    exit(1)
print("✓ Haar cascade loaded")

# Load registry
registry_path = f"{PERSONS_DIR}/registry.json"
if not os.path.exists(registry_path):
    print("ERROR: No registry found")
    exit(1)

with open(registry_path) as f:
    registry = json.load(f)

print(f"✓ Registry loaded: {list(registry.keys())}\n")

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100)
faces, labels = [], []
label_to_name = {}
next_label = 0

for name, data in registry.items():
    label = next_label
    label_to_name[label] = name
    next_label += 1
    
    for img_path in data.get("images", []):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(img)
            labels.append(label)
            print(f"  Loaded: {os.path.basename(img_path)}")

if faces:
    recognizer.train(faces, np.array(labels))
    print(f"\n✓ Trained with {len(faces)} images, {len(label_to_name)} persons")
    print(f"  Labels: {label_to_name}")
else:
    print("ERROR: No training images")
    exit(1)

# Test on camera
print("\n=== Testing on Camera ===")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = haar.detectMultiScale(gray, 1.1, 5, 0, (50, 50), (200, 200))
    
    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y+h, x:x+w]
        
        try:
            label, conf = recognizer.predict(face_roi)
            if conf < 80:
                name = label_to_name.get(label, "unknown")
                color = (0, 255, 0)
                text = f"{name} ({conf:.0f}%)"
            else:
                color = (100, 100, 255)
                text = "Unknown"
        except Exception as e:
            color = (100, 100, 255)
            text = f"Error: {e}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Faces: {len(faces_detected)} | Press 'q' to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Face Recognition Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDone!")
