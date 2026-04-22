import cv2
import registration
import sys
import time
import face_reid
import pose_reid
import tracker

if len(sys.argv) < 2:
    print("Usage: python3 register_visual.py <name>")
    sys.exit(1)

name = sys.argv[1]
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print(f"Registering {name}. Stand in front of camera...")

countdown = 3
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    elapsed = time.time() - start_time
    remaining = max(0, countdown - int(elapsed))
    
    # Visualize detection
    persons = tracker.get_tracked_persons(frame)
    for p in persons:
        x1, y1, x2, y2 = p['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    keypoints = pose_reid.get_keypoints(frame)
    if keypoints is not None:
        for kp in keypoints:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)

    # Show countdown
    cv2.putText(frame, f"Capturing in {remaining}...", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Registering", frame)
    cv2.waitKey(1)
    
    # Auto-capture after countdown
    if remaining == 0:
        if registration.register_person(name, cap):
            print(f"Registered {name} successfully!")
        else:
            print("Failed - no person detected. Try again.")
        break

cap.release()
cv2.destroyAllWindows()
