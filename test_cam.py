import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    ret, frame = cap.read()
    if ret:
        print("Camera working! Frame captured.")
    else:
        print("Error: Could not read frame.")
cap.release()
