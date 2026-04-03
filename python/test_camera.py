#!/usr/bin/env python3
# test_camera.py — Test camera feed
import cv2
import sys

print("Testing camera...")

# Try devices 0, 1, 2
for device in [0, 1, 2]:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if cap.isOpened():
        # Try MJPG format
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera opened on device {device}: {w}x{h}")
        
        # Show one frame
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured: {frame.shape}")
            cv2.imwrite("/home/medpal/MedPalRobotV2/test_camera.jpg", frame)
            print("Saved to test_camera.jpg")
        else:
            print("✗ Failed to capture frame")
        
        cap.release()
        sys.exit(0)
    else:
        print(f"Device {device} not available")

print("✗ No camera found")
sys.exit(1)
