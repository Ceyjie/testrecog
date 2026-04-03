#!/usr/bin/env python3
# test_depth.py — Test Orbbec Astra depth camera
import cv2
import sys
import numpy as np

print("=== Orbbec Astra Depth Camera Test ===\n")

try:
    sys.path.insert(0, '/home/medpal/pyorbbecsdk/sdk')
    import pyorbbecsdk
    from pyorbbecsdk import OBContext, OBPipeline, OBConfig, OBDevice
    from pyorbbecsdk import OBFrame, OBRGBFrame, OBDepthFrame
    HAS_ORBBEC = True
    print("✓ Orbbec SDK loaded")
except ImportError as e:
    HAS_ORBBEC = False
    print(f"✗ Orbbec SDK not available: {e}")
    print("  Install: pip install pyorbbecsdk")

print("\n=== Testing Depth Camera ===")

if HAS_ORBBEC:
    try:
        ctx = OBContext()
        pipeline = OBPipeline(ctx)
        config = OBConfig()
        
        profiles = pipeline.get_stream_profiles(1)  # 1 = depth sensor
        depth_profile = None
        for p in profiles:
            if p.width() == 640 and p.height() == 480 and "Y16" in str(type(p)):
                depth_profile = p
                break
        
        if depth_profile:
            config.enable_stream(depth_profile)
        else:
            config.enable_stream(profiles[0] if profiles else None)
        
        pipeline.start(config)
        print("✓ Orbbec depth stream started")
        
        for i in range(100):
            frames = pipeline.wait_for_frames(100)
            if frames:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    width = depth_frame.width
                    height = depth_frame.height
                    scale = depth_frame.get_value_scale()
                    
                    data = np.frombuffer(depth_frame.data(), dtype=np.uint16)
                    data = data.reshape((height, width))
                    
                    center_x, center_y = width // 2, height // 2
                    center_depth = data[center_y, center_x] * scale
                    
                    if i % 20 == 0:
                        print(f"  Frame {i}: {width}x{height}, center depth: {center_depth:.3f}m")
                    
                    depth_mm = (data * scale * 1000).astype(np.uint16)
                    depth_vis = np.clip(3000 - depth_mm, 0, 3000)
                    depth_vis = (depth_vis / 3000 * 255).astype(np.uint8)
                    
                    cv2.imshow("Depth", depth_vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        pipeline.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"✗ Orbbec error: {e}")
        HAS_ORBBEC = False

if not HAS_ORBBEC:
    print("\nUsing simulated depth (no Orbbec)\n")
    print("Testing depth estimation from face size...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)
    
    haar = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    
    KNOWN_FACE_WIDTH = 0.15  # Average face width in meters
    
    print("Look at camera - distance will be estimated from face size\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5, 0, (50, 50), (200, 200))
        
        h, w = frame.shape[:2]
        
        for (x, y, fw, fh) in faces:
            face_center_x = x + fw // 2
            face_center_y = y + fh // 2
            
            focal_length = w * 0.8
            face_width_px = fw
            if face_width_px > 0:
                distance = (KNOWN_FACE_WIDTH * focal_length) / (face_width_px / w)
                distance = max(0.3, min(5.0, distance))
            else:
                distance = 0
            
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distance:.2f}m", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
        
        cv2.putText(frame, f"Faces: {len(faces)} | Press 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Face Distance Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("\nDone!")
