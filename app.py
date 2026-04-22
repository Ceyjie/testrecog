from flask import Flask, render_template, jsonify, Response
import motor_control as motors
import atexit
import cv2
import threading
import time
import occupancy_map
import tracker
import body_reid
import face_reid
import pose_reid
import registration

app = Flask(__name__)

# Shared state
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 416)  # Reduced from 640 for faster processing
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # Reduced from 480
camera.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce load
frame_lock = threading.Lock()
latest_frame = None

# Config & State
import config
import json
import os

target_track_id = None
track_id_to_name = {} # Mapping for display
current_action = "IDLE"
follow_mode = False # OFF by default

# Optimization: throttle expensive operations
frame_count = 0
RECOGNITION_INTERVAL = 1  # Run recognition every frame for accuracy

registration.load_all()

# Helper to get registered persons
def get_persons():
    if os.path.exists('persons.json'):
        with open('persons.json', 'r') as f:
            return json.load(f).keys()
    return []

def tracking_loop():
    global latest_frame, target_track_id, current_action, frame_count
    loop_start_time = time.time() # Initialize for FPS calculation
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        start_time = time.time() # Track individual frame time
        frame_count += 1
            
        # Tracking logic - always run (needed for follow)
        persons = tracker.get_tracked_persons(frame)
        grid = ['FREE'] * config.GRID_COLS
        
        # Cache keypoints to avoid running pose model twice
        keypoints = None
        keypoints_ran = False
        
        # Recognition throttling - only run expensive ops every Nth frame
        run_recognition = (frame_count % RECOGNITION_INTERVAL == 0)
        
        # Identify target: face first to find the right person, then track by body
        identified_persons = []  # Persons whose face matches target
        
        for person in persons:
            # Check face recognition to identify the target person
            if run_recognition:
                face_name = face_reid.recognize(frame, bbox=person['bbox'])
                if face_name:
                    track_id_to_name[person['id']] = face_name
                    identified_persons.append(person['id'])
                    
                    # If this is the target, set as target track ID
                    if face_name == config.TARGET_NAME:
                        target_track_id = person['id']
        
        # If target found by face, no need to re-identify - just track body
        if target_track_id is None:
            # No face match - check if we have an existing target tracked by body
            for person in persons:
                if person['id'] in track_id_to_name:
                    if track_id_to_name[person['id']] == config.TARGET_NAME:
                        target_track_id = person['id']
                        break
        
        # Fallback: if still no target, try body/pose recognition
        if target_track_id is None and run_recognition:
            for person in persons:
                name = body_reid.recognize(frame, person['bbox'])
                if not name:
                    if not keypoints_ran:
                        name = pose_reid.recognize(frame)
                        keypoints = pose_reid._last_keypoints
                        keypoints_ran = True
                if name:
                    track_id_to_name[person['id']] = name
                    if name == config.TARGET_NAME:
                        target_track_id = person['id']
                        break
            
            # (assignment handled inside for-loop above)
        
        # Draw bounding boxes and names
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            name = track_id_to_name.get(person['id'], "Unknown")
            color = (0, 0, 255) if name == config.TARGET_NAME else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} (ID: {person['id']})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Pose Keypoints - use cached keypoints from pose_reid.recognize()
        if keypoints is not None:
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        with frame_lock:
            latest_frame = frame.copy()
            
        target = next((p for p in persons if p['id'] == target_track_id), None)
        
        # Motor Control & Status
        if follow_mode and target:
            x1, y1, x2, y2 = target['bbox']
            cx = (x1 + x2) // 2
            deviation = cx - config.FRAME_CENTER_X
            
            if grid[config.OBSTACLE_COL] == 'BLOCKED': 
                motors.stop()
                current_action = "OBSTACLE STOP"
            elif deviation < -config.DEVIATION_THRESH: 
                motors.left()
                current_action = "TURNING LEFT"
            elif deviation > config.DEVIATION_THRESH: 
                motors.right()
                current_action = "TURNING RIGHT"
            else: 
                motors.forward()
                current_action = "MOVING FORWARD"
        elif follow_mode:
            if target_track_id is not None: 
                target_track_id = None
            motors.stop()
            current_action = "SEARCHING..."
        else:
            motors.stop()
            current_action = "MANUAL MODE"
        
        # Draw status on frame
        cv2.putText(frame, current_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # FPS Display
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display = int(1.0 / elapsed)
            cv2.putText(frame, f"FPS: {fps_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        time.sleep(0.05)

# Start tracking thread
threading.Thread(target=tracking_loop, daemon=True).start()

def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/forward")
def forward():
    motors.forward()
    return jsonify({"action": "MOVING FORWARD"})

@app.route("/backward")
def backward():
    motors.backward()
    return jsonify({"action": "MOVING BACKWARD"})

@app.route("/left")
def left():
    motors.left()
    return jsonify({"action": "TURNING LEFT"})

@app.route("/right")
def right():
    motors.right()
    return jsonify({"action": "TURNING RIGHT"})

@app.route("/stop")
def stop():
    motors.stop()
    return jsonify({"action": "STOPPED"})

@app.route("/set_speed/<value>")
def set_speed(value):
    new_speed = motors.set_speed(value, auto_apply=True)
    return jsonify({"speed": new_speed})

@app.route("/get_speed")
def get_speed():
    return jsonify({"speed": motors.get_speed()})

@app.route("/get_persons")
def get_persons_route():
    return jsonify({"persons": list(get_persons())})

@app.route("/get_target")
def get_target():
    return jsonify({"target": config.TARGET_NAME})

@app.route("/set_target/<name>")
def set_target(name):
    config.TARGET_NAME = name
    return jsonify({"target": config.TARGET_NAME})

@app.route("/get_follow_mode")
def get_follow_mode():
    return jsonify({"follow_mode": follow_mode})

@app.route("/set_follow_mode/<value>")
def set_follow_mode_route(value):
    global follow_mode
    follow_mode = value.lower() == 'true'
    return jsonify({"follow_mode": follow_mode})

@app.route("/get_status")
def get_status():
    return jsonify({
        "action": current_action,
        "target": config.TARGET_NAME,
        "follow_mode": follow_mode
    })

@app.route("/register/<name>")
def register_person(name):
    import registration
    # Capture frames and register
    success = registration.register_person(name, camera)
    if success:
        return jsonify({"status": "registered", "name": name})
    return jsonify({"status": "failed", "message": "No person detected"})

def cleanup():
    motors.cleanup()

atexit.register(cleanup)

if __name__ == "__main__":
    # IMPORTANT: use_reloader=False prevents the double-start GPIO error
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
