import cv2
import time
import motor_control as motors
import occupancy_map
import tracker
import body_reid
import face_reid
import pose_reid
import registration

# ── Config ───────────────────────────────────────────────────
import config
TARGET_NAME = config.TARGET_NAME  # synced with config.py (currently: Carl)
FRAME_CENTER_X   = config.FRAME_CENTER_X  # 208 for 416px, synced with config.py
DEVIATION_THRESH = 60          # pixels off-center before turning
GRID_COLS        = config.GRID_COLS
OBSTACLE_COL     = config.OBSTACLE_COL

# ── State ────────────────────────────────────────────────────
target_track_id = None         # ByteTrack ID of the target person

# ── Setup ────────────────────────────────────────────────────
registration.load_all()        # restore registered persons from disk

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# For Orbbec depth — replace with your actual camera.py integration
# depth_pipe = your_orbbec_pipeline()

print("MedPal running. Press Ctrl+C to stop.")

try:
    while True:
        ret, color_frame = cap.read()
        if not ret:
            continue

        # ── 1. Get depth and build occupancy grid ─────────────
        # depth_array = depth_pipe.get_depth_numpy()
        # grid = occupancy_map.depth_to_grid(depth_array)
        # Placeholder until Orbbec integrated:
        grid = ['FREE'] * GRID_COLS

        # ── 2. Track all persons in frame ─────────────────────
        persons = tracker.get_tracked_persons(color_frame)

        # ── 3. ReID cascade — find or confirm target ──────────
        for person in persons:
            if person['id'] == target_track_id:
                continue  # already identified this track

            # Try face first (most accurate)
            name = face_reid.recognize(color_frame)

            # Fall back to body ReID
            if not name:
                name = body_reid.recognize(color_frame, person['bbox'])

            # Fall back to pose signature
            if not name:
                name = pose_reid.recognize(color_frame)

            if name == TARGET_NAME:
                target_track_id = person['id']
                print(f"Target locked: {name} (track ID {target_track_id})")

        # ── 4. Drive toward target ────────────────────────────
        target = next(
            (p for p in persons if p['id'] == target_track_id),
            None
        )

        if target:
            x1, y1, x2, y2 = target['bbox']
            cx = (x1 + x2) // 2
            deviation = cx - FRAME_CENTER_X

            if grid[OBSTACLE_COL] == 'BLOCKED':
                motors.stop()
                print("Obstacle — stopped")
            elif deviation < -DEVIATION_THRESH:
                motors.left()
            elif deviation > DEVIATION_THRESH:
                motors.right()
            else:
                motors.forward()
        else:
            # Lost target — reset track ID so ReID runs again
            if target_track_id is not None:
                print("Target lost — searching...")
                target_track_id = None
            motors.stop()

        time.sleep(0.05)  # ~20 Hz control loop

except KeyboardInterrupt:
    print("Stopping...")
    motors.cleanup()
    cap.release()
