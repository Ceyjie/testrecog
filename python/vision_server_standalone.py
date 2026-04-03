#!/usr/bin/env python3
# vision_server_standalone.py — InsightFace detection + face recognition
import cv2, json, time, logging, threading, numpy as np, queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import os
import openvino
import onnxruntime as ort

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vision")

CONFIG = {
    "persons_dir": "/home/medpal/MedPalRobotV2/data/persons",
    "camera_w": 640,
    "camera_h": 480,
    "tolerance": 0.15,
    "yolo_model": "/home/medpal/MedPalRobotV2/models/yolov8n.onnx",
    "yolo11_face_model": "/home/medpal/MedPalRobotV2/models/yolo11n-face.onnx",
    "yunet_model": "/home/medpal/MedPalRobotV2/models/face_yunet.onnx",
    # Astra Pro camera calibration
    # Focal length ~525 for 640x480 (typical for Astra)
    "focal_length": 525.0,
    "body_height_m": 1.7,
    "face_width_m": 0.14,
}

# Face detector globals  
haar_face = None
yolo_net = None
yolo_net_ov = None
yolo11_face = None
yolo11_face_ort = None  # ONNX Runtime session for face
yolo_net_ort = None     # ONNX Runtime session for body
yunet_detector = None
INSIGHTFACE_AVAILABLE = False

# OpenVINO inference engine
ov_core = None
inference_lock = threading.Lock()
inference_thread = None
inference_result = None
inference_frame = None
inference_ready = threading.Event()

# Threaded inference globals
detection_thread = None
detection_running = threading.Event()
detection_queue = queue.Queue(maxsize=2)
detection_results = queue.Queue(maxsize=2)

# Parallel detection
_face_detection_result = None
_face_detection_done = threading.Event()
_body_detection_result = None
_body_detection_done = threading.Event()

def init_openvino():
    """Initialize OpenVINO runtime"""
    global ov_core
    try:
        from openvino import Core
        ov_core = Core()
        log.info("OpenVINO runtime initialized")
        return True
    except Exception as e:
        log.warning(f"OpenVINO init failed: {e}")
        return False

def load_yolo_openvino(model_path, input_size=(320, 320)):
    """Load YOLO model with OpenVINO"""
    global ov_core
    if ov_core is None:
        if not init_openvino():
            return None
    try:
        model = ov_core.read_model(model_path)
        model.reshape({0: [1, 3, input_size[1], input_size[0]]})
        compiled = ov_core.compile_model(model, "CPU")
        log.info(f"YOLO loaded with OpenVINO: {model_path}")
        return compiled
    except Exception as e:
        log.warning(f"OpenVINO YOLO load failed: {e}")
        return None

def infer_yolo_openvino(net, frame, input_size=(320, 320)):
    """Run YOLO inference via OpenVINO"""
    if net is None:
        return None
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, (0, 0, 0), True, False)
        input_tensor = net.input(0)
        infer_request = net.create_infer_request()
        infer_request.set_input_tensor(blob)
        infer_request.infer()
        output = infer_request.get_output_tensor(0).data
        return output
    except Exception as e:
        log.debug(f"OpenVINO inference error: {e}")
        return None

def load_yolo_onnxruntime(model_path, input_size=(640, 640)):
    """Load YOLO model with ONNX Runtime"""
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        log.info(f"YOLO loaded with ONNX Runtime: {model_path}")
        return sess
    except Exception as e:
        log.warning(f"ONNX Runtime YOLO load failed: {e}")
        return None

def infer_yolo_onnxruntime(sess, frame, input_size=(640, 640)):
    """Run YOLO inference via ONNX Runtime"""
    if sess is None:
        return None
    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, (0, 0, 0), True, False)
        output = sess.run(None, {'images': blob})[0]
        return output
    except Exception as e:
        log.debug(f"ONNX Runtime inference error: {e}")
        return None

def start_detection_worker():
    """Start background thread for async detection"""
    global detection_thread, detection_running
    
    if detection_thread is not None and detection_thread.is_alive():
        return
    
    detection_running.set()
    detection_thread = threading.Thread(target=_detection_worker, daemon=True)
    detection_thread.start()
    log.info("Detection worker started")

def _detection_worker():
    """Background worker for detection"""
    global detection_running, detection_queue, detection_results
    
    while detection_running.is_set():
        try:
            frame, frame_id = detection_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            # Create a temporary detector to run detection
            # This is a simplified version - in production you'd use the actual detector
            detections = []  # Will be populated by caller
            
            # Put result back - but we need to do full detection here
            # For now, we'll handle this in the main detection path
            detection_results.put((frame_id, detections))
        except Exception as e:
            log.debug(f"Detection worker error: {e}")
            detection_queue.task_done()

def submit_frame_for_detection(frame, frame_id):
    """Submit frame for async detection"""
    global detection_queue
    
    try:
        detection_queue.put_nowait((frame, frame_id))
        return True
    except queue.Full:
        return False

def get_detection_result(timeout=0.1):
    """Get detection result from worker"""
    global detection_results
    
    try:
        return detection_results.get_nowait()
    except queue.Empty:
        return None

def init_face_detector():
    """Initialize face and body detectors"""
    global haar_face, yolo_net, yolo11_face, yolo11_face_ort, yolo_net_ort, yunet_detector, INSIGHTFACE_AVAILABLE
    
    # Try ONNX Runtime for YOLO11 face first
    if yolo11_face_ort is None:
        try:
            import os
            if os.path.exists(CONFIG["yolo11_face_model"]):
                yolo11_face_ort = load_yolo_onnxruntime(CONFIG["yolo11_face_model"], (640, 640))
                if yolo11_face_ort is not None:
                    log.info("YOLO11 face loaded with ONNX Runtime")
                else:
                    # Fallback to OpenCV DNN
                    yolo11_face = cv2.dnn.readNet(CONFIG["yolo11_face_model"])
                    yolo11_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    yolo11_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    log.info("YOLO11 face ONNX loaded (OpenCV DNN)")
        except Exception as e:
            log.warning(f"YOLO11 face failed: {e}")
    
    if yunet_detector is None:
        # Load YuNet 2023 model - better than Haar
        try:
            import os
            if os.path.exists(CONFIG["yunet_model"]):
                yunet_detector = cv2.FaceDetectorYN.create(
                    CONFIG["yunet_model"],
                    "",
                    (CONFIG["camera_w"], CONFIG["camera_h"]),
                    0.5,  # score threshold
                    nms_threshold=0.4,
                    top_k=10
                )
                log.info("YuNet 2023 face detector loaded")
            else:
                raise FileNotFoundError("Model not found")
        except Exception as e:
            log.warning(f"YuNet 2023 failed: {e}, falling back to Haar")
            yunet_detector = None
    
    if haar_face is None and yunet_detector is None:
        # Fallback to Haar cascade
        cascade_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]
        
        for path in cascade_paths:
            import os
            if os.path.exists(path):
                haar_face = cv2.CascadeClassifier(path)
                if not haar_face.empty():
                    log.info(f"Haar cascade loaded from: {path}")
                    break
    
    if yolo_net_ort is None:
        # Try ONNX Runtime first for faster inference
        yolo_net_ort = load_yolo_onnxruntime(CONFIG["yolo_model"], (320, 320))
        if yolo_net_ort is not None:
            log.info("YOLO loaded with ONNX Runtime for body detection")
        else:
            # Fallback to OpenCV DNN
            try:
                yolo_net = cv2.dnn.readNet(CONFIG["yolo_model"])
                yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                log.info("YOLO loaded for body detection (OpenCV DNN)")
            except Exception as e:
                log.warning(f"YOLO not available: {e}")

class DeepSortTracker:
    """DeepSORT-based tracker for robust person tracking"""
    
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=0.5):
        self.tracker = None
        if DEEPSORT_AVAILABLE:
            try:
                self.tracker = DeepSort(
                    max_age=max_age,
                    n_init=n_init,
                    nms_max_overlap=nms_max_overlap
                )
                log.info("DeepSORT tracker initialized")
            except Exception as e:
                log.warning(f"DeepSORT init failed: {e}")
        else:
            log.warning("DeepSORT not available, using fallback tracker")
    
    def update(self, frame, detections):
        """Update tracker with detections
        
        Args:
            frame: cv2 frame
            detections: list of dicts with keys: x, y, w, h, confidence, is_body
        
        Returns:
            list of dicts with added 'track_id' key
        """
        if self.tracker is None:
            # Fallback to simple tracking
            return self._fallback_update(detections)
        
        # Convert detections to DeepSORT format: [[x, y, w, h], confidence, class_id]
        # Keep track of which original detection index each DeepSORT detection came from
        deepsort_dets = []
        det_index_map = []  # Maps deepsort index -> original detection index
        for i, det in enumerate(detections):
            if det.get("is_body") and det.get("confidence", 0) > 0.3:
                # Only track bodies with sufficient confidence
                bbox = [det["x"], det["y"], det["w"], det["h"]]
                conf = det.get("confidence", 0.8)
                deepsort_dets.append([bbox, conf, 0])  # class_id 0 = person
                det_index_map.append(i)
        
        if not deepsort_dets:
            return detections
        
        # Update tracks
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        
        # Map detections to track IDs using a more robust matching approach
        track_id_map = {}
        
        # Get all active tracks with their bounding boxes
        active_tracks = []
        for track in tracks:
            if track.is_confirmed():
                ltrb = track.to_ltrb()  # left, top, right, bottom
                active_tracks.append({
                    'track_id': track.track_id,
                    'x': ltrb[0],
                    'y': ltrb[1],
                    'w': ltrb[2] - ltrb[0],
                    'h': ltrb[3] - ltrb[1]
                })
        
        # Match each DeepSORT detection back to its original detection
        # by comparing bounding box coordinates
        for ds_idx, orig_idx in enumerate(det_index_map):
            ds_bbox = deepsort_dets[ds_idx][0]
            ds_x, ds_y, ds_w, ds_h = ds_bbox
            
            # Find the best matching track
            best_match = None
            best_iou = 0
            for t in active_tracks:
                # Calculate IoU between detection and track
                x1 = max(ds_x, t['x'])
                y1 = max(ds_y, t['y'])
                x2 = min(ds_x + ds_w, t['x'] + t['w'])
                y2 = min(ds_y + ds_h, t['y'] + t['h'])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = ds_w * ds_h + t['w'] * t['h'] - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou and iou > 0.3:  # Threshold for valid match
                        best_iou = iou
                        best_match = t['track_id']
            
            if best_match is not None:
                track_id_map[orig_idx] = best_match
        
        # Add track IDs to detections
        for i, det in enumerate(detections):
            if i in track_id_map:
                det["track_id"] = track_id_map[i]
        
        return detections
    
    def _fallback_update(self, detections):
        """Simple fallback tracking based on position"""
        # This is already handled by PersonTracker
        return detections


class PersonTracker:
    def __init__(self):
        self.tracked_persons = {}
        self.next_id = 0
        self.max_distance = 100  # Stricter distance matching
    
    def update(self, detections):
        # Don't filter by confidence here - let all through for tracking
        matched_ids = set()
        
        for det in detections:
            cx, cy = det["center_x"], det["center_y"]
            is_face = det.get("is_face", False)
            is_body = det.get("is_body", False)
            name = det.get("name")  # May be set from face detection or body association
            confidence = det.get("confidence", 0.8)
            
            matched_id = None
            min_dist = float('inf')
            
            for pid, person in self.tracked_persons.items():
                prev_cx, prev_cy = person["last_x"], person["last_y"]
                dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                
                # Allow larger distance for known persons (have a name)
                max_dist = self.max_distance
                if person.get("name"):
                    max_dist = 300  # Allow up to 300px for tracked persons
                
                if dist < max_dist and dist < min_dist:
                    min_dist = dist
                    matched_id = pid
            
            if matched_id is not None:
                old_name = self.tracked_persons[matched_id].get("name")
                
                # If we have a name (recognized person), always update
                if name:
                    self.tracked_persons[matched_id]["name"] = name
                    self.tracked_persons[matched_id]["is_face"] = True
                elif old_name:
                    # Keep the old name if we lost face but have body detection
                    # This is the key feature: track body with known name
                    self.tracked_persons[matched_id]["name"] = old_name
                
                # Update based on detection type
                if is_face:
                    self.tracked_persons[matched_id]["is_face"] = True
                if is_body:
                    self.tracked_persons[matched_id]["is_body"] = True
                
                self.tracked_persons[matched_id].update({
                    "last_x": cx, "last_y": cy,
                    "last_seen": time.time(),
                    "face_size": det.get("face_size", 80),
                    "body_size": det.get("body_size", 200),
                    "body_x": det.get("x", cx),
                    "body_y": det.get("y", cy),
                    "body_w": det.get("w", int(det.get("body_size", 200) * 0.6)),
                    "body_h": det.get("h", det.get("body_size", 200)),
                    "confidence": confidence
                })
                matched_ids.add(matched_id)
            else:
                # New detection - track if it's a recognized person, has face, or has body
                if name or is_face or is_body:
                    new_id = self.next_id
                    self.next_id += 1
                    self.tracked_persons[new_id] = {
                        "id": new_id,
                        "name": name,
                        "last_x": cx, "last_y": cy,
                        "last_seen": time.time(),
                        "face_size": det.get("face_size", 80),
                        "body_size": det.get("body_size", 200),
                        "body_x": det.get("x", cx),
                        "body_y": det.get("y", cy),
                        "body_w": det.get("w", int(det.get("body_size", 200) * 0.6)),
                        "body_h": det.get("h", det.get("body_size", 200)),
                        "is_face": is_face,
                        "is_body": is_body,
                        "confidence": confidence
                    }
                    matched_ids.add(new_id)
        
        # Remove old tracked persons not seen recently (increase timeout for body tracking)
        for pid in list(self.tracked_persons.keys()):
            # Keep tracked longer if we have a recognized name (even after turning away)
            person = self.tracked_persons[pid]
            if person.get("name"):
                # Trusted person - keep tracking for up to 5 seconds without detection
                max_age = 5.0
            else:
                max_age = 2.0
            
            if time.time() - person["last_seen"] > max_age:
                del self.tracked_persons[pid]
        
        return self.tracked_persons

class FaceRecognitionSystem:
    def __init__(self):
        self.registered = {}
        self.label_to_name = {}
        self.tracker = PersonTracker()
        self.yolo11_face_ort = None  # Keep local reference
        self.yunet_detector = None
        self.deepsort_tracker = DeepSortTracker(max_age=30, n_init=3)
        self.person_tracker = {}  # Track persons with names persistently
        self._load_registered()
        self._init_detectors()
    
    def _load_registered(self):
        registry_path = os.path.join(CONFIG["persons_dir"], "registry.json")
        
        if not os.path.exists(registry_path):
            log.warning("No registry found")
            return
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        faces, labels = [], []
        self.label_to_name = {}
        next_label = 0
        
        for name, data in registry.items():
            self.label_to_name[next_label] = name
            for img_path in data.get("images", []):
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(next_label)
                        
                        # Add horizontal flip for augmentation
                        flipped = cv2.flip(img, 1)
                        faces.append(flipped)
                        labels.append(next_label)
            next_label += 1
            self.registered[name] = True
        
        if faces:
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=50
            )
            self.lbph_recognizer.train(faces, np.array(labels))
            log.info(f"LBPH trained with {len(faces)} images")
        
        log.info(f"Registered: {list(self.registered.keys())}")
    
    def _init_detectors(self):
        """Initialize face and body detectors"""
        import os
        global haar_face, yolo_net, yolo11_face, yunet_detector, yolo11_face_ort, yolo_net_ort
        
        if self.yolo11_face_ort is None:
            try:
                if os.path.exists(CONFIG["yolo11_face_model"]):
                    self.yolo11_face_ort = load_yolo_onnxruntime(CONFIG["yolo11_face_model"], (640, 640))
                    if self.yolo11_face_ort is not None:
                        log.info("YOLO11 face loaded with ONNX Runtime")
            except Exception as e:
                log.warning(f"YOLO11 face failed: {e}")
        
        if self.yunet_detector is None:
            try:
                if os.path.exists(CONFIG["yunet_model"]):
                    self.yunet_detector = cv2.FaceDetectorYN.create(
                        CONFIG["yunet_model"],
                        "",
                        (CONFIG["camera_w"], CONFIG["camera_h"]),
                        0.5,
                        nms_threshold=0.4,
                        top_k=10
                    )
                    log.info("YuNet 2023 face detector loaded")
            except Exception as e:
                log.warning(f"YuNet failed: {e}")
        
        if haar_face is None:
            cascade_paths = [
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
            ]
            for path in cascade_paths:
                if os.path.exists(path):
                    haar_face = cv2.CascadeClassifier(path)
                    if not haar_face.empty():
                        log.info(f"Haar cascade loaded from: {path}")
                        break
    
    def detect_faces(self, frame):
        """Detect faces using YOLO11 (primary), YuNet, then Haar fallback"""
        global haar_face, yolo_net, yolo11_face
        
        # Initialize on first use
        if self.yolo11_face_ort is None and self.yunet_detector is None and haar_face is None:
            self._init_detectors()
        
        detections = []
        
        # Try YOLO11 ONNX with ONNX Runtime first
        if self.yolo11_face_ort is not None:
            try:
                h, w = frame.shape[:2]
                
                # ONNX Runtime inference
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
                output = self.yolo11_face_ort.run(None, {'images': blob})[0]
                outputs_data = [output]
                
                boxes = []
                for out in outputs_data:
                    predictions = out.reshape(-1, 5)
                    for pred in predictions:
                        conf = 1 / (1 + np.exp(-pred[4]))
                        if conf < 0.5:
                            continue
                        cx, cy, bw, bh = pred[:4]
                        x = int((cx - bw/2) * w / 640)
                        y = int((cy - bh/2) * h / 640)
                        fw = int(bw * w / 640)
                        fh = int(bh * h / 640)
                        if fw < 40 or fh < 40:
                            continue
                        boxes.append([int(x), int(y), int(x+fw), int(y+fh), float(conf)])
                
                if boxes:
                    boxes = np.array(boxes, dtype=np.float32)
                    indices = cv2.dnn.NMSBoxes(
                        boxes[:, :4].tolist(), 
                        boxes[:, 4].tolist(),
                        0.5, 0.4
                    )
                    
                    # Select largest box first
                    best_idx = None
                    best_area = 0
                    for idx in indices:
                        x1, y1, x2, y2, _ = boxes[idx]
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area = area
                            best_idx = idx
                    
                    if best_idx is not None:
                        x1, y1, x2, y2, conf = boxes[best_idx]
                        
                        # Clip to frame bounds
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        
                        fw, fh = x2 - x1, y2 - y1
                        
                        face_roi = frame[y1:y2, x1:x2]
                        
                        name = None
                        if self.registered:
                            name = self._recognize_face(face_roi, threshold=50)
                        
                        detections.append({
                            "x": int(x1), "y": int(y1), "w": int(fw), "h": int(fh),
                            "center_x": int(x1 + fw // 2),
                            "center_y": int(y1 + fh // 2),
                            "face_size": int(max(fw, fh)),
                            "body_size": int(fh * 2.5),
                            "name": name,
                            "is_face": True,
                            "is_body": False,
                            "confidence": float(conf)
                        })
            except Exception as e:
                log.debug(f"YOLO11 error: {e}")
            try:
                h, w = frame.shape[:2]
                self.yunet_detector.setInputSize((w, h))
                result = self.yunet_detector.detect(frame)
                
                if result[1] is not None:
                    for face in result[1]:
                        x, y, w, h = map(int, face[:4])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        score = float(face[-1])
                        
                        if score < 0.5:
                            continue
                        
                        fw, fh = w, h
                        
                        if fw < 40 or fh < 40:
                            continue
                        
                        face_roi = frame[y:y+h, x:x+w]
                        
                        name = None
                        if self.registered:
                            name = self._recognize_face(face_roi, threshold=50)
                        
                        detections.append({
                            "x": x, "y": y, "w": fw, "h": fh,
                            "center_x": x + fw // 2,
                            "center_y": y + fh // 2,
                            "face_size": max(fw, fh),
                            "body_size": int(fh * 2.5),
                            "name": name,
                            "is_face": True,
                            "is_body": False,
                            "confidence": score
                        })
            except Exception as e:
                log.debug(f"YuNet error: {e}")
        
        # Fallback to Haar if no detections
        if len(detections) == 0 and haar_face is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                faces = haar_face.detectMultiScale(
                    gray, 
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(45, 45),
                    maxSize=(400, 400)
                )
                
                for (x, y, fw, fh) in faces:
                    if fw < 45 or fh < 45:
                        continue
                    
                    ratio = fw / fh
                    if ratio < 0.5 or ratio > 1.8:
                        continue
                    
                    face_roi = frame[y:y+fh, x:x+fw]
                    
                    name = None
                    if self.registered:
                        name = self._recognize_face(face_roi, threshold=65)
                    
                    detections.append({
                        "x": x, "y": y, "w": fw, "h": fh,
                        "center_x": x + fw // 2,
                        "center_y": y + fh // 2,
                        "face_size": max(fw, fh),
                        "body_size": int(fh * 2.5),
                        "name": name,
                        "is_face": True,
                        "is_body": False,
                        "confidence": 0.8
                    })
            except Exception as e:
                log.debug(f"Haar error: {e}")
        
        return detections
    
    def _recognize_face_embedding(self, embedding):
        """Recognize face using LBPH (simpler approach)"""
        # For now, return None - InsightFace provides embeddings but we'd need
        # to compare with registered embeddings using a similarity metric
        # The face is detected but recognition needs more setup
        return None
    
    def _recognize_face(self, face_roi, threshold=65):
        """Recognize face using LBPH with validation"""
        if self.lbph_recognizer is None or face_roi is None or face_roi.size == 0:
            return None
        
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            
            gray = cv2.equalizeHist(gray)
            gray = cv2.resize(gray, (200, 200))
            
            label, conf = self.lbph_recognizer.predict(gray)
            
            if conf < threshold:
                name = self.label_to_name.get(label, None)
                if name:
                    log.debug(f"Recognized: {name} (conf: {conf:.1f})")
                    return name
            else:
                log.debug(f"Not recognized (conf: {conf:.1f})")
                
        except Exception as e:
            log.debug(f"Recognition error: {e}")
        
        return None
    
    def _recognize_face_embedding(self, embedding):
        return None
    
    def _validate_face_quality(self, gray_face):
        """Validate that the face region has sufficient quality"""
        if gray_face is None or gray_face.size == 0:
            return False
        
        # Check brightness (not too dark or too bright)
        mean_brightness = np.mean(gray_face)
        if mean_brightness < 50 or mean_brightness > 210:
            return False
        
        # Check contrast
        std_dev = np.std(gray_face)
        if std_dev < 35:
            return False
        
        # Check that face region is not mostly uniform (should have features)
        unique_values = len(np.unique(gray_face))
        if unique_values < 30:
            return False
        
        # Additional check: verify face has adequate edge density
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.05:
            return False
        
        return True
    
    def detect_and_recognize(self, frame):
        """Body detection + face recognition with tracking persistence"""
        global yolo_net_ort
        
        if not hasattr(self, 'body_net_ort') or self.body_net_ort is None:
            try:
                self.body_net_ort = ort.InferenceSession(
                    CONFIG["yolo_model"], 
                    providers=['CPUExecutionProvider']
                )
                log.info("Body YOLO loaded")
            except Exception as e:
                log.warning(f"Body YOLO load failed: {e}")
                self.body_net_ort = None
        
        # Initialize person tracker if needed
        if not hasattr(self, 'person_tracker'):
            self.person_tracker = {}  # {person_id: {"name": "Carl", "last_body": {...}}}
        
        h, w = frame.shape[:2]
        tracked = {}
        
        # Body detection (like C++ code)
        if self.body_net_ort is not None:
            try:
                CONF_BODY = 0.4
                NMS_THRESH = 0.45
                
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), (0, 0, 0), True, False)
                output = self.body_net_ort.run(None, {'images': blob})[0]
                
                out = output.reshape(1, output.shape[1], output.shape[2])
                out = out.reshape(out.shape[1], out.shape[2])
                out = out.transpose(1, 0)
                
                boxes = []
                scores = []
                scaleX = w / 320
                scaleY = h / 320
                
                for i in range(out.shape[0]):
                    row = out[i]
                    conf = row[4]
                    if conf < CONF_BODY:
                        continue
                    cx = row[0] * scaleX
                    cy = row[1] * scaleY
                    bw = row[2] * scaleX
                    bh = row[3] * scaleY
                    boxes.append([int(cx-bw/2), int(cy-bh/2), int(bw), int(bh)])
                    scores.append(float(conf))
                
                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_BODY, NMS_THRESH)
                    
                    # Match with tracked persons by IoU
                    for idx in indices:
                        x1, y1, bw, bh = boxes[idx]
                        cx = x1 + bw//2
                        cy = y1 + bh//2
                        
                        # Find best matching tracked person
                        best_pid = None
                        best_iou = 0
                        
                        for pid, pdata in self.person_tracker.items():
                            last = pdata.get("last_body", {})
                            if last:
                                lx1 = last.get("x1", 0)
                                ly1 = last.get("y1", 0)
                                lx2 = lx1 + last.get("w", 0)
                                ly2 = ly1 + last.get("h", 0)
                                
                                # Calculate IoU
                                inter_x1 = max(x1, lx1)
                                inter_y1 = max(y1, ly1)
                                inter_x2 = min(x1+bw, lx2)
                                inter_y2 = min(y1+bh, ly2)
                                
                                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                    inter = (inter_x2-inter_x1) * (inter_y2-inter_y1)
                                    union = bw*bh + last.get("w",0)*last.get("h",0) - inter
                                    iou = inter / (union + 1e-6)
                                    if iou > best_iou and iou > 0.3:
                                        best_iou = iou
                                        best_pid = pid
                        
                        if best_pid is not None:
                            pid = best_pid
                            # Update existing person
                            self.person_tracker[pid]["last_body"] = {"x1": x1, "y1": y1, "w": bw, "h": bh, "cx": cx, "cy": cy}
                            self.person_tracker[pid]["last_seen"] = time.time()
                        else:
                            # New person - find if any known person was lost
                            pid = len(self.person_tracker)
                            self.person_tracker[pid] = {
                                "name": None,
                                "last_body": {"x1": x1, "y1": y1, "w": bw, "h": bh, "cx": cx, "cy": cy},
                                "last_seen": time.time()
                            }
                        
                        tracked[pid] = {
                            "last_x": cx,
                            "last_y": cy,
                            "body_size": bh,
                            "body_x": x1,
                            "body_y": y1,
                            "body_w": bw,
                            "body_h": bh,
                            "is_body": True,
                            "name": self.person_tracker[pid].get("name"),
                            "face_x": None,
                            "face_y": None,
                            "face_w": None
                        }
            except Exception as e:
                log.debug(f"Body detection error: {e}")
        
        # Face detection + recognition (single sample per frame)
        face_dets = self.detect_faces(frame)
        
        # Associate faces with bodies
        for face in face_dets:
            fx = face.get("center_x", face["x"] + face["w"]//2)
            fy = face.get("center_y", face["y"] + face["h"]//2)
            
            # Find closest body
            best_pid = None
            best_dist = float('inf')
            
            for pid, person in tracked.items():
                bx = person["last_x"]
                by = person["last_y"]
                dist = ((fx - bx)**2 + (fy - by)**2)**0.5
                if dist < best_dist and dist < 200:
                    best_dist = dist
                    best_pid = pid
            
            if best_pid is not None:
                tracked[best_pid]["name"] = None  # No recognition - just detect face
                tracked[best_pid]["face_x"] = face["x"]
                tracked[best_pid]["face_y"] = face["y"]
                tracked[best_pid]["face_w"] = face["w"]
        
        # Clean up old tracked persons (not seen for 5 seconds)
        now = time.time()
        to_remove = []
        for pid in list(self.person_tracker.keys()):
            if now - self.person_tracker[pid].get("last_seen", 0) > 5.0:
                to_remove.append(pid)
        for pid in to_remove:
            if pid in self.person_tracker:
                del self.person_tracker[pid]
        
        return [], tracked
    
    def _detect_bodies_yolo(self, frame):
        """Detect bodies using YOLO - person class"""
        global yolo_net, yolo_net_ort
        
        # Initialize on first use
        if yolo_net_ort is None and yolo_net is None:
            init_face_detector()
        
        if yolo_net_ort is None and yolo_net is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Try ONNX Runtime first
        if yolo_net_ort is not None:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), (0, 0, 0), True, False)
                output = yolo_net_ort.run(None, {'images': blob})[0]
                outs = [output]
            except Exception as e:
                log.debug(f"ONNX Runtime inference error: {e}")
                outs = []
        elif yolo_net is not None:
            # Fallback to OpenCV DNN
            try:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), (0, 0, 0), True, False)
                yolo_net.setInput(blob)
                outs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())
            except Exception as e:
                log.debug(f"OpenCV DNN inference error: {e}")
                outs = []
        else:
            return []
        
        if not outs:
            return []
        
        bodies = []
        
        for output in outs:
            predictions = output.reshape(-1, 84)
            
            candidates = []
            for pred in predictions:
                box = pred[0:4]
                class_scores = 1 / (1 + np.exp(-pred[4:]))  # sigmoid
                
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                # Class 0 = person in COCO, use threshold like C++ code
                if class_id == 0 and confidence > 0.4:
                    candidates.append({
                        'cx': box[0], 'cy': box[1], 
                        'w': box[2], 'h': box[3],
                        'conf': confidence
                    })
            
            # Apply NMS using OpenCV (like C++ code)
            if candidates:
                boxes = []
                scores = []
                for c in candidates:
                    cx, cy, bw, bh = c['cx'], c['cy'], c['w'], c['h']
                    x1 = cx - bw/2
                    y1 = cy - bh/2
                    boxes.append([int(x1), int(y1), int(bw), int(bh)])
                    scores.append(c['conf'])
                
                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
                    for idx in indices:
                        c = candidates[idx]
                        cx, cy, bw, bh = c['cx'], c['cy'], c['w'], c['h']
                        confidence = c['conf']
                        
                        # YOLOv8n uses 320x320 input, scale to frame
                        scale_x = w / 320
                        scale_y = h / 320
                        cx_px = int(cx * scale_x)
                        cy_px = int(cy * scale_y)
                        bw_px = int(bw * scale_x)
                        bh_px = int(bh * scale_y)
                        
                        x = max(0, cx_px - bw_px // 2)
                        y = max(0, cy_px - bh_px // 2)
                        
                        if bw_px > 40 and bh_px > 60:
                            bodies.append({
                                "x": x, "y": y,
                                "w": bw_px, "h": bh_px,
                                "center_x": cx_px, "center_y": cy_px,
                                "body_size": bh_px,
                                "confidence": float(confidence),
                                "is_body": True,
                                "name": None
                            })
        
        return bodies
    
    def estimate_distance(self, size_px, is_body=False):
        """Estimate distance using pinhole camera model"""
        if is_body and size_px > 50:
            # Distance = (real_height * focal_length) / apparent_height
            dist = (CONFIG["body_height_m"] * CONFIG["focal_length"]) / size_px
            return max(0.5, min(8.0, dist))
        elif size_px > 30:
            dist = (CONFIG["face_width_m"] * CONFIG["focal_length"]) / size_px
            return max(0.3, min(4.0, dist))
        return None
    
    def retrain(self):
        self.registered = {}
        self._load_registered()

class VisionServer:
    def __init__(self):
        self.running = False
        self.following = False
        self.target_name = "anyone"
        self.latest_frame = None
        self.latest_status = {
            "persons": 0, 
            "distance": "?", 
            "action": "IDLE", 
            "target_name": "",
            "x_deviation": 0,
            "y_deviation": 0
        }
        self.lock = threading.Lock()
        
        # Tracking state
        self.x_deviation = 0
        self.y_deviation = 0
        self.target_person = None
        self.current_action = "IDLE"
        
        self.face_system = FaceRecognitionSystem()
        self.cap = None
    
    def process_loop(self):
        # Try Orbbec Astra first (device 0 for RGB), then fallback to other cameras
        camera_devices = [0, 1, 2]  # Astra typically on 0 or 1
        
        for device in camera_devices:
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if self.cap.isOpened():
                log.info(f"Camera opened on device {device}")
                break
        
        if self.cap is None or not self.cap.isOpened():
            log.error("Cannot open any camera")
            return
        
        # Set camera properties for Astra (640x480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_w"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_h"])
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Try to set format for Astra
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        except:
            pass
        
        # Get actual camera properties
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened: {actual_w}x{actual_h}")
        
        tolerance = CONFIG["tolerance"]
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            h, w = frame.shape[:2]
            frame_center_x = w / 2
            frame_center_y = h / 2
            
            detections, tracked = self.face_system.detect_and_recognize(frame)
            
            closest_distance = None
            closest_person = None
            target_found = False
            
            # Draw center lines (like reference code)
            cv2.line(frame, (0, int(h/2)), (w, int(h/2)), (255, 0, 0), 1)
            cv2.line(frame, (int(w/2), 0), (int(w/2), h), (255, 0, 0), 1)
            
            # Draw tolerance box
            tol_x1 = int(w/2 - tolerance * w)
            tol_y1 = int(h/2 - tolerance * h)
            tol_x2 = int(w/2 + tolerance * w)
            tol_y2 = int(h/2 + tolerance * h)
            cv2.rectangle(frame, (tol_x1, tol_y1), (tol_x2, tol_y2), (0, 255, 0), 2)
            
            # Find target person - follow the closest person to center
            target_person = None
            target_person_dist = 2.0
            target_center = None
            target_box = None
            min_dist = float('inf')
            
            for person_id, person in tracked.items():
                body_x = person.get("body_x", person.get("last_x", 0))
                body_y = person.get("body_y", person.get("last_y", 0))
                body_w = person.get("body_w", 100)
                body_h = person.get("body_h", 150)
                body_size = person.get("body_size", 150)
                
                x1 = max(0, body_x)
                y1 = max(0, body_y)
                x2 = min(w, body_x + body_w)
                y2 = min(h, body_y + body_h)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                dist_from_center = ((cx - w/2)**2 + (cy - h/2)**2)**0.5
                
                # Use proper distance estimation
                dist_m = self.face_system.estimate_distance(body_size, is_body=True)
                if dist_m is None:
                    dist_m = 2.0
                
                # Color and label based on recognition
                name = person.get("name")
                if name:
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{name} {dist_m:.1f}m"
                else:
                    color = (255, 165, 0)  # Orange for unknown
                    label = f"P{person_id+1} {dist_m:.1f}m"
                
                # Draw face box if we have face location
                if person.get("face_w"):
                    fx = person.get("face_x", 0)
                    fy = person.get("face_y", 0)
                    fw = person.get("face_w", 0)
                    fh = int(fw * 1.2)
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if dist_from_center < min_dist:
                    min_dist = dist_from_center
                    target_person = person
                    target_person_dist = dist_m
                    target_found = True
                    target_center = (cx, cy)
                    target_box = (x1, y1, x2, y2)
            
            if target_found and target_center:
                cx, cy = target_center
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.line(frame, (cx-12, cy), (cx+12, cy), (255, 255, 255), 1)
                cv2.line(frame, (cx, cy-12), (cx, cy+12), (255, 255, 255), 1)
                
                if target_box:
                    tx1, ty1, tx2, ty2 = target_box
                    self.x_deviation = round(0.5 - (tx1+tx2)/2/w, 3)
                    self.y_deviation = round(0.5 - (ty1+ty2)/2/h, 3)
            
            action = "IDLE"
            
            # Only follow if target is a REGISTERED person (name is not None)
            can_follow = self.following and target_found and target_person and target_person.get("name")
            
            if can_follow:
                dist_m = target_person_dist
                
                # Distance-based control
                if abs(self.x_deviation) < tolerance and abs(self.y_deviation) < tolerance:
                    if dist_m < 0.3:
                        action = "TOO_CLOSE"
                    elif dist_m > 1.5:
                        action = "MOVE_FORWARD"
                    else:
                        action = "CENTERED"
                elif abs(self.x_deviation) > abs(self.y_deviation):
                    action = "TURN_LEFT" if self.x_deviation >= tolerance else "TURN_RIGHT"
                else:
                    if dist_m < 0.3:
                        action = "MOVE_BACK"
                    elif dist_m > 1.5:
                        action = "MOVE_FORWARD"
                    else:
                        action = "CENTERED"
            
            # Store action for motor control
            self.current_action = action
            
            closest_distance = target_person_dist
            target_name = target_person.get("name") if target_person else ""
            
            # Draw deviation text
            cv2.putText(frame, f"X: {self.x_deviation:.2f}", (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if abs(self.x_deviation) < tolerance else (0, 0, 255), 2)
            cv2.putText(frame, f"Y: {self.y_deviation:.2f}", (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if abs(self.y_deviation) < tolerance else (0, 0, 255), 2)
            cv2.putText(frame, action, (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update status
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_status["persons"] = len(tracked)
                self.latest_status["target_name"] = target_name or ""
                self.latest_status["distance"] = f"{closest_distance:.1f}m" if closest_distance else "?"
                self.latest_status["action"] = action
                self.latest_status["x_deviation"] = self.x_deviation
                self.latest_status["y_deviation"] = self.y_deviation
            
            time.sleep(0.03)
        
        self.cap.release()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.thread.start()
        log.info("Vision started")
    
    def stop(self):
        self.running = False
        log.info("Vision stopped")
    
    def set_following(self, val):
        with self.lock:
            self.following = val
    
    def get_following(self):
        with self.lock:
            return self.following
    
    def set_target(self, name):
        with self.lock:
            self.target_name = name
    
    def get_status(self):
        with self.lock:
            return self.latest_status.copy()
    
    def get_tracking_info(self):
        """Get deviation values for motor control (like reference code)"""
        with self.lock:
            return {
                "x_deviation": self.x_deviation,
                "y_deviation": self.y_deviation,
                "following": self.following,
                "target_name": self.target_name,
                "action": self.current_action
            }
    
    def get_frame_jpeg(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ret, jpg = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            return jpg.tobytes() if ret else None

class StreamingHandler(BaseHTTPRequestHandler):
    server_instance = None
    
    def do_GET(self):
        if '/frame.jpg' in self.path or self.path == '/':
            frame = self.server_instance.get_frame_jpeg()
            if frame:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(404)
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

vision = None

def start_vision():
    global vision
    vision = VisionServer()
    StreamingHandler.server_instance = vision
    vision.start()
    
    def _start_http():
        try:
            server = ThreadedHTTPServer(('0.0.0.0', 8080), StreamingHandler)
            server.allow_reuse_address = True
            log.info("HTTP stream: http://0.0.0.0:8080/frame.jpg")
            server.serve_forever()
        except OSError as e:
            if e.errno == 98:  # Address already in use
                log.warning(f"Port 8080 in use, trying 8081...")
                try:
                    server = ThreadedHTTPServer(('0.0.0.0', 8081), StreamingHandler)
                    server.allow_reuse_address = True
                    log.info("HTTP stream: http://0.0.0.0:8081/frame.jpg")
                    server.serve_forever()
                except Exception as e2:
                    log.error(f"HTTP server error on 8081: {e2}")
            else:
                log.error(f"HTTP server error: {e}")
        except Exception as e:
            log.error(f"HTTP server error: {e}")
    
    threading.Thread(target=_start_http, daemon=True).start()
    return vision

if __name__ == "__main__":
    v = start_vision()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        v.stop()
