# yolo_face_recognizer.py — YOLOv8 face detection + recognition
import cv2, os, json, numpy as np, logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yolo_face")

PERSONS_DIR = "/home/medpal/MedPalRobotV2/data/persons"

class YOLOFaceRecognizer:
    def __init__(self):
        self.face_model = YOLO("yolov8n.pt")
        self.haar = cv2.CascadeClassifier(
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        )
        
        self.registered_embeddings = {}
        self._load_registered()
    
    def _load_registered(self):
        registry_path = os.path.join(PERSONS_DIR, "registry.json")
        
        if not os.path.exists(registry_path):
            log.warning("No registry found")
            return
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        for name, data in registry.items():
            embeddings = []
            for img_path in data.get("images", []):
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        emb = self._get_embedding_haar(img)
                        if emb is not None:
                            embeddings.append(emb)
            
            if embeddings:
                self.registered_embeddings[name] = np.mean(embeddings, axis=0)
                log.info(f"Loaded {len(embeddings)} samples for {name}")
        
        log.info(f"Total registered: {len(self.registered_embeddings)} persons")
    
    def _get_embedding_haar(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)
            return gray.flatten().astype(np.float32) / 255.0
        except:
            return None
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def detect_faces_yolo(self, frame):
        results = self.face_model(frame, classes=[0], conf=0.5, verbose=False)
        
        faces = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        return faces
    
    def detect_faces_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detection
        all_faces = []
        
        # Original size
        faces1 = self.haar.detectMultiScale(gray, 1.1, 5, 0, (40, 40), (200, 200))
        all_faces.extend(faces1.tolist() if len(faces1) > 0 else [])
        
        # Half size for smaller faces
        h, w = gray.shape
        small = cv2.resize(gray, (w//2, h//2))
        faces2 = self.haar.detectMultiScale(small, 1.1, 5, 0, (20, 20), (100, 100))
        for (x, y, fw, fh) in faces2:
            all_faces.append([x*2, y*2, fw*2, fh*2])
        
        return np.array(all_faces) if all_faces else np.array([])
    
    def recognize(self, face_roi):
        if not self.registered_embeddings:
            return None
        
        embedding = self._get_embedding_haar(face_roi)
        if embedding is None:
            return None
        
        best_name = None
        best_score = 0
        
        for name, known_emb in self.registered_embeddings.items():
            score = self._cosine_similarity(embedding, known_emb)
            if score > best_score and score > 0.6:
                best_score = score
                best_name = name
        
        if best_name:
            log.info(f"Recognized: {best_name} ({best_score:.2f})")
        
        return best_name
    
    def detect_and_recognize(self, frame):
        faces = self.detect_faces_haar(frame)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            name = self.recognize(face_roi)
            results.append({
                "box": (x, y, w, h),
                "name": name
            })
        
        return results
    
    def retrain(self):
        self.registered_embeddings = {}
        self._load_registered()

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = YOLOFaceRecognizer()
    return _recognizer
