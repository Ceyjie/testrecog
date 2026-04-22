# face_recognizer.py — Better face recognition with ensemble
import cv2, os, json, numpy as np, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face_recognizer")

PERSONS_DIR = "/home/medpal/Desktop/cpp_robot/rebot/data/persons"

class FaceRecognizer:
    def __init__(self):
        self.lbph = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=12, grid_x=10, grid_y=10, threshold=100
        )
        self.eigen = cv2.face.EigenFaceRecognizer_create(threshold=1000)
        self.fisher = cv2.face.FisherFaceRecognizer_create()
        
        self.label_to_name = {}
        self.next_label = 0
        self.trained = False
        self._train()
    
    def _preprocess(self, img):
        gray = img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (200, 200))
        
        return gray
    
    def _train(self):
        faces, labels = [], []
        self.label_to_name = {}
        self.next_label = 0
        
        registry_path = os.path.join(PERSONS_DIR, "registry.json")
        
        if not os.path.exists(registry_path):
            log.warning("No registry found")
            return
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        for name, data in registry.items():
            label = self.next_label
            self.label_to_name[label] = name
            self.next_label += 1
            
            for img_path in data.get("images", []):
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        faces.append(self._preprocess(img))
                        labels.append(label)
                        
                        faces.append(self._flip_preprocess(img))
                        labels.append(label)
        
        if len(faces) < 3:
            log.warning("Not enough training images")
            return
        
        self.lbph.train(faces, np.array(labels))
        self.eigen.train(faces, np.array(labels))
        
        try:
            self.fisher.train(faces, np.array(labels))
        except:
            pass
        
        self.trained = True
        log.info(f"Trained with {len(faces)} samples, {len(self.label_to_name)} persons")
        for l, n in self.label_to_name.items():
            log.info(f"  Label {l}: {n}")
    
    def _flip_preprocess(self, img):
        gray = img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        flipped = cv2.flip(gray, 1)
        flipped = cv2.equalizeHist(flipped)
        flipped = cv2.resize(flipped, (200, 200))
        
        return flipped
    
    def recognize(self, face_roi):
        if not self.trained:
            return None
        
        try:
            processed = self._preprocess(face_roi)
            
            # Print confidence to terminal for debugging
            label_lbph, conf_lbph = self.lbph.predict(processed)
            label_eigen, conf_eigen = self.eigen.predict(processed)
            
            print(f"DEBUG: LBPH conf: {conf_lbph}, Eigen conf: {conf_eigen}")
            
            # Loosened threshold for testing (LBPH < 100 is generally acceptable)
            if conf_lbph < 100: 
                name = self.label_to_name.get(label_lbph, "unknown")
                print(f"Recognized: {name}")
                return name
            
        except Exception as e:
            log.warning(f"Recognition error: {e}")
        
        print("Recognized: unknown")
        return "unknown"
    
    def retrain(self):
        self.trained = False
        self._train()

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer
