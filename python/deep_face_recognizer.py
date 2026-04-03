# deep_face_recognizer.py — PyTorch-based face recognition
import cv2, os, json, numpy as np, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deep_face")

PERSONS_DIR = "/home/medpal/MedPalRobotV2/data/persons"

class FaceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(576, 128)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class DeepFaceRecognizer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = FaceEmbedder().to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.registered_embeddings = {}
        self._load_registered_faces()
    
    def _load_registered_faces(self):
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
                        emb = self._get_embedding(img)
                        if emb is not None:
                            embeddings.append(emb)
            
            if embeddings:
                self.registered_embeddings[name] = np.mean(embeddings, axis=0)
                log.info(f"Loaded {len(embeddings)} samples for {name}")
        
        log.info(f"Total registered: {len(self.registered_embeddings)} persons")
    
    def _get_embedding(self, face_roi):
        try:
            # Upscale small faces for better recognition
            h, w = face_roi.shape[:2]
            if w < 80 or h < 80:
                scale = 100 / min(w, h)
                face_roi = cv2.resize(face_roi, (int(w * scale), int(h * scale)))
            
            face = cv2.resize(face_roi, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(tensor)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            log.warning(f"Embedding error: {e}")
            return None
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def recognize(self, face_roi):
        if not self.registered_embeddings:
            return None
        
        embedding = self._get_embedding(face_roi)
        if embedding is None:
            return None
        
        best_name = None
        best_score = 0
        
        for name, known_emb in self.registered_embeddings.items():
            score = self._cosine_similarity(embedding, known_emb)
            if score > best_score and score > 0.5:
                best_score = score
                best_name = name
        
        if best_name:
            log.info(f"Recognized: {best_name} (score: {best_score:.2f})")
        
        return best_name
    
    def add_person(self, name, embeddings):
        self.registered_embeddings[name] = np.mean(embeddings, axis=0)
    
    def retrain(self):
        self.registered_embeddings = {}
        self._load_registered_faces()

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = DeepFaceRecognizer()
    return _recognizer
