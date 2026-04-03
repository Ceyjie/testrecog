#!/bin/bash
# setup.sh — MedPalRobotV2 full setup
# Run once: bash setup.sh

set -e
echo "=== MedPalRobotV2 Setup ==="

BASE="/home/medpal/MedPalRobotV2"
cd $BASE

# ── 1. Create folders ─────────────────────────────────────
echo "[1/8] Creating folders..."
mkdir -p cpp/src cpp/include cpp/build
mkdir -p python/templates python/static
mkdir -p models data/persons piper-voices
echo "  Done."

# ── 2. Python deps ────────────────────────────────────────
echo "[2/8] Installing Python packages..."
pip install fastapi uvicorn vosk sounddevice requests \
            piper-tts pathvalidate gpiozero \
            --break-system-packages -q
echo "  Done."

# ── 3. Vosk model ─────────────────────────────────────────
if [ ! -d "$BASE/vosk-model" ]; then
  echo "[3/8] Downloading Vosk model..."
  wget -q https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip -q vosk-model-small-en-us-0.15.zip
  mv vosk-model-small-en-us-0.15 vosk-model
  rm vosk-model-small-en-us-0.15.zip
  echo "  Done."
else
  echo "[3/8] Vosk model already exists, skipping."
fi

# ── 4. Piper TTS voice ────────────────────────────────────
if [ ! -f "$BASE/piper-voices/en_US-lessac-medium.onnx" ]; then
  echo "[4/8] Downloading Piper voice..."
  cd piper-voices
  wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
  wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
  cd ..
  echo "  Done."
else
  echo "[4/8] Piper voice already exists, skipping."
fi

# ── 5. YOLO ONNX model ────────────────────────────────────
if [ ! -f "$BASE/models/yolov8n.onnx" ]; then
  echo "[5/8] Exporting YOLOv8n ONNX model..."
  python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='onnx', imgsz=320)
import shutil, os
shutil.move('yolov8n.onnx', '$BASE/models/yolov8n.onnx')
print('  Exported to models/yolov8n.onnx')
"
else
  echo "[5/8] YOLO model already exists, skipping."
fi

# ── 6. Ollama ─────────────────────────────────────────────
echo "[6/8] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
  echo "  Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
echo "  Starting Ollama and pulling phi3:mini..."
ollama serve &>/dev/null &
sleep 3
ollama pull phi3:mini
echo "  Done."

# ── 7. Build C++ vision server ────────────────────────────
echo "[7/8] Building C++ vision server..."
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j$(nproc)
cd ../..
echo "  Built: cpp/build/vision_server"

# ── 8. Copy files check ───────────────────────────────────
echo "[8/8] Checking Python files..."
for f in main.py config.py motor_control.py motor_bridge.py \
          vision_bridge.py ai_chat.py voice.py tts.py \
          person_registry.py logger_config.py; do
  if [ -f "python/$f" ]; then
    echo "  OK: python/$f"
  else
    echo "  MISSING: python/$f  <-- copy this file!"
  fi
done

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Register yourself:"
echo "     python3 register_person.py --name YourName"
echo ""
echo "  2. Start the robot:"
echo "     cd python && python3 main.py"
echo ""
echo "  3. Open browser:"
echo "     http://$(hostname -I | awk '{print $1}'):5000"
