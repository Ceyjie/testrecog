# MedPal Robot V2 - Quick Start Guide

## Project Structure

```
MedPalRobotV2/
├── cpp/                      # C++ Vision Engine
│   ├── src/
│   │   └── main.cpp         # Main vision server
│   ├── include/             # Header files
│   ├── build/               # Build output
│   ├── CMakeLists.txt
│   └── build.sh
├── python/                   # Python orchestration
│   ├── main.py              # FastAPI server
│   ├── vision_bridge.py     # C++ communication
│   ├── motor_control.py     # Motor control
│   ├── voice.py             # Voice commands
│   └── ai_chat.py           # AI chat
├── models/                   # ML models
│   ├── yolov8n.onnx         # YOLOv8 detection
│   └── face_embedding.onnx  # Face recognition
├── data/
│   └── persons/             # Registered faces
├── docs/
│   └── BUILD_GUIDE.md       # Full documentation
└── setup.sh                 # Initial setup
```

## Quick Start

### 1. First Time Setup
```bash
cd /home/medpal/MedPalRobotV2
bash setup.sh
```

### 2. Register a Person
```bash
cd /home/medpal/MedPalRobotV2
python3 register_person.py --name "John" --samples 50
```

### 3. Start Robot
```bash
cd python
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000
```

### 4. Access Web Interface
```
http://medpal.local:5000
```

## Voice Commands

| Command | Action |
|---------|--------|
| "follow me" | Start following you |
| "stop" | Stop movement |
| "forward" | Move forward |
| "backward" | Move backward |
| "turn left" | Turn left |
| "turn right" | Turn right |
| "status" | Report robot status |

## Web Controls

- **WASD / Arrow Keys** - Manual movement
- **Follow Toggle** - Enable/disable person following
- **Target Select** - Choose person to follow
- **Speed Slider** - Adjust motor speed

## Coral EdgeTPU Setup

When you plug in the Coral USB:
```bash
# Install runtime
pip install tflite-runtime edgetpu

# Verify
python3 -c "import edgetpu; print('Coral ready!')"
```

The C++ code automatically detects and uses Coral when available.

## Troubleshooting

### Camera not detected
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

### Coral not working
```bash
lsusb | grep coral
# Should show: 1a6e:18d1
```

### Motor issues
```bash
# Test GPIO
gpio mode 12 OUT
gpio write 12 1
```
