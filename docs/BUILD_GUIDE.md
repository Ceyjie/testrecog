# MedPal Robot V2 - Complete Build Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      RASPBERRY PI 5                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Python FastAPI (Orchestration Layer)                    │   │
│  │  ├── Voice Commands (Vosk) ──► Motor Control            │   │
│  │  ├── Web UI (HTTP)              ├── Person Following     │   │
│  │  └── AI Chat (Ollama)          └── Face Recognition      │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │ Unix Socket                              │
│  ┌────────────────────▼───────────────────────────────────┐   │
│  │  C++ Vision Engine (Performance Critical)               │   │
│  │  ├── YOLO Detection (CPU/NPU/Coral)                     │   │
│  │  ├── Depth Processing (Orbbec Astra)                   │   │
│  │  ├── Face Detection (Haar Cascade)                      │   │
│  │  ├── Face Recognition (Embedding + Cosine Similarity)  │   │
│  │  └── Person Tracking & Following                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Hardware:                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐   │
│  │ Orbbec Astra│ │ Google Coral│ │ 4x Motor + 2x BTS7960   │   │
│  │    Camera   │ │   USB       │ │      Driver Board        │   │
│  └─────────────┘ └─────────────┘ └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites Checklist

- [x] 2x Motor Driver BTS7960
- [x] 4x Motor 12V 412RPM  
- [x] Raspberry Pi 5 16GB
- [ ] Google Coral USB (plug & play when available)
- [x] Orbbec Astra Camera

---

## Step 1: System Setup

### 1.1 Flash Raspberry Pi OS
```bash
# Download Raspberry Pi Imager from https://www.raspberrypi.com/software/
# Choose: Raspberry Pi OS (64-bit) with desktop
# Configure: WiFi, SSH, hostname: medpal
```

### 1.2 Initial System Configuration
```bash
# SSH into Pi
ssh medpal@medpal.local

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y cmake build-essential pkg-config \
    libgl1-mesa-dev libglu1-mesa-dev libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    python3-pip python3-dev python3-venv \
    libatlas-base-dev libhdf5-dev \
    git wget curl htop

# Enable I2C and SPI (for motor drivers)
sudo raspi-config
# Navigate: Interface Options → I2C/SPI → Enable
```

### 1.3 Python Virtual Environment
```bash
cd /home/medpal/MedPalRobotV2
python3 -m venv venv
source venv/bin/activate

# Core Python packages
pip install --upgrade pip
pip install numpy opencv-python-headless ultralytics \
    fastapi uvicorn vosk sounddevice requests \
    gpiozero picamera2 pyttsx3
```

---

## Step 2: Orbbec Astra Camera Setup

### 2.1 Install Orbbec SDK
```bash
cd /home/medpal
git clone https://github.com/OrbbecProfessional/pyorbbecsdk.git
cd pyorbbecsdk/sdk
chmod +x install.sh
sudo ./install.sh

# Verify installation
ls /usr/local/lib/libobsensor*
```

### 2.2 Camera Permissions
```bash
# Add udev rules for Orbbec camera
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/60-orbbec.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Step 3: Coral USB EdgeTPU Setup

### 3.1 Install EdgeTPU Runtime (when Coral is plugged in)
```bash
# Check if Coral is detected
lsusb | grep -i coral
# Expected: "Bus 001 Device 002: ID 1a6e:18d1 Google Inc."

# Install EdgeTPU runtime
cd /tmp
wget https://github.com/google-coral/libedgetpu/releases/download/release_STABLE_ec13fba7ed/tflite_runtime-2.14.0-cp39-cp39-linux_aarch64.whl
pip install tflite_runtime-2.14.0-cp39-cp39-linux_aarch64.whl

# Install PyCoral
pip install pycoral
```

### 3.2 Auto-Detection in Code
The C++ code automatically detects Coral at runtime:
```cpp
bool coral_available() {
    return system("lsusb | grep -q '1a6e\\|18d1' 2>/dev/null") == 0;
}
```

---

## Step 4: C++ Vision Engine (Optimized)

### 4.1 CMakeLists.txt (Optimized)
```cmake
cmake_minimum_required(VERSION 3.20)
project(medpal_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenCV
find_package(OpenCV 4.8 REQUIRED HINTS 
    /usr/lib/aarch64-linux-gnu/cmake/opencv4
    /usr/local/share/opencv4
)

# Orbbec SDK
set(OB_SDK "/home/medpal/pyorbbecsdk/sdk")
include_directories(${OB_SDK}/include)
link_directories(${OB_SDK}/lib/arm64)

# Find libedgetpu if available
find_library(EDGETPU_LIBRARY edgetpu HINTS /usr/lib /usr/local/lib)
if(EDGETPU_LIBRARY)
    message(STATUS "Coral EdgeTPU detected")
    add_definitions(-DHAVE_EDGETPU=1)
endif()

# Compiler optimizations
add_compile_options(-O3 -march=armv8-a+fp+simd -ffast-math)
add_compile_options(-Wall -Wextra)

add_executable(vision_server
    src/vision_server.cpp
    src/face_recognizer.cpp
    src/motor_controller.cpp
)

target_link_libraries(vision_server
    ${OpenCV_LIBS}
    ${OpenCV_LIBS}::opencv_dnn
    OrbbecSDK
    pthread
    ${EDGETPU_LIBRARY}
)

# Install
install(TARGETS vision_server DESTINATION bin)
```

### 4.2 Optimized vision_server.cpp
```cpp
// vision_server.cpp - Optimized version with face recognition
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include "libobsensor/ObSensor.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <fstream>

namespace Config {
    constexpr int COLOR_W = 1280, COLOR_H = 720;
    constexpr int DEPTH_W = 640, DEPTH_H = 480;
    constexpr int YOLO_SIZE = 320;
    constexpr int FRAME_SKIP = 3;  // Process every 3rd frame
    constexpr int STREAM_FPS = 30;
    constexpr float FOLLOW_STOP_DIST = 0.25f;
    constexpr float FOLLOW_START_DIST = 1.2f;
    constexpr float CENTER_TOL = 0.12f;
    constexpr float FOLLOW_SPEED = 0.6f;
    constexpr float TURN_SPEED = 0.45f;
    constexpr float CONFIDENCE_THRESHOLD = 0.45f;
    
    const std::string YOLO_MODEL = "/home/medpal/MedPalRobotV2/models/yolov8n.onnx";
    const std::string FACE_CASCADE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    const std::string EMBEDDINGS_DIR = "/home/medpal/MedPalRobotV2/data/persons/embeddings/";
    const std::string SOCKET_PATH = "/tmp/medpal.sock";
}

std::atomic<bool> g_running{true};
std::atomic<bool> g_following{false};
std::atomic<bool> g_coral_available{false};
std::string g_target_name = "anyone";

cv::Mat g_depth;
std::mutex g_depth_mtx;
std::vector<float> g_current_embedding;
std::mutex g_emb_mtx;

struct Detection {
    cv::Rect box;
    float confidence;
    cv::Rect face_box;
    cv::Mat embedding;
};

Detection detect_person(cv::Mat& frame, cv::dnn::Net& net, cv::CascadeClassifier& haar);
float compare_embeddings(const std::vector<float>& a, const std::vector<float>& b);
void send_motor_command(const std::string& cmd);
float sample_depth(int x, int y);

int main() {
    std::cout << "[MedPal] Starting vision server...\n";
    
    // Check Coral
    g_coral_available = (system("lsusb | grep -q '1a6e\\|18d1' 2>/dev/null") == 0);
    std::cout << "[MedPal] Coral TPU: " << (g_coral_available ? "AVAILABLE" : "CPU") << "\n";
    
    // Initialize Orbbec pipeline
    ob::Pipeline pipe;
    auto config = std::make_shared<ob::Config>();
    auto profiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    auto dprofile = profiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y16, 30);
    config->enableStream(dprofile);
    pipe.start(config);
    
    // Depth processing thread
    std::thread depth_thr([&]() {
        while (g_running) {
            auto frames = pipe.waitForFrames(100);
            if (auto df = frames->depthFrame()) {
                cv::Mat raw(df->height(), df->width(), CV_16UC1, (void*)df->data());
                cv::Mat mm; raw.convertTo(mm, CV_16UC1, df->getValueScale());
                std::lock_guard<std::mutex> lk(g_depth_mtx);
                g_depth = mm.clone();
            }
        }
    });
    
    // Initialize YOLO
    cv::dnn::Net yolo_net = cv::dnn::readNetFromONNX(Config::YOLO_MODEL);
    if (g_coral_available) {
        yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
    } else {
        yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(4);
    }
    
    // Haar cascade for face detection
    cv::CascadeClassifier face_cascade;
    face_cascade.load(Config::FACE_CASCADE);
    
    // Camera
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, Config::COLOR_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::COLOR_H);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    int frame_count = 0;
    Detection last_detection;
    
    while (g_running) {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;
        
        frame_count++;
        
        // YOLO detection every N frames
        if (frame_count % Config::FRAME_SKIP == 0) {
            last_detection = detect_person(frame, yolo_net, face_cascade);
        }
        
        // Draw detection
        if (!last_detection.box.empty()) {
            cv::rectangle(frame, last_detection.box, {0, 255, 0}, 2);
            if (!last_detection.face_box.empty()) {
                cv::rectangle(frame, last_detection.face_box, {255, 100, 0}, 2);
            }
            
            int cx = last_detection.box.x + last_detection.box.width/2;
            int cy = last_detection.box.y + last_detection.box.height/2;
            cv::circle(frame, {cx, cy}, 5, {0, 0, 255}, -1);
            
            float dist = sample_depth(cx, cy);
            if (dist > 0) {
                std::string dist_str = cv::format("%.2fm", dist);
                cv::putText(frame, dist_str, {cx+10, cy-10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 200, 255}, 2);
            }
            
            // Following logic
            if (g_following && dist > 0) {
                float dx = (float)(cx - Config::COLOR_W/2) / (Config::COLOR_W/2);
                std::string cmd;
                
                if (dist > Config::FOLLOW_START_DIST) {
                    cmd = "forward";
                } else if (dist < Config::FOLLOW_STOP_DIST) {
                    cmd = "backward";
                } else if (std::abs(dx) > Config::CENTER_TOL) {
                    cmd = dx > 0 ? "right" : "left";
                } else {
                    cmd = "stop";
                }
                send_motor_command(cmd);
            }
        }
        
        // Overlay info
        std::string overlay = cv::format("Persons: %d | Follow: %s | Coral: %s",
            !last_detection.box.empty(), g_following?"ON":"OFF", g_coral_available?"YES":"NO");
        cv::putText(frame, overlay, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);
        
        // Frame rate control
        auto elapsed = std::chrono::steady_clock::now() - t0;
        float ms = std::chrono::duration<float>(elapsed).count() * 1000;
        int delay = std::max(1, (int)(1000.0/Config::STREAM_FPS - ms));
        cv::waitKey(delay);
    }
    
    g_running = false;
    depth_thr.join();
    pipe.stop();
    return 0;
}

Detection detect_person(cv::Mat& frame, cv::dnn::Net& net, cv::CascadeClassifier& haar) {
    Detection det;
    
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, {Config::YOLO_SIZE, Config::YOLO_SIZE},
        {0, 0, 0}, true, false);
    net.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    cv::Mat out = outputs[0].reshape(1, outputs[0].size[1]);
    cv::transpose(out, out);
    
    float scale_x = (float)Config::COLOR_W / Config::YOLO_SIZE;
    float scale_y = (float)Config::COLOR_H / Config::YOLO_SIZE;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (int i = 0; i < out.rows; i++) {
        float* row = out.ptr<float>(i);
        float confidence = row[4];
        if (confidence < Config::CONFIDENCE_THRESHOLD) continue;
        
        float cx = row[0] * scale_x;
        float cy = row[1] * scale_y;
        float w = row[2] * scale_x;
        float h = row[3] * scale_y;
        
        boxes.push_back({(int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h});
        scores.push_back(confidence);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.4f, 0.45f, indices);
    
    if (!indices.empty()) {
        det.box = boxes[indices[0]];
        det.confidence = scores[indices[0]];
        
        // Face detection within body
        cv::Mat gray;
        cv::cvtColor(frame(det.box), gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        haar.detectMultiScale(gray, faces, 1.1, 4, 0, {20, 20}, {100, 100});
        
        if (!faces.empty()) {
            det.face_box = faces[0] + cv::Point(det.box.tl());
        }
    }
    
    return det;
}

float sample_depth(int x, int y) {
    std::lock_guard<std::mutex> lk(g_depth_mtx);
    if (g_depth.empty()) return -1;
    
    int dx = std::clamp(x * Config::DEPTH_W / Config::COLOR_W, 2, Config::DEPTH_W-3);
    int dy = std::clamp(y * Config::DEPTH_H / Config::COLOR_H, 2, Config::DEPTH_H-3);
    
    cv::Rect roi(dx-2, dy-2, 5, 5);
    auto region = g_depth(roi);
    float sum = 0; int count = 0;
    
    for (int i = 0; i < region.rows; i++)
        for (int j = 0; j < region.cols; j++) {
            uint16_t v = region.at<uint16_t>(i, j);
            if (v > 0) { sum += v; count++; }
        }
    
    return count > 0 ? (sum / count) / 1000.0f : -1;
}

void send_motor_command(const std::string& cmd) {
    // Send to Python via socket or directly control GPIO
    // Implementation depends on motor control architecture
}
```

---

## Step 5: Face Recognition System

### 5.1 Face Embedding Extractor
```cpp
// face_recognizer.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class FaceRecognizer {
public:
    FaceRecognizer(const std::string& embeddings_dir);
    std::vector<float> extract_embedding(const cv::Mat& face);
    std::string recognize(const std::vector<float>& embedding, float threshold = 0.6f);
    void save_embedding(const std::string& name, const std::vector<float>& embedding);
    
private:
    std::string embeddings_dir;
    cv::dnn::Net embed_net;
    std::vector<std::pair<std::string, std::vector<float>>> known_embeddings;
    
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    void load_embeddings();
};
```

### 5.2 Face Recognition Implementation
```cpp
// face_recognizer.cpp
#include "face_recognizer.h"

FaceRecognizer::FaceRecognizer(const std::string& dir) : embeddings_dir(dir) {
    // Load FaceNet MobileNet-based embedding model
    // Use InsightFace or ArcFace model for best accuracy
    std::string model_path = "/home/medpal/MedPalRobotV2/models/face_embedding.onnx";
    embed_net = cv::dnn::readNetFromONNX(model_path);
    embed_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    embed_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    load_embeddings();
}

std::vector<float> FaceRecognizer::extract_embedding(const cv::Mat& face) {
    cv::Mat blob;
    cv::dnn::blobFromImage(face, blob, 1/127.5, {112, 112}, {127.5, 127.5, 127.5}, true, false);
    
    embed_net.setInput(blob);
    cv::Mat embedding = embed_net.forward();
    
    // L2 normalize
    float norm = 0;
    for (int i = 0; i < embedding.cols; i++) norm += embedding.at<float>(0, i) * embedding.at<float>(0, i);
    norm = std::sqrt(norm);
    for (int i = 0; i < embedding.cols; i++) embedding.at<float>(0, i) /= norm;
    
    std::vector<float> result;
    embedding.copyTo(result);
    return result;
}

float FaceRecognizer::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::string FaceRecognizer::recognize(const std::vector<float>& embedding, float threshold) {
    float best_sim = 0;
    std::string best_name = "unknown";
    
    for (const auto& [name, known] : known_embeddings) {
        float sim = cosine_similarity(embedding, known);
        if (sim > best_sim && sim > threshold) {
            best_sim = sim;
            best_name = name;
        }
    }
    
    return best_name;
}

void FaceRecognizer::save_embedding(const std::string& name, const std::vector<float>& embedding) {
    std::ofstream f(embeddings_dir + name + ".bin", std::ios::binary);
    size_t size = embedding.size();
    f.write((char*)&size, sizeof(size));
    f.write((char*)embedding.data(), embedding.size() * sizeof(float));
}
```

---

## Step 6: Motor Control (C++ for Real-Time)

### 6.1 High-Performance Motor Controller
```cpp
// motor_controller.h
#pragma once
#include <atomic>
#include <thread>
#include <chrono>

class MotorController {
public:
    MotorController(int left_rpwm, int left_lpwm, int right_rpwm, int right_lpwm,
                   int left_ren, int left_len, int right_ren, int right_len);
    ~MotorController();
    
    void forward(float speed = 0.5f);
    void backward(float speed = 0.5f);
    void left(float speed = 0.5f);
    void right(float speed = 0.5f);
    void stop();
    void set_speed(float speed);
    
private:
    void init_gpio();
    void cleanup();
    
    int left_rpwm_, left_lpwm_, right_rpwm_, right_lpwm_;
    int left_ren_, left_len_, right_ren_, right_len_;
    std::atomic<float> current_speed_{0.5f};
    int pwm_fd_[4];
};
```

---

## Step 7: Voice Command System (Python)

### 7.1 Optimized Voice Handler
```python
# voice_handler.py - Low-latency voice commands
import vosk
import sounddevice as sd
import json
import threading
import logging
from queue import Queue

log = logging.getLogger("voice")

class VoiceHandler:
    def __init__(self, model_path, callback):
        self.callback = callback
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.audio_queue = Queue(maxsize=8)
        self.running = False
        
        # Optimized settings for Raspberry Pi
        self.sample_rate = 16000
        self.buffer_size = 4096
        
        # Command mapping
        self.commands = {
            "follow": ["follow me", "follow", "track me"],
            "stop": ["stop", "halt", "wait"],
            "forward": ["go forward", "move forward", "ahead"],
            "backward": ["go back", "move backward", "back"],
            "left": ["turn left", "go left"],
            "right": ["turn right", "go right"],
            "status": ["status", "how are you", "report"],
        }
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.start()
        log.info("Voice handler started")
    
    def _listen_loop(self):
        with sd.InputStream(samplerate=self.sample_rate, 
                          blocksize=self.buffer_size,
                          dtype='int16',
                          channels=1) as stream:
            while self.running:
                data, _ = stream.read(self.buffer_size)
                self.audio_queue.put(data.tobytes())
                
                if self.rec.AcceptWaveform(data.tobytes()):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").lower()
                    if text:
                        cmd = self._match_command(text)
                        if cmd:
                            log.info(f"Voice command: {cmd}")
                            self.callback(cmd, text)
    
    def _match_command(self, text):
        for cmd, phrases in self.commands.items():
            for phrase in phrases:
                if phrase in text:
                    return cmd
        return None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
```

---

## Step 8: System Integration

### 8.1 Main Orchestration (Python)
```python
# robot_main.py - Central orchestration
import asyncio
import logging
from fastapi import FastAPI, WebSocket
import uvicorn

from voice_handler import VoiceHandler
from motor_controller import MotorController
from vision_bridge import VisionBridge
import tts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("robot")

app = FastAPI()
motors = MotorController()
vision = VisionBridge()

@app.on_event("startup")
async def startup():
    log.info("Starting MedPal Robot...")
    vision.start()
    voice = VoiceHandler("/home/medpal/MedPalRobotV2/vosk-model", on_voice_command)
    voice.start()
    log.info("All systems ready!")

@app.on_event("shutdown")
async def shutdown():
    vision.stop()
    motors.stop()

def on_voice_command(cmd, text):
    if cmd == "follow":
        vision.set_following(True)
        tts.speak("Following you now")
    elif cmd == "stop":
        vision.set_following(False)
        motors.stop()
        tts.speak("Stopped")
    elif cmd == "forward":
        motors.forward()
    elif cmd == "backward":
        motors.backward()
    elif cmd == "left":
        motors.left()
    elif cmd == "right":
        motors.right()
    elif cmd == "status":
        status = vision.get_status()
        tts.speak(f"I see {status['persons']} people. Distance is {status['distance']} meters")

@app.get("/status")
async def status():
    return vision.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

---

## Step 9: Build and Run

### 9.1 Build Script
```bash
#!/bin/bash
set -e
cd /home/medpal/MedPalRobotV2

echo "=== Building C++ Vision Engine ==="
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

echo "=== Installing Python dependencies ==="
source ../venv/bin/activate
pip install -r requirements.txt

echo "=== Downloading models ==="
# YOLO
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320)"

# Face embedding model (ArcFace)
wget -O models/face_embedding.onnx "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model.onnx"

echo "=== Setup complete! ==="
```

### 9.2 Run Commands
```bash
cd /home/medpal/MedPalRobotV2/python

# Register a person
python3 register_person.py --name "John" --samples 50

# Start robot (run from python directory)
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000

# Access web interface
# http://medpal.local:5000
```

---

## Step 10: Performance Optimization Tips

### 10.1 For Raspberry Pi 5
```bash
# Enable maximum performance
sudo apt install cpufrequtils
sudo cpufreq-set -g performance

# Increase GPU memory
echo "gpu_mem=256" | sudo tee -a /boot/config.txt

# Use DMA for GPIO
echo "dtoverlay=gpio-no-irq" | sudo tee -a /boot/config.txt
```

### 10.2 YOLO Model Selection
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.3MB | Fastest | Low | Testing |
| YOLOv8s | 22MB | Fast | Medium | Production |
| YOLOv8m | 52MB | Medium | High | High accuracy |
| YOLOv8n-OpenVINO | 6.3MB | Very Fast | Low | Intel NPU |

### 10.3 Coral Optimization
```bash
# Install Coral runtime with optimal settings
pip install edgetpu tflite-runtime

# Set threads for Coral
export EDGETPU_THREAD_NUM=4
```

---

## Troubleshooting

### Camera not detected
```bash
lsusb | grep -i orbbec
v4l2-ctl --list-devices
```

### Coral not working
```bash
lsusb | grep "1a6e\|18d1"
# If not found, install: https://coral.ai/docs/accelerator/get-started/
```

### Motor issues
```bash
# Test GPIO
gpio mode 12 PWM
gpio pwm 12 512
```
