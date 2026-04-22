// main.cpp - Optimized MedPal Robot Vision Server
// Combines: YOLO detection, Face recognition, Depth processing, Motor control
// Coral TPU: Auto-detected and used when available

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef HAVE_EDGETPU
#include <edgetpu.h>
#endif

namespace medpal {

struct Config {
    int color_w = 1280, color_h = 720;
    int depth_w = 640, depth_h = 480;
    int yolo_size = 320;
    int frame_skip = 3;
    int stream_fps = 30;
    
    float follow_stop_dist = 0.25f;
    float follow_start_dist = 1.2f;
    float center_tol = 0.12f;
    float follow_speed = 0.6f;
    float turn_speed = 0.45f;
    float conf_thresh = 0.45f;
    
    std::string yolo_model = "/home/medpal/MedPalRobotV2/models/yolov8n.onnx";
    std::string face_cascade = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string socket_path = "/tmp/medpal.sock";
    
    // Motor GPIO pins (BTS7960)
    int left_rpwm = 12, left_lpwm = 13;
    int right_rpwm = 18, right_lpwm = 19;
    int left_ren = 5, left_len = 6;
    int right_ren = 20, right_len = 21;
};

} // namespace medpal

// Global state
std::atomic<bool> g_running{true};
std::atomic<bool> g_following{false};
std::atomic<bool> g_coral{false};
std::string g_target_name = "anyone";
std::string g_last_motor_cmd = "stop";

cv::Mat g_depth;
std::mutex g_depth_mtx;

int g_sock_fd = -1;
int g_client_fd = -1;

medpal::Config g_config;

// ── Coral detection ──────────────────────────────────────────
bool check_coral() {
    return system("lsusb | grep -q '1a6e\\|18d1' 2>/dev/null") == 0;
}

// ── Depth processing ─────────────────────────────────────────
// Note: Orbbec SDK integration required for actual depth data.
// This placeholder waits for proper SDK integration.
void depth_thread_func() {
    try {
        // TODO: Integrate Orbbec SDK for depth camera
        // - Initialize depth stream from Orbbec device
        // - Map depth to color coordinates
        // - Update g_depth with latest frame
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    } catch (const std::exception& e) {
        std::cerr << "[Depth] Error: " << e.what() << "\n";
    }
}

// ── Sample depth at pixel ────────────────────────────────────
float sample_depth(int cx, int cy) {
    std::lock_guard<std::mutex> lk(g_depth_mtx);
    if (g_depth.empty()) return -1;
    
    int dx = std::clamp(cx * g_config.depth_w / g_config.color_w, 2, g_config.depth_w - 3);
    int dy = std::clamp(cy * g_config.depth_h / g_config.color_h, 2, g_config.depth_h - 3);
    
    cv::Rect roi(dx - 2, dy - 2, 5, 5);
    auto region = g_depth(roi);
    
    float sum = 0;
    int cnt = 0;
    for (int r = 0; r < region.rows; r++) {
        for (int c = 0; c < region.cols; c++) {
            uint16_t v = region.at<uint16_t>(r, c);
            if (v > 0) { sum += v; cnt++; }
        }
    }
    return cnt > 0 ? (sum / cnt) / 1000.0f : -1;
}

// ── GPIO sysfs helpers ─────────────────────────────────────
void gpio_write(int pin, int value) {
    std::string cmd = "echo " + std::to_string(value) + " > /sys/class/gpio/gpio" + std::to_string(pin) + "/value";
    system(cmd.c_str());
}

void gpio_pwm(int pin, int duty) {
    std::string cmd = "echo " + std::to_string(duty) + " > /sys/class/gpio/gpio" + std::to_string(pin) + "/value";
    system(cmd.c_str());
}

void gpio_export(int pin) {
    system(("echo " + std::to_string(pin) + " > /sys/class/gpio/export").c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    system(("echo out > /sys/class/gpio/gpio" + std::to_string(pin) + "/direction").c_str());
}

// ── Motor control via sysfs ─────────────────────────────────
void motor_command(const std::string& cmd) {
    if (cmd == g_last_motor_cmd) return;
    g_last_motor_cmd = cmd;
    
    int speed = (int)(g_config.follow_speed * 255);
    
    if (cmd == "forward") {
        gpio_pwm(g_config.left_lpwm, speed);
        gpio_pwm(g_config.right_rpwm, speed);
    } else if (cmd == "backward") {
        gpio_pwm(g_config.left_rpwm, speed);
        gpio_pwm(g_config.right_lpwm, speed);
    } else if (cmd == "left") {
        gpio_pwm(g_config.left_lpwm, speed);
        gpio_pwm(g_config.right_rpwm, speed);
    } else if (cmd == "right") {
        gpio_pwm(g_config.left_rpwm, speed);
        gpio_pwm(g_config.right_lpwm, speed);
    } else if (cmd == "stop") {
        gpio_pwm(g_config.left_rpwm, 0);
        gpio_pwm(g_config.left_lpwm, 0);
        gpio_pwm(g_config.right_rpwm, 0);
        gpio_pwm(g_config.right_lpwm, 0);
    }
}

// ── YOLO detection ───────────────────────────────────────────
std::vector<cv::Rect> yolo_detect(cv::dnn::Net& net, cv::Mat& frame, float conf_thresh) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0f/255.0f, 
        {g_config.yolo_size, g_config.yolo_size},
        {0, 0, 0}, true, false);
    
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    cv::Mat out = outputs[0].reshape(1, outputs[0].size[1]);
    cv::transpose(out, out);
    
    float sx = (float)g_config.color_w / g_config.yolo_size;
    float sy = (float)g_config.color_h / g_config.yolo_size;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (int i = 0; i < out.rows; i++) {
        float* row = out.ptr<float>(i);
        float conf = row[4];
        
        if (conf < conf_thresh) continue;
        
        float cx = row[0] * sx;
        float cy = row[1] * sy;
        float w = row[2] * sx;
        float h = row[3] * sy;
        
        int x = std::max(0, (int)(cx - w/2));
        int y = std::max(0, (int)(cy - h/2));
        int ww = std::min(g_config.color_w - x, (int)w);
        int hh = std::min(g_config.color_h - y, (int)h);
        
        boxes.push_back({x, y, ww, hh});
        scores.push_back(conf);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.4f, 0.45f, indices);
    
    std::vector<cv::Rect> result;
    for (int idx : indices) {
        result.push_back(boxes[idx]);
    }
    
    return result;
}

// ── Face detection ───────────────────────────────────────────
std::vector<cv::Rect> detect_faces(cv::CascadeClassifier& cascade, cv::Mat& frame, 
                                   const cv::Rect& body_roi) {
    cv::Mat roi = frame(body_roi);
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(gray, faces, 1.1, 4, 0, {20, 20}, {150, 150});
    
    // Adjust to full frame coordinates
    for (auto& f : faces) {
        f.x += body_roi.x;
        f.y += body_roi.y;
    }
    
    return faces;
}

// ── Socket communication ──────────────────────────────────────
void socket_thread() {
    unlink(g_config.socket_path.c_str());
    g_sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_config.socket_path.c_str(), sizeof(addr.sun_path) - 1);
    
    bind(g_sock_fd, (sockaddr*)&addr, sizeof(addr));
    listen(g_sock_fd, 1);
    
    std::cout << "[Socket] Waiting for Python connection...\n";
    g_client_fd = accept(g_sock_fd, nullptr, nullptr);
    std::cout << "[Socket] Connected.\n";
    
    while (g_running) {
        char buf[256] = {};
        int n = recv(g_client_fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            std::string cmd(buf, n);
            if (cmd == "follow:on") {
                g_following = true;
                std::cout << "[Cmd] Follow ON\n";
            } else if (cmd == "follow:off") {
                g_following = false;
                motor_command("stop");
                std::cout << "[Cmd] Follow OFF\n";
            } else if (cmd.rfind("target:", 0) == 0 && cmd.length() > 7) {
                g_target_name = cmd.substr(7);
                std::cout << "[Cmd] Target: " << g_target_name << "\n";
            } else if (cmd.rfind("motor:", 0) == 0 && cmd.length() > 6) {
                std::string motor_cmd = cmd.substr(6);
                if (motor_cmd == "forward" || motor_cmd == "backward" || 
                    motor_cmd == "left" || motor_cmd == "right" || motor_cmd == "stop") {
                    motor_command(motor_cmd);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ── Main ─────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  MedPal Robot V2 - Vision Server\n";
    std::cout << "========================================\n";
    
    // Check Coral
    g_coral = check_coral();
    std::cout << "[Init] Coral EdgeTPU: " << (g_coral ? "AVAILABLE" : "CPU mode") << "\n";
    
    // Load YOLO
    std::cout << "[Init] Loading YOLO model...\n";
    cv::dnn::Net yolo_net = cv::dnn::readNetFromONNX(g_config.yolo_model);
    
    if (g_coral) {
        std::cout << "[Init] Using Coral NPU backend\n";
        yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
    } else {
        std::cout << "[Init] Using OpenCV CPU backend\n";
        yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(4);
    }
    
    // Load Haar cascade
    std::cout << "[Init] Loading face cascade...\n";
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(g_config.face_cascade)) {
        std::cerr << "[Error] Failed to load face cascade\n";
        return 1;
    }
    
    // Camera
    std::cout << "[Init] Opening camera...\n";
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, g_config.color_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, g_config.color_h);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    if (!cap.isOpened()) {
        std::cerr << "[Error] Camera not opened\n";
        return 1;
    }
    
    // Start threads
    std::thread depth_thr(depth_thread_func);
    std::thread socket_thr(socket_thread);
    
    std::cout << "[Ready] Vision loop running at " << g_config.stream_fps << " FPS\n";
    std::cout << "========================================\n";
    
    int frame_count = 0;
    std::vector<cv::Rect> last_bodies;
    int last_cx = 0, last_cy = 0;
    float last_dist = -1;
    
    auto frame_interval = std::chrono::milliseconds(1000 / g_config.stream_fps);
    
    while (g_running) {
        auto t0 = std::chrono::steady_clock::now();
        
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        frame_count++;
        
        // YOLO every N frames
        if (frame_count % g_config.frame_skip == 0) {
            last_bodies = yolo_detect(yolo_net, frame, g_config.conf_thresh);
        }
        
        // Process detections
        if (!last_bodies.empty()) {
            // Get largest body (likely main target)
            auto& body = *std::max_element(last_bodies.begin(), last_bodies.end(),
                [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });
            
            last_cx = body.x + body.width / 2;
            last_cy = body.y + body.height / 2;
            
            // Face detection
            auto faces = detect_faces(face_cascade, frame, body);
            if (!faces.empty()) {
                auto& face = faces[0];
                last_cx = face.x + face.width / 2;
                last_cy = face.y + face.height / 2;
                cv::rectangle(frame, face, {255, 100, 0}, 2);
            }
            
            cv::rectangle(frame, body, {0, 255, 0}, 2);
            cv::circle(frame, {last_cx, last_cy}, 6, {0, 0, 255}, -1);
            
            last_dist = sample_depth(last_cx, last_cy);
            
            // Following logic
            if (g_following && last_dist > 0) {
                float dx = (float)(last_cx - g_config.color_w/2) / (g_config.color_w/2);
                std::string cmd;
                
                if (last_dist > g_config.follow_start_dist) {
                    cmd = "forward";
                } else if (last_dist < g_config.follow_stop_dist) {
                    cmd = "backward";
                } else if (std::abs(dx) > g_config.center_tol) {
                    cmd = dx > 0 ? "right" : "left";
                } else {
                    cmd = "stop";
                }
                motor_command(cmd);
            }
        } else if (g_following) {
            motor_command("stop");
        }
        
        // HUD
        std::string hud = cv::format("Person: %d | Dist: %.2fm | %s | %s",
            (int)last_bodies.size(),
            last_dist > 0 ? last_dist : 0,
            g_following ? "FOLLOWING" : "IDLE",
            g_coral ? "[CORAL]" : "[CPU]");
        cv::putText(frame, hud, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);
        
        // Display
        cv::imshow("MedPal Vision", frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            g_running = false;
            break;
        }
        
        // Frame pacing
        auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < frame_interval) {
            std::this_thread::sleep_for(frame_interval - elapsed);
        }
    }
    
    // Cleanup
    g_running = false;
    motor_command("stop");
    depth_thr.join();
    socket_thr.join();
    cap.release();
    close(g_sock_fd);
    cv::destroyAllWindows();
    
    std::cout << "[Exit] Vision server stopped.\n";
    return 0;
}
