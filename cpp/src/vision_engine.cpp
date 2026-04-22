#include "vision_engine.hpp"
#include "logger.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <unistd.h>

namespace medpal {

VisionEngine::VisionEngine() {}

VisionEngine::~VisionEngine() {
    stop();
}

bool VisionEngine::initialize() {
    LOG_INFO("Initializing VisionEngine...");

    if (!depth_proc_.initialize()) {
        LOG_ERROR("Failed to initialize Astra Depth sensor");
    }

    // Initialize Camera
    cap_.open(0, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        LOG_ERROR("Failed to open camera!");
        return false;
    }
    
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap_.set(cv::CAP_PROP_FPS, 30);
    
    // Load YOLO
    LOG_INFO("Loading YOLO model...");
    detector_ = cv::dnn::readNetFromONNX("/home/medpal/Desktop/cpp_robot/rebot/models/yolo11n.onnx");
    detector_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    detector_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    // Load YuNet
    LOG_INFO("Loading YuNet model...");
    face_detector_ = cv::FaceDetectorYN::create(
        "/home/medpal/Desktop/cpp_robot/rebot/models/face_yunet.onnx",
        "", cv::Size(1280, 720), 0.9f, 0.3f, 5000, 0, 0);
    
    return true;
}

void VisionEngine::start() {
    LOG_INFO("Starting VisionEngine threads...");
    running_ = true;
    capture_thr_ = std::thread(&VisionEngine::capture_thread, this);
    inference_thr_ = std::thread(&VisionEngine::inference_thread, this);
    // Start streaming thread
    std::thread(&VisionEngine::streaming_thread, this).detach(); 
    depth_proc_.start();
}

void VisionEngine::stop() {
    LOG_INFO("Stopping VisionEngine...");
    running_ = false;
    if (capture_thr_.joinable()) capture_thr_.join();
    if (inference_thr_.joinable()) inference_thr_.join();
    depth_proc_.stop();
    cap_.release();
}

std::string VisionEngine::recognize_face(const cv::Mat& face_crop) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/tmp/medpal_rec.sock", sizeof(addr.sun_path)-1);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return "service_down";
    }
    
    std::vector<uchar> buf;
    cv::imencode(".jpg", face_crop, buf);
    send(sock, buf.data(), buf.size(), 0);
    
    char res[64] = {0};
    recv(sock, res, sizeof(res)-1, 0);
    close(sock);
    
    return std::string(res);
}

void VisionEngine::streaming_thread() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 3);

    LOG_INFO("MJPEG Streaming started on port 8080");

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 50};

    while (running_) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        std::string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);

        while (running_) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lk(processed_frame_mtx_);
                if (processed_frame_.empty()) continue;
                frame = processed_frame_.clone();
            }

            std::vector<uchar> buf;
            cv::imencode(".jpg", frame, buf, params);
            std::string msg = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
            send(client_fd, msg.c_str(), msg.length(), 0);
            send(client_fd, buf.data(), buf.size(), 0);
            send(client_fd, "\r\n", 2, 0);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        close(client_fd);
    }
    close(server_fd);
}

void VisionEngine::gpio_pwm(int pin, int duty) {
    std::string path = "/sys/class/gpio/gpio" + std::to_string(pin) + "/value";
    std::ofstream ofs(path);
    if (ofs.is_open()) {
        ofs << duty;
    }
}

void VisionEngine::set_motor_speed(const std::string& cmd, float speed) {
    int duty = static_cast<int>(speed * 255);
    
    if (cmd == "forward") {
        gpio_pwm(13, duty); // Left LPWM
        gpio_pwm(18, duty); // Right RPWM
    } else if (cmd == "stop") {
        gpio_pwm(12, 0);
        gpio_pwm(13, 0);
        gpio_pwm(18, 0);
        gpio_pwm(19, 0);
    }
}

void VisionEngine::capture_thread() {
    while (running_) {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) continue;
        
        std::lock_guard<std::mutex> lk(raw_frame_mtx_);
        raw_frame_ = frame.clone();
    }
}

void VisionEngine::inference_thread() {
    while (running_) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lk(raw_frame_mtx_);
            if (raw_frame_.empty()) continue;
            frame = raw_frame_.clone();
        }
        
        // 1. Run detections
        std::vector<cv::Rect> body_detections = run_yolo(frame);
        std::vector<cv::Rect> face_detections = run_face_detection(frame);
        
        // 2. Recognize faces
        for (const auto& box : face_detections) {
            cv::Rect valid_box = box & cv::Rect(0, 0, frame.cols, frame.rows);
            if(valid_box.width < 10 || valid_box.height < 10) continue;
            
            cv::Mat face_crop = frame(valid_box);
            std::string name = recognize_face(face_crop); 
            
            cv::rectangle(frame, valid_box, {255, 100, 0}, 2);
            cv::putText(frame, name, {valid_box.x, valid_box.y - 10}, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 100, 0}, 2);
        }

        // Draw body detections
        for (const auto& box : body_detections) {
            cv::rectangle(frame, box, {0, 255, 0}, 2);
        }
        
        std::lock_guard<std::mutex> lk(processed_frame_mtx_);
        processed_frame_ = frame;
    }
}

std::vector<cv::Rect> VisionEngine::run_yolo(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0f/255.0f, {320, 320}, {0, 0, 0}, true, false);
    detector_.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    detector_.forward(outputs, detector_.getUnconnectedOutLayersNames());
    
    cv::Mat out = outputs[0].reshape(1, outputs[0].size[1]);
    cv::transpose(out, out);
    
    float sx = 1280.0f / 320.0f;
    float sy = 720.0f / 320.0f;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (int i = 0; i < out.rows; i++) {
        float* row = out.ptr<float>(i);
        if (row[4] < 0.45f) continue;
        
        float cx = row[0] * sx;
        float cy = row[1] * sy;
        float w = row[2] * sx;
        float h = row[3] * sy;
        
        boxes.push_back({(int)(cx - w/2), (int)(cy - h/2), (int)w, (int)h});
        scores.push_back(row[4]);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.4f, 0.45f, indices);
    
    std::vector<cv::Rect> result;
    for (int idx : indices) result.push_back(boxes[idx]);
    
    return result;
}

std::vector<cv::Rect> VisionEngine::run_face_detection(cv::Mat& frame) {
    cv::Mat faces;
    face_detector_->detect(frame, faces);
    
    std::vector<cv::Rect> result;
    for (int i = 0; i < faces.rows; ++i) {
        result.push_back(cv::Rect(
            (int)faces.at<float>(i, 0), (int)faces.at<float>(i, 1),
            (int)faces.at<float>(i, 2), (int)faces.at<float>(i, 3)));
    }
    return result;
}

cv::Mat VisionEngine::get_latest_frame() {
    std::lock_guard<std::mutex> lk(processed_frame_mtx_);
    return processed_frame_.clone();
}

} // namespace medpal
