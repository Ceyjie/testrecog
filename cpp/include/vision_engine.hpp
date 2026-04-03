#pragma once
#include <atomic>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace medpal {

struct RobotConfig {
    int color_width = 1280;
    int color_height = 720;
    int depth_width = 640;
    int depth_height = 480;
    int yolo_size = 320;
    int frame_skip = 3;
    int stream_fps = 30;
    
    float follow_stop_dist = 0.25f;
    float follow_start_dist = 1.2f;
    float center_tolerance = 0.12f;
    float follow_speed = 0.6f;
    float turn_speed = 0.45f;
    float confidence_threshold = 0.45f;
    
    std::string yolo_model = "/home/medpal/MedPalRobotV2/models/yolov8n.onnx";
    std::string face_cascade = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string socket_path = "/tmp/medpal.sock";
};

struct Detection {
    cv::Rect body_box;
    cv::Rect face_box;
    float confidence = 0.0f;
    int center_x = 0;
    int center_y = 0;
    float distance = -1.0f;
    std::string person_name;
};

class VisionEngine {
public:
    VisionEngine(const RobotConfig& config);
    ~VisionEngine();
    
    bool init();
    void start();
    void stop();
    
    Detection get_detection();
    bool coral_available() const { return coral_available_; }
    bool is_following() const { return following_; }
    
    void set_following(bool val) { following_ = val; }
    void set_target(const std::string& name) { target_name_ = name; }
    
private:
    void camera_loop();
    void depth_loop();
    
    Detection detect_person(cv::Mat& frame);
    cv::Mat capture_frame();
    float sample_depth(int x, int y);
    
    RobotConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> following_{false};
    std::atomic<bool> coral_available_{false};
    
    cv::VideoCapture cap_;
    cv::dnn::Net yolo_net_;
    cv::CascadeClassifier face_cascade_;
    
    cv::Mat latest_frame_;
    cv::Mat depth_frame_;
    Detection current_detection_;
    
    std::mutex frame_mtx_;
    std::mutex depth_mtx_;
    std::mutex detection_mtx_;
    
    std::thread camera_thread_;
    std::thread depth_thread_;
};

}
