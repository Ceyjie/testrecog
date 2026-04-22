#pragma once
#include "depth_processor.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

namespace medpal {

class VisionEngine {
public:
    VisionEngine();
    ~VisionEngine();

    bool initialize();
    void start();
    void stop();

    cv::Mat get_latest_frame();
    std::string recognize_face(const cv::Mat& face_crop);
    void set_motor_speed(const std::string& cmd, float speed);

private:
    void capture_thread();
    void inference_thread();
    void streaming_thread();

    std::vector<cv::Rect> run_yolo(cv::Mat& frame);
    std::vector<cv::Rect> run_face_detection(cv::Mat& frame);

    void gpio_pwm(int pin, int duty);

    std::atomic<bool> running_{false};
    
    cv::VideoCapture cap_;
    cv::dnn::Net detector_;
    cv::Ptr<cv::FaceDetectorYN> face_detector_;
    
    DepthProcessor depth_proc_;
    
    cv::Mat raw_frame_;
    std::mutex raw_frame_mtx_;
    cv::Mat processed_frame_;
    std::mutex processed_frame_mtx_;
    
    std::thread capture_thr_;
    std::thread inference_thr_;
};

} // namespace medpal
