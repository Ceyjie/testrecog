#pragma once
#include "libobsensor/ObSensor.hpp"
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <memory>

namespace medpal {

class DepthProcessor {
public:
    DepthProcessor();
    ~DepthProcessor();

    bool initialize();
    void start();
    void stop();
    
    // Returns distance in millimeters
    float get_distance(int x, int y);

private:
    void depth_thread();

    std::shared_ptr<ob::Pipeline> pipe_;
    cv::Mat depth_frame_;
    std::mutex mtx_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace medpal
