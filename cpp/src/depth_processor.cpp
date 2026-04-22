#include "depth_processor.hpp"
#include <iostream>

namespace medpal {

DepthProcessor::DepthProcessor() {}

DepthProcessor::~DepthProcessor() {
    stop();
}

bool DepthProcessor::initialize() {
    try {
        pipe_ = std::make_shared<ob::Pipeline>();
        auto profiles = pipe_->getStreamProfileList(OB_SENSOR_DEPTH);
        auto dpro = profiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y11, 30);
        auto cfg = std::make_shared<ob::Config>();
        cfg->enableStream(dpro);
        pipe_->start(cfg);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Depth initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void DepthProcessor::start() {
    running_ = true;
    thread_ = std::thread(&DepthProcessor::depth_thread, this);
}

void DepthProcessor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (pipe_) pipe_->stop();
}

void DepthProcessor::depth_thread() {
    while (running_) {
        auto fs = pipe_->waitForFrames(100);
        if (!fs) continue;
        auto df = fs->depthFrame();
        if (!df) continue;

        cv::Mat raw(df->height(), df->width(), CV_16UC1, (void*)df->data());
        
        std::lock_guard<std::mutex> lk(mtx_);
        depth_frame_ = raw.clone();
    }
}

float DepthProcessor::get_distance(int x, int y) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (depth_frame_.empty()) return -1.0f;
    
    // Scale coordinates from color (1280x720) to depth (640x480)
    int dx = std::clamp(static_cast<int>(x * 0.5f), 0, depth_frame_.cols - 1);
    int dy = std::clamp(static_cast<int>(y * 0.666f), 0, depth_frame_.rows - 1);
    
    return static_cast<float>(depth_frame_.at<uint16_t>(dy, dx));
}

} // namespace medpal
