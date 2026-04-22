#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_engine.hpp"

int main() {
    std::cout << "[MedPal] Initializing VisionEngine..." << std::endl;

    medpal::VisionEngine engine;
    if (!engine.initialize()) {
        std::cerr << "[Error] Failed to initialize VisionEngine!" << std::endl;
        return 1;
    }

    engine.start();
    std::cout << "[MedPal] VisionEngine started. Streaming on port 8080." << std::endl;
    std::cout << "[MedPal] Press Ctrl+C to exit." << std::endl;

    // Keep main thread alive
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    engine.stop();
    return 0;
}
