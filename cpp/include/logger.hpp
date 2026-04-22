#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace medpal {

enum class LogLevel { INFO, WARN, ERROR };

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::string levelStr;
        switch (level) {
            case LogLevel::INFO: levelStr = "[INFO]"; break;
            case LogLevel::WARN: levelStr = "[WARN]"; break;
            case LogLevel::ERROR: levelStr = "[ERROR]"; break;
        }

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") 
           << " " << levelStr << " " << message;
        
        if (logFile_.is_open()) {
            logFile_ << ss.str() << std::endl;
        }
        std::cout << ss.str() << std::endl;
    }

private:
    Logger() {
        logFile_.open("/home/medpal/Desktop/cpp_robot/rebot/medpal.log", std::ios::app);
    }
    ~Logger() {
        if (logFile_.is_open()) logFile_.close();
    }
    std::ofstream logFile_;
    std::mutex mutex_;
};

#define LOG_INFO(msg) medpal::Logger::getInstance().log(medpal::LogLevel::INFO, msg)
#define LOG_WARN(msg) medpal::Logger::getInstance().log(medpal::LogLevel::WARN, msg)
#define LOG_ERROR(msg) medpal::Logger::getInstance().log(medpal::LogLevel::ERROR, msg)

} // namespace medpal
