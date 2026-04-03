#pragma once
#include <atomic>
#include <cstdint>

namespace medpal {

class MotorController {
public:
    MotorController();
    ~MotorController();
    
    void init(int left_rpwm, int left_lpwm, int right_rpwm, int right_lpwm,
              int left_ren, int left_len, int right_ren, int right_len);
    
    void forward(float speed = 0.5f);
    void backward(float speed = 0.5f);
    void left(float speed = 0.5f);
    void right(float speed = 0.5f);
    void stop();
    
    void set_speed(float speed) { current_speed_.store(speed); }
    float get_speed() const { return current_speed_.load(); }
    
private:
    void write_pwm(int pin, float value);
    void set_direction(bool left_forward, bool right_forward);
    
    std::atomic<float> current_speed_{0.5f};
    int gpio_pins_[8] = {0};
    bool initialized_ = false;
    int pwm_export_fd_ = -1;
};

}
