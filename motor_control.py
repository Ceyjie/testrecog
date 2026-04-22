# motor_control.py — pure lgpio, no gpiozero
# WHY: gpiozero opens its own chip handle. After a crash, the kernel
# still marks those pins as owned by the dead process. Our lgpio
# gpio_free() call cannot release a handle owned by another process.
# Using lgpio directly means WE own the handle and can always reclaim.

import subprocess
import lgpio
import threading
import time
import config

# Kill any zombie process holding gpiochip0
subprocess.run("sudo fuser -k /dev/gpiochip0",
               shell=True, stderr=subprocess.DEVNULL)

_chip = lgpio.gpiochip_open(0)

def _claim(pin):
    try:
        lgpio.gpio_free(_chip, pin)
    except Exception:
        pass
    lgpio.gpio_claim_output(_chip, pin, 0)

_ALL_PINS = [
    config.LEFT_RPWM, config.LEFT_LPWM,
    config.RIGHT_RPWM, config.RIGHT_LPWM,
    config.LEFT_REN,  config.LEFT_LEN,
    config.RIGHT_REN, config.RIGHT_LEN,
]
for _p in _ALL_PINS:
    _claim(_p)

# =========================
# GLOBAL STATE
# =========================
current_speed = config.DEFAULT_SPEED
_ramp_thread = None
_ramp_running = False
_active_pins = []  # list of (pin, duty) tuples currently driving

# =========================
# Low-level helpers
# =========================
def _pwm(pin, duty):
    lgpio.tx_pwm(_chip, pin, config.PWM_FREQ, duty * 100)

def _digital(pin, val):
    lgpio.gpio_write(_chip, pin, 1 if val else 0)

# =========================
# Ramping
# =========================
def _ramp_worker(target, duration):
    global _ramp_running
    _ramp_running = True
    steps = 20
    for i in range(1, steps + 1):
        if not _ramp_running:
            break
        spd = target * i / steps
        for pin in _active_pins:
            _pwm(pin, spd)
        time.sleep(duration / steps)
    for pin in _active_pins:
        _pwm(pin, target)

def _start_ramp(target, duration=0.4):
    global _ramp_thread, _ramp_running
    _ramp_running = False
    if _ramp_thread and _ramp_thread.is_alive():
        _ramp_thread.join(0.1)
    _ramp_thread = threading.Thread(
        target=_ramp_worker, args=(target, duration), daemon=True)
    _ramp_thread.start()

def _stop_ramp():
    global _ramp_running
    _ramp_running = False

# =========================
# Enable / Disable
# =========================
def enable_motors():
    _digital(config.LEFT_REN,  1)
    _digital(config.LEFT_LEN,  1)
    _digital(config.RIGHT_REN, 1)
    _digital(config.RIGHT_LEN, 1)
    print("✅ Motors enabled")

def disable_motors():
    _digital(config.LEFT_REN,  0)
    _digital(config.LEFT_LEN,  0)
    _digital(config.RIGHT_REN, 0)
    _digital(config.RIGHT_LEN, 0)
    print("🛑 Motors disabled")

enable_motors()

# =========================
# Speed control
# =========================
def set_speed(speed, auto_apply=True):
    global current_speed
    try:
        speed = float(speed)
        current_speed = max(0.0, min(1.0, speed))
        if auto_apply and _active_pins:
            _stop_ramp()
            _start_ramp(current_speed)
        print(f"⚡ Speed: {current_speed:.0%}")
        return current_speed
    except Exception:
        return current_speed

def get_speed():
    return current_speed

# =========================
# Movement
# =========================
def stop():
    global _active_pins
    _stop_ramp()
    _active_pins = []
    for pin in [config.LEFT_RPWM, config.LEFT_LPWM,
                config.RIGHT_RPWM, config.RIGHT_LPWM]:
        _pwm(pin, 0)

def forward():
    global _active_pins
    stop()
    _active_pins = [config.LEFT_RPWM, config.RIGHT_RPWM]
    _pwm(config.LEFT_LPWM,  0)
    _pwm(config.RIGHT_LPWM, 0)
    _pwm(config.LEFT_RPWM,  current_speed)
    _pwm(config.RIGHT_RPWM, current_speed)

def backward():
    global _active_pins
    stop()
    _active_pins = [config.LEFT_LPWM, config.RIGHT_LPWM]
    _pwm(config.LEFT_RPWM,  0)
    _pwm(config.RIGHT_RPWM, 0)
    _pwm(config.LEFT_LPWM,  current_speed)
    _pwm(config.RIGHT_LPWM, current_speed)

def left():
    # Swing turn: only right wheel drives
    global _active_pins
    stop()
    _active_pins = [config.RIGHT_RPWM]
    _pwm(config.LEFT_RPWM,  0)
    _pwm(config.LEFT_LPWM,  0)
    _pwm(config.RIGHT_LPWM, 0)
    _pwm(config.RIGHT_RPWM, current_speed)

def right():
    # Swing turn: only left wheel drives
    global _active_pins
    stop()
    _active_pins = [config.LEFT_RPWM]
    _pwm(config.RIGHT_RPWM, 0)
    _pwm(config.RIGHT_LPWM, 0)
    _pwm(config.LEFT_LPWM,  0)
    _pwm(config.LEFT_RPWM,  current_speed)

# =========================
# Cleanup
# =========================
def cleanup():
    stop()
    disable_motors()
    for pin in _ALL_PINS:
        try:
            lgpio.gpio_free(_chip, pin)
        except Exception:
            pass
    try:
        lgpio.gpiochip_close(_chip)
    except Exception:
        pass
    print("🧹 GPIO cleaned up")
