# motor_bridge.py
# Reads motor commands from vision_bridge status
# and drives the actual GPIO motors.
# Called from main.py in a background thread.

import threading, time, logging
import motor_control as motors
import vision_bridge as vision

log = logging.getLogger("motor_bridge")

_running      = False
_last_cmd     = ""
_manual_mode  = False   # True = manual control, ignore C++ motor cmds
_manual_timer = 0.0
MANUAL_TIMEOUT = 2.0    # seconds after last manual cmd before auto resumes

def _loop():
    global _last_cmd, _manual_mode, _manual_timer
    log.info("Motor bridge started.")
    while _running:
        now = time.time()

        # Auto-release manual mode after timeout
        if _manual_mode and (now - _manual_timer) > MANUAL_TIMEOUT:
            _manual_mode = False
            log.info("Manual mode released — returning to auto.")

        if not _manual_mode:
            st  = vision.get_status()
            cmd = st.get("motor", "stop")

            if cmd != _last_cmd:
                log.info(f"Motor command: {_last_cmd!r} → {cmd!r}")
                _last_cmd = cmd
                if cmd == "forward":
                    motors.set_speed(0.5)
                    motors.forward()
                elif cmd == "left":
                    motors.set_speed(0.4)
                    motors.left()
                elif cmd == "right":
                    motors.set_speed(0.4)
                    motors.right()
                else:
                    motors.stop()

        time.sleep(0.05)  # 20Hz motor update

    motors.stop()
    log.info("Motor bridge stopped.")

_thread = None

def start():
    global _running, _thread
    _running = True
    _thread  = threading.Thread(target=_loop, daemon=True)
    _thread.start()

def stop():
    global _running
    _running = False

def manual(cmd: str):
    """Called by web UI / voice for manual motor commands."""
    global _manual_mode, _manual_timer, _last_cmd
    _manual_mode  = True
    _manual_timer = time.time()
    _last_cmd     = ""  # force re-apply on return to auto
    if cmd == "forward":
        motors.forward()
    elif cmd == "backward":
        motors.backward()
    elif cmd == "left":
        motors.left()
    elif cmd == "right":
        motors.right()
    else:
        motors.stop()
        _manual_mode = False
    log.debug(f"Manual: {cmd}")
