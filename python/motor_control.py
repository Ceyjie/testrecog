# motor_control.py — BTS7960 motor control with GPIO fallback
import logging, time
import config

log = logging.getLogger("motor")

GPIO_OK = False

class _MockPWM:
    value = 0
class _MockDig:
    def on(self): pass
    def off(self): pass

def _try_gpiozero():
    try:
        from gpiozero import PWMOutputDevice, DigitalOutputDevice
        return PWMOutputDevice, DigitalOutputDevice
    except Exception as e:
        log.warning(f"gpiozero failed: {e}")
        return None, None

def _try_pigpio():
    try:
        import pigpio
        pi = pigpio.pi()
        if pi.connected:
            log.info("Using pigpio daemon")
            return pi
        pi.stop()
    except Exception as e:
        log.warning(f"pigpio failed: {e}")
    return None

def _init_gpiozero():
    global GPIO_OK
    PWMOut, DigitalOut = _try_gpiozero()
    if PWMOut is None:
        return False
    
    try:
        global left_rpwm, left_lpwm, right_rpwm, right_lpwm
        global left_ren, left_len, right_ren, right_len
        
        left_rpwm  = PWMOut(config.LEFT_RPWM,  frequency=config.PWM_FREQ)
        left_lpwm  = PWMOut(config.LEFT_LPWM,  frequency=config.PWM_FREQ)
        right_rpwm = PWMOut(config.RIGHT_RPWM, frequency=config.PWM_FREQ)
        right_lpwm = PWMOut(config.RIGHT_LPWM, frequency=config.PWM_FREQ)
        left_ren   = DigitalOut(config.LEFT_REN)
        left_len   = DigitalOut(config.LEFT_LEN)
        right_ren  = DigitalOut(config.RIGHT_REN)
        right_len  = DigitalOut(config.RIGHT_LEN)
        
        GPIO_OK = True
        return True
    except Exception as e:
        log.warning(f"gpiozero init failed: {e}")
        return False

def _init_pigpio(pi):
    global GPIO_OK
    try:
        import pigpio as pg
        pins = [
            config.LEFT_RPWM, config.LEFT_LPWM,
            config.RIGHT_RPWM, config.RIGHT_LPWM,
            config.LEFT_REN, config.LEFT_LEN,
            config.RIGHT_REN, config.RIGHT_LEN
        ]
        for pin in pins:
            pi.set_mode(pin, pg.OUTPUT)
            pi.write(pin, 0)
        
        pi.set_PWM_frequency(config.LEFT_RPWM, config.PWM_FREQ)
        pi.set_PWM_frequency(config.LEFT_LPWM, config.PWM_FREQ)
        pi.set_PWM_frequency(config.RIGHT_RPWM, config.PWM_FREQ)
        pi.set_PWM_frequency(config.RIGHT_LPWM, config.PWM_FREQ)
        
        log.info("pigpio initialized successfully")
        GPIO_OK = True
        return True, pi
    except Exception as e:
        log.warning(f"pigpio init failed: {e}")
        return False, None

PWMOut, DigitalOut = _try_gpiozero()
pi = _try_pigpio()

if pi:
    _success, pi = _init_pigpio(pi)
    if _success:
        pass
    else:
        pi = None

if not pi:
    _init_gpiozero()

if not GPIO_OK:
    log.warning("No GPIO available — using MOCK mode")
    left_rpwm = left_lpwm = right_rpwm = right_lpwm = _MockPWM()
    left_ren = left_len = right_ren = right_len = _MockDig()
    pi = None

current_speed = config.DEFAULT_SPEED

def enable_motors():
    if pi:
        pi.write(config.LEFT_REN, 1)
        pi.write(config.LEFT_LEN, 1)
        pi.write(config.RIGHT_REN, 1)
        pi.write(config.RIGHT_LEN, 1)
    elif GPIO_OK:
        left_ren.on(); left_len.on()
        right_ren.on(); right_len.on()
    log.info("Motors enabled.")

def disable_motors():
    if pi:
        pi.write(config.LEFT_REN, 0)
        pi.write(config.LEFT_LEN, 0)
        pi.write(config.RIGHT_REN, 0)
        pi.write(config.RIGHT_LEN, 0)
    elif GPIO_OK:
        left_ren.off(); left_len.off()
        right_ren.off(); right_len.off()
    log.info("Motors disabled.")

def stop():
    if pi:
        pi.set_PWM_dutycycle(config.LEFT_RPWM, 0)
        pi.set_PWM_dutycycle(config.LEFT_LPWM, 0)
        pi.set_PWM_dutycycle(config.RIGHT_RPWM, 0)
        pi.set_PWM_dutycycle(config.RIGHT_LPWM, 0)
    elif GPIO_OK:
        left_rpwm.value = 0
        left_lpwm.value = 0
        right_rpwm.value = 0
        right_lpwm.value = 0
    log.debug("STOP")

def forward():
    stop()
    duty = int(current_speed * 255)
    if pi:
        pi.set_PWM_dutycycle(config.LEFT_RPWM, duty)
        pi.set_PWM_dutycycle(config.RIGHT_RPWM, duty)
    elif GPIO_OK:
        left_rpwm.value = current_speed
        right_rpwm.value = current_speed
    log.debug(f"FORWARD @ {current_speed:.2f}")

def backward():
    stop()
    duty = int(current_speed * 255)
    if pi:
        pi.set_PWM_dutycycle(config.LEFT_LPWM, duty)
        pi.set_PWM_dutycycle(config.RIGHT_LPWM, duty)
    elif GPIO_OK:
        left_lpwm.value = current_speed
        right_lpwm.value = current_speed
    log.debug(f"BACKWARD @ {current_speed:.2f}")

def left():
    stop()
    duty = int(current_speed * 255)
    if pi:
        pi.set_PWM_dutycycle(config.LEFT_LPWM, duty)
        pi.set_PWM_dutycycle(config.RIGHT_RPWM, duty)
    elif GPIO_OK:
        left_lpwm.value = current_speed
        right_rpwm.value = current_speed
    log.debug(f"LEFT @ {current_speed:.2f}")

def right():
    stop()
    duty = int(current_speed * 255)
    if pi:
        pi.set_PWM_dutycycle(config.LEFT_RPWM, duty)
        pi.set_PWM_dutycycle(config.RIGHT_LPWM, duty)
    elif GPIO_OK:
        left_rpwm.value = current_speed
        right_lpwm.value = current_speed
    log.debug(f"RIGHT @ {current_speed:.2f}")

def set_speed(speed):
    global current_speed
    current_speed = max(0.0, min(1.0, float(speed)))
    log.debug(f"Speed → {current_speed:.2f}")
    return current_speed

def get_speed():
    return current_speed

def cleanup():
    stop()
    disable_motors()
    if pi:
        pi.stop()
    log.info("Motors cleaned up.")

enable_motors()
