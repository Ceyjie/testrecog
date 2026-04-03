# vision_bridge.py
# Connects Python to the C++ vision_server via Unix socket.
# C++ handles: YOLO, depth, Haar, motor commands.
# Python reads status JSON and sends commands back.

import socket, json, threading, logging, time, subprocess, os
import config, motor_control as motors

log = logging.getLogger("bridge")

_status  = {
    "persons": 0, "distance": "?", "action": "IDLE",
    "following": False, "temp": 0, "coral": False,
    "target": ""
}
_lock    = threading.Lock()
_running = False
_sock    = None
_proc    = None

# ── Start C++ process ─────────────────────────────────────
def _start_cpp():
    global _proc
    log.info(f"Launching C++ vision server: {config.CPP_BIN}")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = config.CPP_LIB_PATH
    _proc = subprocess.Popen(
        [config.CPP_BIN],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    # Log C++ output
    def _log_cpp():
        for line in _proc.stdout:
            log.info(f"[C++] {line.decode().strip()}")
    threading.Thread(target=_log_cpp, daemon=True).start()
    log.info(f"C++ PID: {_proc.pid}")

# ── Connect to C++ socket ─────────────────────────────────
def _connect():
    global _sock
    log.info("Waiting for C++ socket...")
    for attempt in range(30):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(config.SOCKET_PATH)
            s.settimeout(2.0)
            _sock = s
            log.info("Connected to C++ vision server.")
            return True
        except Exception:
            time.sleep(0.5)
    log.error("Failed to connect to C++ socket after 30 attempts.")
    return False

# ── Read status from C++ ──────────────────────────────────
def _reader():
    global _status
    buf = ""
    while _running:
        try:
            data = _sock.recv(4096).decode(errors='ignore')
            if not data:
                break
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    st = json.loads(line)
                    with _lock:
                        # Convert float distance to string
                        d = st.get("distance", -1)
                        if isinstance(d, (int, float)) and d > 0:
                            st["distance"] = f"{d:.2f}m"
                        elif isinstance(d, (int, float)):
                            st["distance"] = "?"
                        _status.update(st)
                except json.JSONDecodeError:
                    pass
        except socket.timeout:
            continue
        except Exception as e:
            log.warning(f"Socket read error: {e}")
            break
    log.warning("C++ reader thread exited.")

# ── Send command to C++ ───────────────────────────────────
def _send(cmd: str):
    global _sock
    if _sock:
        try:
            _sock.sendall((cmd + "\n").encode())
            log.debug(f"Sent to C++: {cmd}")
        except Exception as e:
            log.error(f"Socket send error: {e}")

# ── Public API ────────────────────────────────────────────
def start():
    global _running
    _running = True
    _start_cpp()
    time.sleep(2.0)
    if _connect():
        threading.Thread(target=_reader, daemon=True).start()
        log.info("Vision bridge ready.")
    else:
        log.error("Vision bridge failed — C++ not reachable.")

def stop():
    global _running
    _running = False
    if _proc:
        _proc.terminate()
        log.info("C++ process terminated.")
    if _sock:
        try:
            _sock.close()
        except:
            pass

def set_following(val: bool):
    _send(f"follow:{'on' if val else 'off'}")
    with _lock:
        _status["following"] = val
    if not val:
        motors.stop()
    log.info(f"Following → {val}")

def set_target(name: str):
    _send(f"target:{name}")
    with _lock:
        _status["target"] = name
    log.info(f"Target → {name}")

def send_motor(cmd: str):
    """Send motor command directly to C++ (follow/stop/left/right)."""
    _send(f"motor:{cmd}")

def get_status():
    with _lock:
        return _status.copy()

def get_frame():
    """C++ serves JPEG frames via HTTP on port 8081."""
    try:
        import urllib.request
        with urllib.request.urlopen(
            "http://localhost:8081/frame", timeout=0.1) as r:
            return r.read()
    except:
        return None
