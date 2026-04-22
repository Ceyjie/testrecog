# config.py — All configuration for MedPalRobotV2
import os

# Base directory - auto-detect
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── GPIO Pins ─────────────────────────────────────────────
LEFT_RPWM  = 12
LEFT_LPWM  = 13
LEFT_REN   = 5
LEFT_LEN   = 6
RIGHT_RPWM = 18
RIGHT_LPWM = 19
RIGHT_REN  = 20
RIGHT_LEN  = 21
PWM_FREQ   = 1000
DEFAULT_SPEED = 0.6

# ── Person following ──────────────────────────────────────
FOLLOW_STOP_DIST  = 0.20
FOLLOW_START_DIST = 1.5
FRAME_CENTER_TOL  = 0.15
FOLLOW_SPEED      = 0.5
TURN_SPEED        = 0.4

# ── C++ vision server ─────────────────────────────────────
SOCKET_PATH  = "/tmp/medpal.sock"
CPP_BIN      = os.path.join(BASE_DIR, "cpp/build/vision_server")
CPP_LIB_PATH = "/home/medpal/pyorbbecsdk/sdk/lib/arm64"

# ── Vosk voice ────────────────────────────────────────────
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "vosk-model")
WAKE_WORD       = "medpal"

# ── Ollama AI ─────────────────────────────────────────────
OLLAMA_MODEL   = "phi3:mini"
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60

# ── Piper TTS ─────────────────────────────────────────────
PIPER_MODEL = os.path.join(BASE_DIR, "piper-voices/en_US-lessac-medium.onnx")

# ── Paths ─────────────────────────────────────────────────
PERSONS_DIR = os.path.join(BASE_DIR, "data/persons")
LOG_FILE    = os.path.join(BASE_DIR, "medpal.log")
