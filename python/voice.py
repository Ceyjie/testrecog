# voice.py — Fast Vosk offline voice recognition
import threading, json, queue, logging, time
import config

log = logging.getLogger("voice")
_running = False
_cb = None
_q = queue.Queue()

COMMANDS = {
    "follow me": "FOLLOW",
    "stop": "STOP",
    "hold": "STOP",
    "go forward": "FORWARD",
    "forward": "FORWARD",
    "come here": "COME",
    "come": "COME",
    "go back": "BACKWARD",
    "backward": "BACKWARD",
    "turn left": "LEFT",
    "go left": "LEFT",
    "turn right": "RIGHT",
    "go right": "RIGHT",
    "status": "STATUS",
}

WAKE_WORDS = ["medpal", "med pal", "hey medpal", "hello medpal", "hi medpal"]

def _audio_cb(indata, frames, time_info, status):
    try:
        if status:
            log.warning(f"Audio status: {status}")
        _q.put(bytes(indata))
    except Exception as e:
        log.error(f"Audio callback error: {e}")

def _loop():
    global _running
    try:
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer
    except ImportError as e:
        log.error(f"Missing: {e}")
        return

    log.info("Loading Vosk model...")
    try:
        model = Model(config.VOSK_MODEL_PATH)
    except Exception as e:
        log.error(f"Vosk model error: {e}")
        return

    rec = KaldiRecognizer(model, 16000)
    log.info(f"Listening for: {WAKE_WORDS}")

    try:
        # List available audio devices for debugging
        try:
            devices = sd.query_devices()
            log.info(f"Audio devices: {devices}")
        except:
            pass
        
        stream = sd.InputStream(
            samplerate=16000,
            blocksize=1024,
            dtype='int16',
            channels=1,
            callback=_audio_cb
        )
        with stream:
            log.info("Audio stream started")
            wake_detected = False
            while _running:
                try:
                    data = _q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").lower().strip()
                    
                    if not text:
                        continue
                    
                    log.info(f"Heard: '{text}'")
                    
                    # Check for wake word first
                    wake_found = False
                    for wake in WAKE_WORDS:
                        if wake in text:
                            wake_found = True
                            break
                    
                    if wake_found:
                        wake_detected = True
                        # Extract command after wake word
                        cmd_text = text
                        for w in WAKE_WORDS:
                            cmd_text = cmd_text.replace(w, "").strip()
                        
                        matched = None
                        for phrase, cmd in COMMANDS.items():
                            if phrase in cmd_text:
                                matched = cmd
                                break
                        
                        if matched and _cb:
                            log.info(f"Command: {matched} from '{cmd_text}'")
                            threading.Thread(target=_cb, args=(matched, cmd_text), daemon=True).start()
                    elif wake_detected:
                        # After wake word, any command without wake is also valid
                        for phrase, cmd in COMMANDS.items():
                            if phrase in text:
                                log.info(f"Command (no wake): {cmd} from '{text}'")
                                if _cb:
                                    threading.Thread(target=_cb, args=(cmd, text), daemon=True).start()
                                break
    except Exception as e:
        log.error(f"Voice error: {e}")
        import traceback
        log.error(traceback.format_exc())

_thread = None

def start(callback):
    global _running, _thread, _cb
    if _running:
        return
    _cb = callback
    _running = True
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()
    log.info("Voice started.")

def stop():
    global _running
    _running = False
    log.info("Voice stopped.")
