# tts.py — Piper TTS local speech
import subprocess, threading, logging
import config

log   = logging.getLogger("tts")
_lock = threading.Lock()

def speak(text: str):
    with _lock:
        try:
            log.info(f"TTS: {text[:80]}")
            piper = subprocess.Popen(
                ['piper', '--model', config.PIPER_MODEL, '--output-raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            aplay = subprocess.Popen(
                ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-'],
                stdin=piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            piper.stdin.write(text.encode())
            piper.stdin.close()
            piper.wait(timeout=15)
            aplay.wait(timeout=15)
        except FileNotFoundError:
            log.error("piper not found. pip install piper-tts")
        except Exception as e:
            log.error(f"TTS error: {e}")

def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()
