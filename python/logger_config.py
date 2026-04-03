# logger_config.py
import logging, os

LOG_FILE = "/home/medpal/MedPalRobotV2/medpal.log"

def setup():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fmt = logging.Formatter(
        '[%(asctime)s][%(name)-10s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    for name in ["urllib3","httpx","httpcore","ultralytics",
                 "multipart","uvicorn.access","uvicorn.error"]:
        logging.getLogger(name).setLevel(logging.WARNING)
