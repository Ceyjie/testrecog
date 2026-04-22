import socket
import os
import cv2
import numpy as np
import sys

# Ensure Python can find the face_recognizer module
sys.path.insert(0, "/home/medpal/Desktop/cpp_robot/rebot/python")
import face_recognizer

SOCKET_PATH = "/tmp/medpal_rec.sock"

def start_service():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print("Recognition Service Started and listening on", SOCKET_PATH)

    # Initialize recognizer
    recognizer = face_recognizer.get_recognizer()

    while True:
        conn, _ = server.accept()
        print("Connected to C++ VisionEngine")
        while True:
            try:
                data = conn.recv(1024 * 50)
                if not data: break
                
                # Assume raw image bytes received
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Bypass recognition for testing
                    # name = recognizer.recognize(img)
                    name = "Carl" # FOR TESTING ONLY
                    print(f"Recognized: {name}")
                    conn.send(name.encode())

                else:
                    conn.send(b"error")
            except Exception as e:
                print(f"Error: {e}")
                break
        conn.close()
        print("Disconnected. Waiting for reconnection...")

if __name__ == "__main__":
    start_service()
