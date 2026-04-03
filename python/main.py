#!/usr/bin/env python3
# main.py — MedPal Robot CLI Controller
# No web interface - just voice + vision + motor control

import logging
import threading
import time
import sys
import os
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
log = logging.getLogger("main")

# Add python path
sys.path.insert(0, '/home/medpal/MedPalRobotV2/python')

import motor_control as motors
import vision_server_standalone
import voice
import tts

# Global state
vision_server = None
following = False
running = True
show_camera = True

def on_voice_command(cmd: str, text: str):
    """Handle voice commands"""
    global following
    
    log.info(f"Voice command: {cmd} | '{text}'")
    st = vision_server.get_status() if vision_server else {}
    target = st.get("target_name", "")
    
    if cmd == "FOLLOW":
        following = True
        vision_server.set_following(True) if vision_server else None
        tts.speak("Following you now.")
    
    elif cmd == "STOP":
        following = False
        vision_server.set_following(False) if vision_server else None
        motors.stop()
        tts.speak("Stopped.")
    
    elif cmd == "FORWARD":
        following = False
        vision_server.set_following(False) if vision_server else None
        motors.forward()
        tts.speak("Moving forward.")
    
    elif cmd == "COME":
        following = True
        vision_server.set_following(True) if vision_server else None
        tts.speak("Coming to you.")
    
    elif cmd == "BACKWARD":
        following = False
        vision_server.set_following(False) if vision_server else None
        motors.backward()
        tts.speak("Moving backward.")
    
    elif cmd == "LEFT":
        following = False
        vision_server.set_following(False) if vision_server else None
        motors.left()
        tts.speak("Turning left.")
    
    elif cmd == "RIGHT":
        following = False
        vision_server.set_following(False) if vision_server else None
        motors.right()
        tts.speak("Turning right.")
    
    elif cmd == "STATUS":
        if target:
            tts.speak(f"I see {target}.")
        else:
            tts.speak("I see no registered person.")


def follow_loop():
    """Background thread for person following"""
    global running
    
    log.info("Follow loop started")
    
    while running:
        if following and vision_server:
            info = vision_server.get_tracking_info()
            action = info.get("action", "IDLE")
            target_name = info.get("target_name", "")
            
            # Only follow registered person
            if not target_name:
                motors.stop()
                time.sleep(0.1)
                continue
            
            # Execute action based on vision
            if action == "MOVE_FORWARD":
                motors.forward()
            elif action == "MOVE_BACKWARD":
                motors.backward()
            elif action == "TURN_LEFT":
                motors.left()
            elif action == "TURN_RIGHT":
                motors.right()
            elif action == "CENTERED":
                motors.stop()
            elif action == "TOO_CLOSE":
                motors.backward()
            else:
                motors.stop()
        else:
            motors.stop()
        
        time.sleep(0.1)


def main():
    global vision_server, running
    
    log.info("=== MedPal Robot Starting ===")
    
    # Initialize vision
    log.info("Starting vision system...")
    vision_server = vision_server_standalone.start_vision()
    time.sleep(2)
    
    # Initialize voice
    log.info("Starting voice recognition...")
    voice.start(on_voice_command)
    time.sleep(1)
    
    # Start follow loop
    follow_thread = threading.Thread(target=follow_loop, daemon=True)
    follow_thread.start()
    
    log.info("=== MedPal Robot Ready ===")
    log.info("Say 'MedPal' to activate, then 'follow me' to start following")
    log.info("Press Ctrl+C to stop")
    
    # Enable motors
    motors.enable_motors()
    
    # Main loop - show camera and keep alive
    try:
        while running:
            if show_camera and vision_server:
                frame = vision_server.latest_frame
                if frame is not None:
                    cv2.imshow("MedPal Robot", frame)
                    
                    # Press 'q' to quit, 'f' to toggle follow
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                    elif key == ord('f'):
                        following = not following
                        vision_server.set_following(following)
                        if following:
                            tts.speak("Following you now.")
                        else:
                            motors.stop()
                            tts.speak("Stopped.")
            
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        log.info("\nStopping...")
        running = False
    
    cv2.destroyAllWindows()
    
    # Cleanup
    vision_server.stop()
    voice.stop()
    motors.cleanup()
    log.info("MedPal Robot stopped.")


if __name__ == "__main__":
    main()
