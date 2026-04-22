import cv2
import argparse
import time

parser = argparse.ArgumentParser(description="MedPal Robot Stream Viewer")
parser.add_argument(
    "--ip",
    required=True,
    help="Raspberry Pi IP address (e.g. 192.168.1.42)"
)
parser.add_argument(
    "--port",
    default=5000,
    type=int,
    help="Pi Flask port (default: 5000)"
)
parser.add_argument(
    "--fullscreen",
    action="store_true",
    help="Display stream fullscreen"
)
args = parser.parse_args()

stream_url = f"http://{args.ip}:{args.port}/video"

print(f"\n{'='*50}")
print(f"  MedPal Robot Viewer")
print(f"{'='*50}")
print(f"  Connecting to: {stream_url}")
print(f"  Press Q to quit")
print(f"{'='*50}\n")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Could not connect.")
    print("   Check:")
    print("   1. Pi IP is correct")
    print("   2. app.py is running on the Pi")
    print("   3. Both devices are on the same WiFi network")
    exit(1)

print("✅ Connected!\n")

window_name = "MedPal Robot View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if args.fullscreen:
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

consecutive_failures = 0
MAX_FAILURES = 30

while True:
    ret, frame = cap.read()

    if not ret:
        consecutive_failures += 1
        if consecutive_failures >= MAX_FAILURES:
            print("⚠️  Connection lost. Reconnecting...")
            cap.open(stream_url)
            consecutive_failures = 0
            time.sleep(0.5)
        continue

    consecutive_failures = 0
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Viewer closed.")
