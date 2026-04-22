# 🤖 MedPal Robot

Web-controlled Raspberry Pi robot using:
- Raspberry Pi 5
- 2x BTS7960 motor drivers
- Flask web interface

## Features

- Forward / Backward / Left / Right control
- Speed slider
- Real-time status display
- Clean dashboard UI
- Safe motor stop on exit

## Folder Structure

MedPalRobot/
│
├── app.py
├── motor_control.py
├── config.py
├── requirements.txt
├── templates/
└── static/

## Installation

Create virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run:

python app.py

Access from browser:

http://YOUR_PI_IP:5000
