import io
import time
import threading
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from libcamera import Transform
from picamera2 import Picamera2

from piracer.vehicles import PiRacerStandard


PORT = 8000

CAM_W = 480
CAM_H = 360
CAM_FPS = 20

LOWER_ORANGE = np.array([7, 100, 100])
UPPER_ORANGE = np.array([18, 255, 255])

MIN_AREA = 500

THROTTLE = 0.50

KP = 1.50
MAX_STEERING = 1.00
STEERING_SIGN = -1

ROI_TOP_PERCENT = 0.75

car = None


class StreamBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def set_frame(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()


stream_buffer = StreamBuffer()

state_lock = threading.Lock()
state_found = False
state_cx = 0
state_error = 0
state_steering = 0.0
state_throttle = 0.0
state_auto = False


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Orange Line Autopilot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            background: #111;
            color: white;
            text-align: center;
            font-family: Arial;
        }

        img {
            width: 95%;
            max-width: 720px;
            border: 2px solid white;
        }

        button {
            font-size: 22px;
            padding: 14px 24px;
            margin: 10px;
            border: none;
            border-radius: 10px;
            color: white;
        }

        #start {
            background: #008000;
        }

        #stop {
            background: #8b0000;
        }
    </style>
</head>

<body>
    <h2>Orange Line Autopilot</h2>

    <img src="/stream.mjpg">

    <h3 id="state">State: loading...</h3>

    <button id="start" onclick="setAuto(1)">START AUTOPILOT</button>
    <button id="stop" onclick="setAuto(0)">STOP</button>

    <p>
        Orange line follower<br>
        Speed = 30%<br>
        If line is lost, car stops automatically.
    </p>

<script>
function setAuto(value) {
    fetch("/auto?enabled=" + value, {cache: "no-store"}).catch(() => {});
}

setInterval(function() {
    fetch("/state", {cache: "no-store"})
        .then(r => r.json())
        .then(data => {
            document.getElementById("state").innerText =
                "AUTO: " + data.auto +
                " | found: " + data.found +
                " | cx: " + data.cx +
                " | error: " + data.error +
                " | throttle: " + data.throttle.toFixed(2) +
                " | steering: " + data.steering.toFixed(2);
        })
        .catch(() => {
            document.getElementById("state").innerText = "State: disconnected";
        });
}, 300);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global state_auto

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            data = HTML.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/auto":
            enabled = query.get("enabled", ["0"])[0] == "1"

            with state_lock:
                state_auto = enabled

            if not enabled and car is not None:
                car.set_throttle_percent(0.0)
                car.set_steering_percent(0.0)

            self.send_response(204)
            self.end_headers()
            return

        if path == "/state":
            with state_lock:
                obj = {
                    "auto": state_auto,
                    "found": state_found,
                    "cx": state_cx,
                    "error": state_error,
                    "steering": state_steering,
                    "throttle": state_throttle,
                }

            data = json.dumps(obj).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while True:
                    with stream_buffer.condition:
                        stream_buffer.condition.wait()
                        frame = stream_buffer.frame

                    if frame is None:
                        continue

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")

            except Exception:
                pass

            return

        self.send_error(404)


def clamp(x, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, x))


def process_frame(frame):
    global state_found
    global state_cx
    global state_error
    global state_steering
    global state_throttle

    frame = frame.copy()

    h, w, _ = frame.shape

    roi_top = int(h * ROI_TOP_PERCENT)
    roi = frame[roi_top:h, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    moments = cv2.moments(mask)

    found = False
    cx = 0
    error = 0
    steering = 0.0
    throttle = 0.0

    with state_lock:
        auto_enabled = state_auto

    if moments["m00"] > MIN_AREA:
        found = True

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        error = cx - (w // 2)
        error_norm = error / (w / 2)

        steering = STEERING_SIGN * KP * error_norm
        steering = clamp(steering, -MAX_STEERING, MAX_STEERING)

        if auto_enabled:
            throttle = THROTTLE
        else:
            throttle = 0.0

        cv2.circle(roi, (cx, cy), 8, (255, 0, 0), -1)
        cv2.line(roi, (w // 2, 0), (w // 2, roi.shape[0]), (0, 0, 255), 2)
        cv2.line(roi, (cx, 0), (cx, roi.shape[0]), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"AUTO={auto_enabled} cx={cx} err={error} th={throttle:.2f} st={steering:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    else:
        found = False
        cx = 0
        error = 0
        steering = 0.0
        throttle = 0.0

        cv2.putText(
            frame,
            "NO ORANGE LINE - STOP",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    if auto_enabled and found:
        car.set_throttle_percent(throttle)
        car.set_steering_percent(steering)
    else:
        car.set_throttle_percent(0.0)
        car.set_steering_percent(0.0)

    with state_lock:
        state_found = found
        state_cx = cx
        state_error = error
        state_steering = steering
        state_throttle = throttle if auto_enabled and found else 0.0

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.resize(mask_bgr, (w, h))

    combined = np.hstack((frame, mask_bgr))

    ok, jpg = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    if not ok:
        return None

    return jpg.tobytes()


def camera_loop():
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"},
        controls={"FrameRate": CAM_FPS},
        transform=Transform(hflip=1, vflip=1)
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(1)

    while True:
        frame = picam2.capture_array()
        jpg = process_frame(frame)

        if jpg is not None:
            stream_buffer.set_frame(jpg)

        time.sleep(1.0 / CAM_FPS)


def main():
    global car

    print("Starting PiRacer...")
    car = PiRacerStandard()

    car.set_throttle_percent(0.0)
    car.set_steering_percent(0.0)

    print("Starting camera...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    print()
    print("Open:")
    print("http://RASPBERRY_PI_IP:8000")
    print()
    print("Autopilot starts only after pressing START AUTOPILOT on website.")
    print("Speed:", THROTTLE)
    print("KP:", KP)
    print("MAX_STEERING:", MAX_STEERING)
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        car.set_throttle_percent(0.0)
        car.set_steering_percent(0.0)

        try:
            server.server_close()
        except Exception:
            pass

        print("Stopped safely.")


if __name__ == "__main__":
    main()
