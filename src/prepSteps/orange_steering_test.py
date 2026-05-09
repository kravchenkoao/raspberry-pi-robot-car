import io
import time
import threading
import json
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

LOWER_ORANGE = np.array([5, 100, 100])
UPPER_ORANGE = np.array([22, 255, 255])

MIN_AREA = 500

KP = 0.75
MAX_STEERING = 0.90
STEERING_SIGN = -1

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


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Orange Steering Test</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="background:#111; color:white; text-align:center; font-family:Arial;">
    <h2>Orange Line Steering Test</h2>

    <img src="/stream.mjpg" style="width:95%; max-width:720px; border:2px solid white;">

    <h3 id="state">State: loading...</h3>

    <p>
        This test controls steering only.<br>
        Throttle = 0.0
    </p>

<script>
setInterval(function() {
    fetch("/state", {cache: "no-store"})
        .then(r => r.json())
        .then(data => {
            document.getElementById("state").innerText =
                "found: " + data.found +
                " | cx: " + data.cx +
                " | error: " + data.error +
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
        if self.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/state":
            with state_lock:
                obj = {
                    "found": state_found,
                    "cx": state_cx,
                    "error": state_error,
                    "steering": state_steering,
                }

            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/stream.mjpg":
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
    global state_found, state_cx, state_error, state_steering

    frame = frame.copy()

    h, w, _ = frame.shape

    roi_top = int(h * 0.55)
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

    if moments["m00"] > MIN_AREA:
        found = True

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        error = cx - (w // 2)
        error_norm = error / (w / 2)

        steering = STEERING_SIGN * KP * error_norm
        steering = clamp(steering, -MAX_STEERING, MAX_STEERING)

        cv2.circle(roi, (cx, cy), 8, (255, 0, 0), -1)
        cv2.line(roi, (w // 2, 0), (w // 2, roi.shape[0]), (0, 0, 255), 2)
        cv2.line(roi, (cx, 0), (cx, roi.shape[0]), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"cx={cx} error={error} steering={steering:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

    else:
        steering = 0.0

        cv2.putText(
            frame,
            "NO ORANGE LINE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )

    car.set_throttle_percent(0.0)
    car.set_steering_percent(steering)

    with state_lock:
        state_found = found
        state_cx = cx
        state_error = error
        state_steering = steering

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_rgb = cv2.resize(mask_rgb, (w, h))

    combined = np.hstack((frame, mask_rgb))

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
    print("Steering test only. Throttle = 0.0")
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
