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

LOWER_ORANGE = np.array([0, 100, 100])
UPPER_ORANGE = np.array([25, 255, 255])

MIN_AREA = 500

ROI_TOP_PERCENT = 0.4
ROI_BOTTOM_PERCENT = 0.85

KP = 1.50
MAX_STEERING = 1.00
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

state_mode = "none"
state_contours = 0
state_left_x = 0
state_right_x = 0
state_lane_center = 0
state_error = 0
state_steering = 0.0


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Lane Steering ROI Test</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="background:#111; color:white; text-align:center; font-family:Arial;">
    <h2>Orange Lane - Steering ROI Test</h2>

    <img src="/stream.mjpg" style="width:95%; max-width:1000px; border:2px solid white;">

    <h3 id="state">State: loading...</h3>

    <p>
        Steering only. Throttle = 0.0<br>
        ROI is limited between top and bottom percent.<br>
        If one line is visible, missing line is assumed at image edge.
    </p>

<script>
setInterval(function() {
    fetch("/state", {cache: "no-store"})
        .then(r => r.json())
        .then(data => {
            document.getElementById("state").innerText =
                "mode: " + data.mode +
                " | contours: " + data.contours +
                " | left_x: " + data.left_x +
                " | right_x: " + data.right_x +
                " | lane_center: " + data.lane_center +
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
                    "mode": state_mode,
                    "contours": state_contours,
                    "left_x": state_left_x,
                    "right_x": state_right_x,
                    "lane_center": state_lane_center,
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
    global state_mode
    global state_contours
    global state_left_x
    global state_right_x
    global state_lane_center
    global state_error
    global state_steering

    frame = frame.copy()

    h, w, _ = frame.shape

    roi_top = int(h * ROI_TOP_PERCENT)
    roi_bottom = int(h * ROI_BOTTOM_PERCENT)

    roi = frame[roi_top:roi_bottom, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    good_contours = []

    for c in contours:
        area = cv2.contourArea(c)

        if area >= MIN_AREA:
            x, y, cw, ch = cv2.boundingRect(c)
            cx = x + cw // 2
            cy = y + ch // 2

            good_contours.append({
                "contour": c,
                "area": area,
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "cx": cx,
                "cy": cy,
            })

    good_contours = sorted(good_contours, key=lambda item: item["area"], reverse=True)

    selected = good_contours[:2]
    selected = sorted(selected, key=lambda item: item["cx"])

    camera_center = w // 2

    left_x = 0
    right_x = 0
    lane_center = camera_center
    error = 0
    steering = 0.0
    mode = "no_line"

    cv2.line(frame, (0, roi_top), (w, roi_top), (0, 255, 255), 2)
    cv2.line(frame, (0, roi_bottom), (w, roi_bottom), (0, 255, 255), 2)

    cv2.line(frame, (camera_center, roi_top), (camera_center, roi_bottom), (255, 0, 0), 2)

    if len(selected) >= 2:
        left = selected[0]
        right = selected[1]

        left_x = left["cx"]
        right_x = right["cx"]

        lane_center = (left_x + right_x) // 2
        mode = "two_lines"

        for item in selected:
            x = item["x"]
            y = item["y"]
            cw = item["w"]
            ch = item["h"]
            cx = item["cx"]
            cy = item["cy"]

            y_full = y + roi_top
            cy_full = cy + roi_top

            cv2.rectangle(frame, (x, y_full), (x + cw, y_full + ch), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy_full), 7, (0, 255, 0), -1)

    elif len(selected) == 1:
        one = selected[0]

        one_x = one["cx"]

        x = one["x"]
        y = one["y"]
        cw = one["w"]
        ch = one["h"]
        cx = one["cx"]
        cy = one["cy"]

        y_full = y + roi_top
        cy_full = cy + roi_top

        cv2.rectangle(frame, (x, y_full), (x + cw, y_full + ch), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy_full), 7, (0, 255, 0), -1)

        if one_x < camera_center:
            left_x = one_x
            right_x = w - 1
            lane_center = (left_x + right_x) // 2
            mode = "one_left_edge_fallback"

            cv2.line(frame, (right_x, roi_top), (right_x, roi_bottom), (255, 0, 255), 2)

        else:
            left_x = 0
            right_x = one_x
            lane_center = (left_x + right_x) // 2
            mode = "one_right_edge_fallback"

            cv2.line(frame, (left_x, roi_top), (left_x, roi_bottom), (255, 0, 255), 2)

    else:
        lane_center = camera_center
        mode = "no_line"

    error = lane_center - camera_center
    error_norm = error / (w / 2)

    if mode != "no_line":
        steering = STEERING_SIGN * KP * error_norm
        steering = clamp(steering, -MAX_STEERING, MAX_STEERING)
    else:
        steering = 0.0

    car.set_throttle_percent(0.0)
    car.set_steering_percent(steering)

    cv2.line(frame, (lane_center, roi_top), (lane_center, roi_bottom), (0, 0, 255), 3)
    cv2.circle(frame, (lane_center, roi_top + 25), 8, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"{mode} center={lane_center} error={error} steering={steering:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_top:roi_bottom, 0:w] = mask

    mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((frame, mask_bgr))

    ok, jpg = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    if not ok:
        return None

    with state_lock:
        state_mode = mode
        state_contours = len(good_contours)
        state_left_x = left_x
        state_right_x = right_x
        state_lane_center = lane_center
        state_error = error
        state_steering = steering

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
    print("Steering only. Throttle = 0.0")
    print("ROI top:", ROI_TOP_PERCENT)
    print("ROI bottom:", ROI_BOTTOM_PERCENT)
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

        print("Stopped")


if __name__ == "__main__":
    main()
