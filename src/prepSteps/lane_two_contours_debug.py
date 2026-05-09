import io
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from libcamera import Transform
from picamera2 import Picamera2


PORT = 8000

CAM_W = 480
CAM_H = 360
CAM_FPS = 20

LOWER_ORANGE = np.array([3, 100, 100])
UPPER_ORANGE = np.array([22, 255, 255])

MIN_AREA = 500
ROI_TOP_PERCENT = 0.5


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

state_contours = 0
state_left_x = 0
state_right_x = 0
state_lane_center = 0
state_error = 0
state_mode = "none"


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Lane Contours Debug</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="background:#111; color:white; text-align:center; font-family:Arial;">
    <h2>Orange Lane - Two Contours Debug</h2>

    <img src="/stream.mjpg" style="width:95%; max-width:1000px; border:2px solid white;">

    <h3 id="state">State: loading...</h3>

    <p>
        Left image: camera with detected contours<br>
        Right image: orange mask only in ROI<br>
        No motor control in this test.
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
                " | error: " + data.error;
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
                    "contours": state_contours,
                    "left_x": state_left_x,
                    "right_x": state_right_x,
                    "lane_center": state_lane_center,
                    "error": state_error,
                    "mode": state_mode,
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


def process_frame(frame):
    global state_contours
    global state_left_x
    global state_right_x
    global state_lane_center
    global state_error
    global state_mode

    frame = frame.copy()

    h, w, _ = frame.shape

    roi_top = int(h * ROI_TOP_PERCENT)
    roi = frame[roi_top:h, 0:w]

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
    lane_center = 0
    error = 0
    mode = "none"

    cv2.line(frame, (0, roi_top), (w, roi_top), (0, 255, 255), 2)
    cv2.line(frame, (camera_center, roi_top), (camera_center, h), (255, 0, 0), 2)

    if len(selected) >= 2:
        left = selected[0]
        right = selected[1]

        left_x = left["cx"]
        right_x = right["cx"]

        lane_center = (left_x + right_x) // 2
        error = lane_center - camera_center
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

        cv2.line(frame, (lane_center, roi_top), (lane_center, h), (0, 0, 255), 3)
        cv2.circle(frame, (lane_center, roi_top + 20), 8, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"LEFT={left_x} RIGHT={right_x} CENTER={lane_center} ERROR={error}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    elif len(selected) == 1:
        one = selected[0]

        left_x = one["cx"]
        right_x = 0
        lane_center = one["cx"]
        error = lane_center - camera_center
        mode = "one_line"

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

        cv2.line(frame, (cx, roi_top), (cx, h), (0, 0, 255), 2)

        cv2.putText(
            frame,
            f"ONLY ONE LINE cx={cx} error={error}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    else:
        mode = "no_line"

        cv2.putText(
            frame,
            "NO ORANGE CONTOURS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_top:h, 0:w] = mask

    mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((frame, mask_bgr))

    ok, jpg = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    if not ok:
        return None

    with state_lock:
        state_contours = len(good_contours)
        state_left_x = left_x
        state_right_x = right_x
        state_lane_center = lane_center
        state_error = error
        state_mode = mode

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
    print("Starting camera...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    print()
    print("Open:")
    print("http://RASPBERRY_PI_IP:8000")
    print()
    print("Two orange contours debug only. No motor control.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        try:
            server.server_close()
        except Exception:
            pass

        print("Stopped")


if __name__ == "__main__":
    main()
