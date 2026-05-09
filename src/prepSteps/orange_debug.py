import io
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from libcamera import Transform
from picamera2 import Picamera2


PORT = 8000

CAM_W = 320
CAM_H = 240
CAM_FPS = 15

LOWER_ORANGE = np.array([7, 100, 100])
UPPER_ORANGE = np.array([18, 255, 255])

class StreamBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def set_frame(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()


stream_buffer = StreamBuffer()


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Yellow Line Debug</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="background:#111; color:white; text-align:center; font-family:Arial;">
    <h2>Yellow Line Debug</h2>
    <img src="/stream.mjpg" style="width:95%; max-width:720px; border:2px solid white;">
    <p>Put yellow line in front of camera. No motor control in this test.</p>
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
#    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.copy()
    h, w, _ = frame.shape

    roi_top = int(h * 0.55)
    roi = frame[roi_top:h, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    moments = cv2.moments(mask)

    cx = None

    if moments["m00"] > 500:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        cv2.circle(roi, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(roi, (w // 2, 0), (w // 2, roi.shape[0]), (255, 0, 0), 2)
        cv2.line(roi, (cx, 0), (cx, roi.shape[0]), (0, 0, 255), 2)

        error = cx - (w // 2)
        cv2.putText(frame, f"cx={cx} error={error}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "NO YELLOW LINE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    print()
    print("Open:")
    print("http://RASPBERRY_PI_IP:8000")
    print()
    print("This test only detects yellow line. Motors are disabled.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("Stopped")


if __name__ == "__main__":
    main()
