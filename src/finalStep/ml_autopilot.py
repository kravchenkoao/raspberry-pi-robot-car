import io
import time
import json
import threading
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from libcamera import Transform
from picamera2 import Picamera2

from piracer.vehicles import PiRacerStandard


try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


PORT = 8000

CAM_W = 480
CAM_H = 360
CAM_FPS = 20

MODEL_PATH = Path("models/manual_model.tflite")
CONFIG_PATH = Path("models/manual_model_preprocess_config.json")

FIXED_THROTTLE = 0.35

STEERING_SIGN = 1.0
STEERING_GAIN = 1.0
MAX_STEERING = 1.0

SMOOTH_STEERING = True
STEERING_SMOOTHING = 0.65

USE_MODEL_THROTTLE = False
MODEL_THROTTLE_GAIN = 1.0
MAX_THROTTLE = 0.25
MIN_THROTTLE = 0.08

STREAM_JPEG_QUALITY = 70

car = None
picam2 = None
running = True

interpreter = None
input_details = None
output_details = None

preprocess_config = None

state_lock = threading.Lock()
state_auto = False
state_model_loaded = False
state_prediction_ok = False
state_steering_raw = 0.0
state_steering = 0.0
state_throttle = 0.0
state_prediction_ms = 0.0
state_fps = 0.0
state_error = ""


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
    <title>ML Autopilot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            background:#111;
            color:white;
            text-align:center;
            font-family:Arial;
        }

        img {
            width:95%;
            max-width:900px;
            border:2px solid white;
        }

        button {
            font-size:22px;
            padding:14px 24px;
            margin:10px;
            border:none;
            border-radius:10px;
            color:white;
        }

        #start {
            background:#008000;
        }

        #stop {
            background:#8b0000;
        }

        .info {
            color:#ccc;
            font-size:15px;
        }
    </style>
</head>

<body>
    <h2>ML Autopilot</h2>

    <img src="/stream.mjpg">

    <h3 id="state">State: loading...</h3>

    <button id="start" onclick="setAuto(1)">START ML AUTOPILOT</button>
    <button id="stop" onclick="setAuto(0)">STOP</button>

    <p class="info">
        Start only on low speed and be ready to stop the car.<br>
        If steering direction is wrong, change STEERING_SIGN in Python code.
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
                " | model: " + data.model_loaded +
                " | pred_ok: " + data.prediction_ok +
                " | steering_raw: " + data.steering_raw.toFixed(3) +
                " | steering: " + data.steering.toFixed(3) +
                " | throttle: " + data.throttle.toFixed(3) +
                " | pred_ms: " + data.prediction_ms.toFixed(1) +
                " | fps: " + data.fps.toFixed(1) +
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
                    "model_loaded": state_model_loaded,
                    "prediction_ok": state_prediction_ok,
                    "steering_raw": state_steering_raw,
                    "steering": state_steering,
                    "throttle": state_throttle,
                    "prediction_ms": state_prediction_ms,
                    "fps": state_fps,
                    "error": state_error,
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

                    self.wfile.write(b"--FRAME\\r\\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\\r\\n")

            except Exception:
                pass

            return

        self.send_error(404)


def clamp(x, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, x))


def load_model_and_config():
    global interpreter
    global input_details
    global output_details
    global preprocess_config
    global state_model_loaded
    global state_error

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        preprocess_config = json.load(f)

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with state_lock:
        state_model_loaded = True
        state_error = ""

    print("Model loaded:", MODEL_PATH)
    print("Config loaded:", CONFIG_PATH)
    print("Input details:", input_details)
    print("Output details:", output_details)


def preprocess_frame(frame_bgr):
    img_w = int(preprocess_config["img_w"])
    img_h = int(preprocess_config["img_h"])

    crop_top = float(preprocess_config["crop_top_percent"])
    crop_bottom = float(preprocess_config["crop_bottom_percent"])

    h, w, _ = frame_bgr.shape

    y1 = int(h * crop_top)
    y2 = int(h * crop_bottom)

    frame_bgr = frame_bgr[y1:y2, 0:w]
    frame_bgr = cv2.resize(frame_bgr, (img_w, img_h))

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0

    x = np.expand_dims(frame_rgb, axis=0)

    return x


def run_model(frame_bgr):
    x = preprocess_frame(frame_bgr)

    inp = input_details[0]
    out = output_details[0]

    input_index = inp["index"]
    output_index = out["index"]

    input_dtype = inp["dtype"]

    if input_dtype == np.float32:
        model_input = x.astype(np.float32)
    elif input_dtype == np.uint8:
        scale, zero_point = inp["quantization"]
        model_input = x / scale + zero_point
        model_input = np.clip(model_input, 0, 255).astype(np.uint8)
    elif input_dtype == np.int8:
        scale, zero_point = inp["quantization"]
        model_input = x / scale + zero_point
        model_input = np.clip(model_input, -128, 127).astype(np.int8)
    else:
        model_input = x.astype(input_dtype)

    interpreter.set_tensor(input_index, model_input)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)[0]

    output_dtype = out["dtype"]

    if output_dtype in [np.uint8, np.int8]:
        scale, zero_point = out["quantization"]
        pred = (pred.astype(np.float32) - zero_point) * scale

    steering_raw = float(pred[0])

    if len(pred) >= 2 and USE_MODEL_THROTTLE:
        throttle = float(pred[1]) * MODEL_THROTTLE_GAIN
        throttle = abs(throttle)
        throttle = clamp(throttle, MIN_THROTTLE, MAX_THROTTLE)
    else:
        throttle = FIXED_THROTTLE

    return steering_raw, throttle


def camera_loop():
    global picam2
    global state_prediction_ok
    global state_steering_raw
    global state_steering
    global state_throttle
    global state_prediction_ms
    global state_fps
    global state_error

    print("Starting camera loop...")

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"},
        controls={"FrameRate": CAM_FPS},
        transform=Transform(hflip=1, vflip=1)
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(1.0)

    smoothed_steering = 0.0
    last_frame_time = time.time()

    while running:
        try:
            frame = picam2.capture_array()

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            fps = 1.0 / dt if dt > 0 else 0.0

            with state_lock:
                auto_enabled = state_auto

            steering_raw = 0.0
            steering = 0.0
            throttle = 0.0
            pred_ms = 0.0
            prediction_ok = False
            err = ""

            if auto_enabled and state_model_loaded:
                t0 = time.time()

                steering_raw, throttle = run_model(frame)

                pred_ms = (time.time() - t0) * 1000.0

                steering = STEERING_SIGN * STEERING_GAIN * steering_raw
                steering = clamp(steering, -MAX_STEERING, MAX_STEERING)

                if SMOOTH_STEERING:
                    smoothed_steering = (
                        STEERING_SMOOTHING * smoothed_steering +
                        (1.0 - STEERING_SMOOTHING) * steering
                    )
                    steering = smoothed_steering
                else:
                    smoothed_steering = steering

                car.set_steering_percent(steering)
                car.set_throttle_percent(throttle)

                prediction_ok = True

            else:
                smoothed_steering = 0.0
                car.set_steering_percent(0.0)
                car.set_throttle_percent(0.0)

            with state_lock:
                state_prediction_ok = prediction_ok
                state_steering_raw = float(steering_raw)
                state_steering = float(steering)
                state_throttle = float(throttle)
                state_prediction_ms = float(pred_ms)
                state_fps = float(fps)
                state_error = err

            stream_frame = frame.copy()

            crop_top = float(preprocess_config["crop_top_percent"])
            crop_bottom = float(preprocess_config["crop_bottom_percent"])

            h, w, _ = stream_frame.shape
            y1 = int(h * crop_top)
            y2 = int(h * crop_bottom)

            cv2.line(stream_frame, (0, y1), (w, y1), (0, 255, 255), 2)
            cv2.line(stream_frame, (0, y2), (w, y2), (0, 255, 255), 2)

            cv2.putText(
                stream_frame,
                f"ML AUTO={auto_enabled} pred_ok={prediction_ok}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                stream_frame,
                f"raw={steering_raw:.3f} steering={steering:.3f} throttle={throttle:.3f} pred={pred_ms:.1f}ms fps={fps:.1f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            ok, jpg = cv2.imencode(
                ".jpg",
                stream_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
            )

            if ok:
                stream_buffer.set_frame(jpg.tobytes())

        except Exception as e:
            err = str(e)
            print("Camera loop error:", err)

            with state_lock:
                state_prediction_ok = False
                state_error = err

            if car is not None:
                car.set_throttle_percent(0.0)
                car.set_steering_percent(0.0)

            time.sleep(0.2)

        time.sleep(1.0 / CAM_FPS)


def main():
    global car
    global running

    print("Loading model...")
    load_model_and_config()

    print("Starting PiRacer...")
    car = PiRacerStandard()
    car.set_throttle_percent(0.0)
    car.set_steering_percent(0.0)

    print("Starting camera...")
    t_camera = threading.Thread(target=camera_loop, daemon=True)
    t_camera.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    print()
    print("Open:")
    print("http://RASPBERRY_PI_IP:8000")
    print()
    print("Model:", MODEL_PATH)
    print("Config:", CONFIG_PATH)
    print("Fixed throttle:", FIXED_THROTTLE)
    print("Steering sign:", STEERING_SIGN)
    print("Steering gain:", STEERING_GAIN)
    print()
    print("IMPORTANT:")
    print("Start with low throttle.")
    print("If the car turns in the wrong direction, change STEERING_SIGN from 1.0 to -1.0.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        running = False

        if car is not None:
            car.set_throttle_percent(0.0)
            car.set_steering_percent(0.0)

        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass

        try:
            server.server_close()
        except Exception:
            pass

        print("Stopped safely.")


if __name__ == "__main__":
    main()
