import io
import time
import threading
import urllib.parse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from libcamera import Transform
from picamera2 import Picamera2

from piracer.vehicles import PiRacerStandard
from piracer.gamepads import ShanWanGamepad

from data_recorder import DataRecorder


PORT = 8000

CAM_W = 480
CAM_H = 360
CAM_FPS = 30

WEB_THROTTLE = 0.5
WEB_STEERING = -0.80

USE_REMOTE = True

REMOTE_THROTTLE_SCALE = 0.5
REMOTE_STEERING_SCALE = 0.80
REMOTE_DEADZONE = 0.15
REMOTE_TIMEOUT = 0.3
REMOTE_HZ = 50

REMOTE_THROTTLE_INVERT = False
REMOTE_STEERING_INVERT = True

CONTROL_HZ = 50
COMMAND_TIMEOUT = 0.8

THROTTLE_STEP = 0.08
STEERING_STEP = 0.25

STREAM_JPEG_QUALITY = 70

RECORD_ONLY_WHEN_MOVING = True
RECORD_MIN_THROTTLE = 0.03

car = None
picam2 = None
running = True


recorder = DataRecorder(
    base_dir="dataset",
    source="fpv",
    jpeg_quality=85,
    save_every_n_frames=3,
    max_queue_size=300,
    frame_color="BGR"
)


class StreamBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def set_frame(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()


stream_buffer = StreamBuffer()

mode_lock = threading.Lock()
control_mode = "web"

web_lock = threading.Lock()
web_target_throttle = 0.0
web_target_steering = 0.0
web_last_time = 0.0

remote_lock = threading.Lock()
remote_throttle = 0.0
remote_steering = 0.0
remote_last_time = 0.0
remote_connected = False

state_lock = threading.Lock()
state_source = "idle"
state_throttle = 0.0
state_steering = 0.0
state_remote = "off"
state_mode = "web"


def clamp(x):
    return max(-1.0, min(1.0, x))


def deadzone(x, dz):
    if abs(x) < dz:
        return 0.0
    return x


def move_towards(current, target, step):
    if current < target:
        return min(current + step, target)
    if current > target:
        return max(current - step, target)
    return current


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Donkey Car FPV Recorder</title>
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
            max-width: 640px;
            border: 2px solid white;
        }

        button {
            font-size: 20px;
            padding: 12px 20px;
            margin: 8px;
            border: none;
            border-radius: 8px;
            background: #333;
            color: white;
        }

        button.active {
            background: #0a84ff;
        }

        .record {
            background: #005bbb;
        }

        .record_stop {
            background: #444;
        }

        .info {
            color: #ccc;
            font-size: 15px;
        }
    </style>
</head>

<body>
    <h2>Donkey Car FPV + Dataset Recorder</h2>

    <img src="/stream.mjpg">

    <h3>Control mode</h3>

    <button id="btn_web" onclick="setMode('web')">WEB / WASD</button>
    <button id="btn_remote" onclick="setMode('remote')">REMOTE / GAMEPAD</button>

    <br>

    <button class="record" onclick="setRecord(1)">START RECORDING</button>
    <button class="record_stop" onclick="setRecord(0)">STOP RECORDING</button>

    <h3 id="keys">Keys: none</h3>
    <h3 id="state">State: loading...</h3>
    <h3 id="record_state">Recording: loading...</h3>

    <p class="info">
        WEB mode: W/S/A/D + Space<br>
        REMOTE mode: only gamepad controls the car<br>
        Recording saves clean camera frames in the same dataset format as OpenCV autopilot.
    </p>

<script>
let controlMode = "web";

let w = false;
let a = false;
let s = false;
let d = false;
let space = false;

let requestInFlight = false;

function updateModeButtons() {
    document.getElementById("btn_web").classList.toggle("active", controlMode === "web");
    document.getElementById("btn_remote").classList.toggle("active", controlMode === "remote");
}

function clearKeys() {
    w = false;
    a = false;
    s = false;
    d = false;
    space = false;
    updateKeysText();
}

function setMode(mode) {
    controlMode = mode;
    updateModeButtons();

    clearKeys();

    fetch("/mode?mode=" + mode, {cache: "no-store"})
        .catch(() => {});
}

function setRecord(value) {
    fetch("/record?enabled=" + value, {cache: "no-store"})
        .catch(() => {});
}

function calc() {
    let th = 0;
    let st = 0;

    if (!space) {
        if (w && !s) th = 1;
        if (s && !w) th = -1;

        if (a && !d) st = -1;
        if (d && !a) st = 1;
    }

    return {th: th, st: st};
}

function updateKeysText() {
    document.getElementById("keys").innerText =
        "Mode=" + controlMode +
        " | W=" + w +
        " A=" + a +
        " S=" + s +
        " D=" + d +
        " SPACE=" + space;
}

function sendControl() {
    updateKeysText();

    if (controlMode !== "web") {
        return;
    }

    if (requestInFlight) {
        return;
    }

    const c = calc();

    requestInFlight = true;

    fetch("/control?th=" + c.th + "&st=" + c.st, {cache: "no-store"})
        .catch(() => {})
        .finally(() => {
            requestInFlight = false;
        });
}

document.addEventListener("keydown", function(e) {
    if (e.code === "KeyW") { w = true; e.preventDefault(); }
    if (e.code === "KeyA") { a = true; e.preventDefault(); }
    if (e.code === "KeyS") { s = true; e.preventDefault(); }
    if (e.code === "KeyD") { d = true; e.preventDefault(); }
    if (e.code === "Space") { space = true; e.preventDefault(); }

    sendControl();
});

document.addEventListener("keyup", function(e) {
    if (e.code === "KeyW") { w = false; e.preventDefault(); }
    if (e.code === "KeyA") { a = false; e.preventDefault(); }
    if (e.code === "KeyS") { s = false; e.preventDefault(); }
    if (e.code === "KeyD") { d = false; e.preventDefault(); }
    if (e.code === "Space") { space = false; e.preventDefault(); }

    sendControl();
});

window.addEventListener("blur", function() {
    clearKeys();
    sendControl();
});

setInterval(function() {
    if (controlMode === "web") {
        sendControl();
    }
}, 70);

setInterval(function() {
    fetch("/state", {cache: "no-store"})
        .then(r => r.json())
        .then(data => {
            controlMode = data.mode;
            updateModeButtons();

            document.getElementById("state").innerText =
                "Mode: " + data.mode +
                " | Source: " + data.source +
                " | throttle: " + data.throttle.toFixed(2) +
                " | steering: " + data.steering.toFixed(2) +
                " | remote: " + data.remote;

            document.getElementById("record_state").innerText =
                "REC: " + data.recording +
                " | written: " + data.written_frames +
                " | queued: " + data.queued_frames +
                " | dropped: " + data.dropped_frames +
                " | queue: " + data.queue_size +
                " | session: " + data.session_dir;
        })
        .catch(() => {
            document.getElementById("state").innerText = "State: disconnected";
            document.getElementById("record_state").innerText = "Recording: disconnected";
        });
}, 700);

updateModeButtons();
updateKeysText();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global control_mode
        global web_target_throttle, web_target_steering, web_last_time

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

        if path == "/mode":
            mode = query.get("mode", ["web"])[0]

            if mode not in ["web", "remote"]:
                mode = "web"

            with mode_lock:
                control_mode = mode

            with web_lock:
                web_target_throttle = 0.0
                web_target_steering = 0.0
                web_last_time = time.time()

            self.send_response(204)
            self.end_headers()
            return

        if path == "/control":
            with mode_lock:
                mode = control_mode

            if mode != "web":
                self.send_response(204)
                self.end_headers()
                return

            th_raw = clamp(float(query.get("th", ["0"])[0]))
            st_raw = clamp(float(query.get("st", ["0"])[0]))

            with web_lock:
                web_target_throttle = th_raw * WEB_THROTTLE
                web_target_steering = st_raw * WEB_STEERING
                web_last_time = time.time()

            self.send_response(204)
            self.end_headers()
            return

        if path == "/record":
            enabled = query.get("enabled", ["0"])[0] == "1"

            if enabled:
                recorder.start(meta={
                    "source": "fpv",
                    "camera_width": CAM_W,
                    "camera_height": CAM_H,
                    "camera_fps": CAM_FPS,
                    "frame_color": "BGR",
                    "record_only_when_moving": RECORD_ONLY_WHEN_MOVING,
                    "record_min_throttle": RECORD_MIN_THROTTLE,

                    "web_throttle": WEB_THROTTLE,
                    "web_steering": WEB_STEERING,

                    "use_remote": USE_REMOTE,
                    "remote_throttle_scale": REMOTE_THROTTLE_SCALE,
                    "remote_steering_scale": REMOTE_STEERING_SCALE,
                    "remote_deadzone": REMOTE_DEADZONE,
                    "remote_throttle_invert": REMOTE_THROTTLE_INVERT,
                    "remote_steering_invert": REMOTE_STEERING_INVERT,

                    "control_hz": CONTROL_HZ,
                    "command_timeout": COMMAND_TIMEOUT,
                    "throttle_step": THROTTLE_STEP,
                    "steering_step": STEERING_STEP
                })
            else:
                recorder.stop()

            self.send_response(204)
            self.end_headers()
            return

        if path == "/state":
            with state_lock:
                obj = {
                    "source": state_source,
                    "throttle": state_throttle,
                    "steering": state_steering,
                    "remote": state_remote,
                    "mode": state_mode,
                }

            obj.update(recorder.get_stats())

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


def remote_loop():
    global remote_throttle, remote_steering, remote_last_time, remote_connected

    try:
        gamepad = ShanWanGamepad()
        print("Remote: connected")
    except Exception as e:
        print("Remote: not found:", e)
        with remote_lock:
            remote_connected = False
        return

    with remote_lock:
        remote_connected = True

    last_error_print = 0.0

    while running:
        try:
            data = gamepad.read_data()

            st = data.analog_stick_left.x
            th = data.analog_stick_right.y

            if REMOTE_STEERING_INVERT:
                st = -st

            if REMOTE_THROTTLE_INVERT:
                th = -th

            st = deadzone(st, REMOTE_DEADZONE)
            th = deadzone(th, REMOTE_DEADZONE)

            st = clamp(st) * REMOTE_STEERING_SCALE
            th = clamp(th) * REMOTE_THROTTLE_SCALE

            with remote_lock:
                remote_throttle = th
                remote_steering = st
                remote_last_time = time.time()
                remote_connected = True

        except Exception as e:
            now = time.time()
            if now - last_error_print > 2.0:
                print("Remote read error:", e)
                last_error_print = now

            with remote_lock:
                remote_connected = False

        time.sleep(1.0 / REMOTE_HZ)


def read_web():
    with web_lock:
        th = web_target_throttle
        st = web_target_steering
        last = web_last_time

    if time.time() - last > COMMAND_TIMEOUT:
        return 0.0, 0.0, False

    active = abs(th) > 0.01 or abs(st) > 0.01

    return th, st, active


def read_remote():
    with remote_lock:
        th = remote_throttle
        st = remote_steering
        last = remote_last_time
        connected = remote_connected

    if not connected:
        return 0.0, 0.0, False, "off"

    active = abs(th) > 0.01 or abs(st) > 0.01

    if last == 0.0:
        status = "waiting"
    else:
        age = time.time() - last

        if age < 1.0:
            status = "ok"
        else:
            status = "hold"

    return th, st, active, status


def control_loop():
    global state_source, state_throttle, state_steering, state_remote, state_mode

    actual_throttle = 0.0
    actual_steering = 0.0

    while running:
        with mode_lock:
            mode = control_mode

        web_th, web_st, web_active = read_web()
        rem_th, rem_st, rem_active, rem_status = read_remote()

        if mode == "web":
            if web_active:
                target_throttle = web_th
                target_steering = web_st
                source = "web"
            else:
                target_throttle = 0.0
                target_steering = 0.0
                source = "web_idle"

        elif mode == "remote":
            if rem_active:
                target_throttle = rem_th
                target_steering = rem_st
                source = "remote"
            else:
                target_throttle = 0.0
                target_steering = 0.0
                source = "remote_idle"

        else:
            target_throttle = 0.0
            target_steering = 0.0
            source = "idle"

        if mode == "remote":
            actual_throttle = target_throttle
            actual_steering = target_steering
        else:
            actual_throttle = move_towards(actual_throttle, target_throttle, THROTTLE_STEP)
            actual_steering = move_towards(actual_steering, target_steering, STEERING_STEP)

        car.set_throttle_percent(actual_throttle)
        car.set_steering_percent(actual_steering)

        with state_lock:
            state_source = source
            state_throttle = actual_throttle
            state_steering = actual_steering
            state_remote = rem_status
            state_mode = mode

        time.sleep(1.0 / CONTROL_HZ)


def camera_loop():
    global picam2

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

    while running:
        try:
            frame = picam2.capture_array()

            with state_lock:
                th = float(state_throttle)
                st = float(state_steering)
                src = str(state_source)
                mode = str(state_mode)
                remote_status = str(state_remote)

            moving = abs(th) > RECORD_MIN_THROTTLE

            should_record = recorder.is_recording()

            if RECORD_ONLY_WHEN_MOVING:
                should_record = should_record and moving

            if should_record:
                recorder.record(
                    frame,
                    steering=st,
                    throttle=th,
                    metadata={
                        "can_drive": int(moving),
                        "auto_enabled": 0,
                        "mode": f"fpv_{mode}_{src}",

                        "error": "",
                        "error_norm": "",
                        "lane_center": "",
                        "left_x": "",
                        "right_x": "",
                        "lane_width": "",
                        "lane_locked": "",

                        "visible_side": "",
                        "missing_side": "",
                        "contours": ""
                    }
                )

            stream_frame = frame.copy()

            rec_stats = recorder.get_stats()

            cv2.putText(
                stream_frame,
                f"mode={mode} src={src} th={th:.2f} st={st:.2f} remote={remote_status}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                stream_frame,
                f"REC={rec_stats['recording']} written={rec_stats['written_frames']} dropped={rec_stats['dropped_frames']}",
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
            print("Camera loop error:", e)
            time.sleep(0.2)

        time.sleep(1.0 / CAM_FPS)


def main():
    global car
    global running

    print("Starting car...")
    car = PiRacerStandard()

    car.set_throttle_percent(0.0)
    car.set_steering_percent(0.0)

    print("Starting camera thread...")
    t_camera = threading.Thread(target=camera_loop, daemon=True)
    t_camera.start()

    if USE_REMOTE:
        print("Starting remote thread...")
        t_remote = threading.Thread(target=remote_loop, daemon=True)
        t_remote.start()

    print("Starting control loop...")
    t_control = threading.Thread(target=control_loop, daemon=True)
    t_control.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    print()
    print("Open:")
    print("http://RASPBERRY_PI_IP:8000")
    print()
    print("Default mode: WEB")
    print("Use buttons on website to switch WEB / REMOTE")
    print("Recording format is compatible with OpenCV autopilot dataset.")
    print("Saved images use the same BGR recording path as OpenCV autopilot.")
    print("Dataset path: dataset/session_..._fpv/")
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        running = False

        try:
            recorder.close()
        except Exception:
            pass

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
