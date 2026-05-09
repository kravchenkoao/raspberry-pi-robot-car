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

from data_recorder import DataRecorder


PORT = 8000

CAM_W = 480
CAM_H = 360
CAM_FPS = 20

LOWER_ORANGE = np.array([0, 70, 70])
UPPER_ORANGE = np.array([25, 255, 255])

MIN_AREA = 500

ROI_TOP_PERCENT = 0.55
ROI_BOTTOM_PERCENT = 0.85

KP = 2.50
MAX_STEERING = 1.00
STEERING_SIGN = -1

THROTTLE_STRAIGHT = 0.50
THROTTLE_TURN = 0.47
THROTTLE_SHARP_TURN = 0.45

DEFAULT_LANE_WIDTH = int(CAM_W * 0.55)
MIN_LANE_WIDTH = int(CAM_W * 0.25)
MAX_LANE_WIDTH = int(CAM_W * 0.95)
LANE_WIDTH_SMOOTHING = 0.80

LOCKED_MATCH_MAX_DISTANCE = int(CAM_W * 0.30)
LOCKED_MATCH_RECOVERY_DISTANCE = int(CAM_W * 0.45)

RECORD_ONLY_WHEN_CAN_DRIVE = True

car = None

recorder = DataRecorder(
    base_dir="dataset",
    source="opencv",
    jpeg_quality=85,
    save_every_n_frames=2,
    max_queue_size=300,
    frame_color="BGR"
)

last_lane_width = DEFAULT_LANE_WIDTH

track_lock = threading.Lock()
lane_locked = False
tracked_left_x = 0
tracked_right_x = 0
single_visible_side = None
missing_lane_side = "none"


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

state_auto = False
state_mode = "none"
state_contours = 0
state_left_x = 0
state_right_x = 0
state_lane_center = 0
state_lane_width = DEFAULT_LANE_WIDTH
state_lane_locked = False
state_visible_side = "none"
state_missing_side = "none"
state_error = 0
state_error_norm = 0.0
state_steering = 0.0
state_throttle = 0.0
state_can_drive = False


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Lane Autopilot Recorder</title>
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
            max-width:1000px;
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

        #record_start {
            background:#005bbb;
        }

        #record_stop {
            background:#444;
        }

        .info {
            color:#ccc;
            font-size:15px;
        }
    </style>
</head>

<body>
    <h2>Orange Lane Autopilot + Dataset Recorder</h2>

    <img src="/stream.mjpg">

    <h3 id="state">State: loading...</h3>
    <h3 id="record_state">Recording: loading...</h3>

    <button id="start" onclick="setAuto(1)">START AUTOPILOT</button>
    <button id="stop" onclick="setAuto(0)">STOP AUTOPILOT</button>

    <br>

    <button id="record_start" onclick="setRecord(1)">START RECORDING</button>
    <button id="record_stop" onclick="setRecord(0)">STOP RECORDING</button>

    <p class="info">
        Press START AUTOPILOT only when both orange lines are visible.<br>
        Recording saves clean camera frames, not the debug stream with mask.<br>
        By default, frames are recorded only when the autopilot can drive.
    </p>

<script>
function setAuto(value) {
    fetch("/auto?enabled=" + value, {cache: "no-store"}).catch(() => {});
}

function setRecord(value) {
    fetch("/record?enabled=" + value, {cache: "no-store"}).catch(() => {});
}

setInterval(function() {
    fetch("/state", {cache: "no-store"})
        .then(r => r.json())
        .then(data => {
            document.getElementById("state").innerText =
                "AUTO: " + data.auto +
                " | can_drive: " + data.can_drive +
                " | locked: " + data.lane_locked +
                " | mode: " + data.mode +
                " | visible: " + data.visible_side +
                " | missing: " + data.missing_side +
                " | contours: " + data.contours +
                " | width: " + data.lane_width +
                " | center: " + data.lane_center +
                " | error: " + data.error +
                " | throttle: " + data.throttle.toFixed(2) +
                " | steering: " + data.steering.toFixed(2);

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

            reset_lane_tracking()

            with state_lock:
                state_auto = enabled

            if not enabled and car is not None:
                car.set_throttle_percent(0.0)
                car.set_steering_percent(0.0)

            self.send_response(204)
            self.end_headers()
            return

        if path == "/record":
            enabled = query.get("enabled", ["0"])[0] == "1"

            if enabled:
                recorder.start(meta={
                    "source": "opencv",
                    "camera_width": CAM_W,
                    "camera_height": CAM_H,
                    "camera_fps": CAM_FPS,
                    "frame_color": "RGB",
                    "record_only_when_can_drive": RECORD_ONLY_WHEN_CAN_DRIVE,

                    "lower_orange": LOWER_ORANGE.tolist(),
                    "upper_orange": UPPER_ORANGE.tolist(),
                    "min_area": MIN_AREA,

                    "roi_top_percent": ROI_TOP_PERCENT,
                    "roi_bottom_percent": ROI_BOTTOM_PERCENT,

                    "kp": KP,
                    "max_steering": MAX_STEERING,
                    "steering_sign": STEERING_SIGN,

                    "throttle_straight": THROTTLE_STRAIGHT,
                    "throttle_turn": THROTTLE_TURN,
                    "throttle_sharp_turn": THROTTLE_SHARP_TURN,

                    "default_lane_width": DEFAULT_LANE_WIDTH,
                    "min_lane_width": MIN_LANE_WIDTH,
                    "max_lane_width": MAX_LANE_WIDTH,
                    "lane_width_smoothing": LANE_WIDTH_SMOOTHING,

                    "locked_match_max_distance": LOCKED_MATCH_MAX_DISTANCE,
                    "locked_match_recovery_distance": LOCKED_MATCH_RECOVERY_DISTANCE
                })
            else:
                recorder.stop()

            self.send_response(204)
            self.end_headers()
            return

        if path == "/state":
            with state_lock:
                obj = {
                    "auto": state_auto,
                    "mode": state_mode,
                    "contours": state_contours,
                    "left_x": state_left_x,
                    "right_x": state_right_x,
                    "lane_center": state_lane_center,
                    "lane_width": state_lane_width,
                    "lane_locked": state_lane_locked,
                    "visible_side": state_visible_side,
                    "missing_side": state_missing_side,
                    "error": state_error,
                    "error_norm": state_error_norm,
                    "steering": state_steering,
                    "throttle": state_throttle,
                    "can_drive": state_can_drive,
                }

            rec_stats = recorder.get_stats()
            obj.update(rec_stats)

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


def clamp_int(x, min_value, max_value):
    return int(max(min_value, min(max_value, x)))


def reset_lane_tracking():
    global last_lane_width
    global lane_locked
    global tracked_left_x
    global tracked_right_x
    global single_visible_side
    global missing_lane_side

    with track_lock:
        last_lane_width = DEFAULT_LANE_WIDTH
        lane_locked = False
        tracked_left_x = 0
        tracked_right_x = 0
        single_visible_side = None
        missing_lane_side = "none"


def draw_contour(frame, item, roi_top, color):
    x = item["x"]
    y = item["y"]
    cw = item["w"]
    ch = item["h"]
    cx = item["cx"]
    cy = item["cy"]

    y_full = y + roi_top
    cy_full = cy + roi_top

    cv2.rectangle(frame, (x, y_full), (x + cw, y_full + ch), color, 2)
    cv2.circle(frame, (cx, cy_full), 7, color, -1)


def find_closest_to_x(items, target_x, exclude_item=None, max_distance=None):
    best = None
    best_dist = None

    for item in items:
        if exclude_item is not None and item is exclude_item:
            continue

        dist = abs(item["cx"] - target_x)

        if max_distance is not None and dist > max_distance:
            continue

        if best is None or dist < best_dist:
            best = item
            best_dist = dist

    return best, best_dist


def is_valid_lane_width(left_x, right_x):
    width = right_x - left_x
    return MIN_LANE_WIDTH <= width <= MAX_LANE_WIDTH


def process_frame(frame):
    global last_lane_width
    global lane_locked
    global tracked_left_x
    global tracked_right_x
    global single_visible_side
    global missing_lane_side

    global state_mode
    global state_contours
    global state_left_x
    global state_right_x
    global state_lane_center
    global state_lane_width
    global state_lane_locked
    global state_visible_side
    global state_missing_side
    global state_error
    global state_error_norm
    global state_steering
    global state_throttle
    global state_can_drive

    raw_frame = frame.copy()
    frame = raw_frame.copy()

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

    camera_center = w // 2

    left_x = 0
    right_x = 0
    lane_center = camera_center
    error = 0
    error_norm = 0.0
    steering = 0.0
    throttle = 0.0
    mode = "no_line"

    visible_side = "none"
    current_missing_side = "none"
    current_lane_locked = False

    cv2.line(frame, (0, roi_top), (w, roi_top), (0, 255, 255), 2)
    cv2.line(frame, (0, roi_bottom), (w, roi_bottom), (0, 255, 255), 2)
    cv2.line(frame, (camera_center, roi_top), (camera_center, roi_bottom), (255, 0, 0), 2)

    with state_lock:
        auto_enabled = state_auto

    selected = good_contours[:2]
    selected = sorted(selected, key=lambda item: item["cx"])

    two_lines_valid_for_start = False

    if len(selected) >= 2:
        start_left = selected[0]
        start_right = selected[1]

        if is_valid_lane_width(start_left["cx"], start_right["cx"]):
            two_lines_valid_for_start = True

    if not auto_enabled:
        if two_lines_valid_for_start:
            left = selected[0]
            right = selected[1]

            left_x = left["cx"]
            right_x = right["cx"]
            lane_center = (left_x + right_x) // 2
            mode = "preview_two_lines"

            draw_contour(frame, left, roi_top, (0, 255, 0))
            draw_contour(frame, right, roi_top, (0, 255, 0))

        elif len(good_contours) >= 1:
            one = good_contours[0]
            lane_center = one["cx"]
            mode = "preview_one_line"
            draw_contour(frame, one, roi_top, (0, 255, 0))

        else:
            lane_center = camera_center
            mode = "preview_no_line"

    else:
        with track_lock:
            if not lane_locked:
                if two_lines_valid_for_start:
                    left = selected[0]
                    right = selected[1]

                    left_x = left["cx"]
                    right_x = right["cx"]

                    last_lane_width = right_x - left_x

                    tracked_left_x = left_x
                    tracked_right_x = right_x

                    lane_locked = True
                    current_lane_locked = True

                    single_visible_side = None
                    missing_lane_side = "none"

                    lane_center = (left_x + right_x) // 2

                    visible_side = "both"
                    current_missing_side = "none"
                    mode = "lock_captured_two_lines"

                    draw_contour(frame, left, roi_top, (0, 255, 0))
                    draw_contour(frame, right, roi_top, (0, 255, 0))

                else:
                    lane_center = camera_center
                    mode = "waiting_two_lines_for_lock"

                    visible_side = "none"
                    current_missing_side = "none"
                    current_lane_locked = False

                    if len(good_contours) >= 1:
                        draw_contour(frame, good_contours[0], roi_top, (0, 165, 255))

            else:
                current_lane_locked = True

                left_candidate = None
                right_candidate = None
                left_dist = None
                right_dist = None

                if single_visible_side == "left":
                    left_candidate, left_dist = find_closest_to_x(
                        good_contours,
                        tracked_left_x,
                        max_distance=LOCKED_MATCH_RECOVERY_DISTANCE
                    )

                    if left_candidate is not None:
                        predicted_right_x = left_candidate["cx"] + last_lane_width

                        right_candidate, right_dist = find_closest_to_x(
                            good_contours,
                            predicted_right_x,
                            exclude_item=left_candidate,
                            max_distance=LOCKED_MATCH_MAX_DISTANCE
                        )

                        if right_candidate is not None:
                            if not is_valid_lane_width(left_candidate["cx"], right_candidate["cx"]):
                                right_candidate = None

                elif single_visible_side == "right":
                    right_candidate, right_dist = find_closest_to_x(
                        good_contours,
                        tracked_right_x,
                        max_distance=LOCKED_MATCH_RECOVERY_DISTANCE
                    )

                    if right_candidate is not None:
                        predicted_left_x = right_candidate["cx"] - last_lane_width

                        left_candidate, left_dist = find_closest_to_x(
                            good_contours,
                            predicted_left_x,
                            exclude_item=right_candidate,
                            max_distance=LOCKED_MATCH_MAX_DISTANCE
                        )

                        if left_candidate is not None:
                            if not is_valid_lane_width(left_candidate["cx"], right_candidate["cx"]):
                                left_candidate = None

                else:
                    left_candidate, left_dist = find_closest_to_x(
                        good_contours,
                        tracked_left_x,
                        max_distance=LOCKED_MATCH_MAX_DISTANCE
                    )

                    right_candidate, right_dist = find_closest_to_x(
                        good_contours,
                        tracked_right_x,
                        max_distance=LOCKED_MATCH_MAX_DISTANCE
                    )

                    if left_candidate is not None and right_candidate is not None and left_candidate is right_candidate:
                        if left_dist <= right_dist:
                            right_candidate = None
                        else:
                            left_candidate = None

                    if left_candidate is not None and right_candidate is not None:
                        if not is_valid_lane_width(left_candidate["cx"], right_candidate["cx"]):
                            if left_dist <= right_dist:
                                right_candidate = None
                            else:
                                left_candidate = None

                if left_candidate is not None and right_candidate is not None:
                    left_x = left_candidate["cx"]
                    right_x = right_candidate["cx"]

                    current_width = right_x - left_x

                    last_lane_width = int(
                        LANE_WIDTH_SMOOTHING * last_lane_width +
                        (1.0 - LANE_WIDTH_SMOOTHING) * current_width
                    )

                    tracked_left_x = left_x
                    tracked_right_x = right_x

                    lane_center = (left_x + right_x) // 2

                    single_visible_side = None
                    missing_lane_side = "none"

                    visible_side = "both"
                    current_missing_side = "none"
                    mode = "locked_both_hard"

                    draw_contour(frame, left_candidate, roi_top, (0, 255, 0))
                    draw_contour(frame, right_candidate, roi_top, (0, 255, 0))

                elif left_candidate is not None:
                    left_x = left_candidate["cx"]
                    right_x = left_x + last_lane_width

                    tracked_left_x = left_x
                    tracked_right_x = right_x

                    lane_center = left_x + last_lane_width // 2

                    single_visible_side = "left"
                    missing_lane_side = "right"

                    visible_side = "left"
                    current_missing_side = "right"
                    mode = "locked_only_left_hard"

                    draw_contour(frame, left_candidate, roi_top, (0, 255, 0))

                    right_x_draw = clamp_int(right_x, 0, w - 1)
                    cv2.line(frame, (right_x_draw, roi_top), (right_x_draw, roi_bottom), (255, 0, 255), 2)

                elif right_candidate is not None:
                    right_x = right_candidate["cx"]
                    left_x = right_x - last_lane_width

                    tracked_right_x = right_x
                    tracked_left_x = left_x

                    lane_center = right_x - last_lane_width // 2

                    single_visible_side = "right"
                    missing_lane_side = "left"

                    visible_side = "right"
                    current_missing_side = "left"
                    mode = "locked_only_right_hard"

                    draw_contour(frame, right_candidate, roi_top, (0, 255, 0))

                    left_x_draw = clamp_int(left_x, 0, w - 1)
                    cv2.line(frame, (left_x_draw, roi_top), (left_x_draw, roi_bottom), (255, 0, 255), 2)

                else:
                    left_x = tracked_left_x
                    right_x = tracked_right_x
                    lane_center = (tracked_left_x + tracked_right_x) // 2

                    visible_side = "none"
                    current_missing_side = "both"
                    mode = "locked_lost_all_stop"

    lane_center = clamp_int(lane_center, 0, w - 1)

    error = lane_center - camera_center
    error_norm = error / (w / 2)

    can_drive = (
        auto_enabled
        and current_lane_locked
        and mode != "waiting_two_lines_for_lock"
        and mode != "locked_lost_all_stop"
        and mode != "no_line"
    )

    if can_drive:
        steering = STEERING_SIGN * KP * error_norm
        steering = clamp(steering, -MAX_STEERING, MAX_STEERING)

        if abs(error_norm) > 0.55:
            throttle = THROTTLE_SHARP_TURN
        elif abs(error_norm) > 0.30:
            throttle = THROTTLE_TURN
        else:
            throttle = THROTTLE_STRAIGHT
    else:
        steering = 0.0
        throttle = 0.0

    if can_drive:
        car.set_throttle_percent(throttle)
        car.set_steering_percent(steering)
    else:
        car.set_throttle_percent(0.0)
        car.set_steering_percent(0.0)
        throttle = 0.0

    should_record = recorder.is_recording()

    if RECORD_ONLY_WHEN_CAN_DRIVE:
        should_record = should_record and can_drive and throttle > 0.0

    if should_record:
        recorder.record(
            raw_frame,
            steering=steering,
            throttle=throttle,
            metadata={
                "can_drive": int(can_drive),
                "auto_enabled": int(auto_enabled),
                "mode": mode,

                "error": int(error),
                "error_norm": float(error_norm),
                "lane_center": int(lane_center),
                "left_x": int(left_x),
                "right_x": int(right_x),
                "lane_width": int(last_lane_width),
                "lane_locked": int(current_lane_locked),

                "visible_side": visible_side,
                "missing_side": current_missing_side,
                "contours": int(len(good_contours))
            }
        )

    lane_center_draw = clamp_int(lane_center, 0, w - 1)
    left_x_draw = clamp_int(left_x, 0, w - 1)
    right_x_draw = clamp_int(right_x, 0, w - 1)

    cv2.line(frame, (lane_center_draw, roi_top), (lane_center_draw, roi_bottom), (0, 0, 255), 3)
    cv2.circle(frame, (lane_center_draw, roi_top + 25), 8, (0, 0, 255), -1)

    if current_lane_locked:
        cv2.line(frame, (left_x_draw, roi_top), (left_x_draw, roi_bottom), (0, 180, 255), 2)
        cv2.line(frame, (right_x_draw, roi_top), (right_x_draw, roi_bottom), (0, 180, 255), 2)

    for item in good_contours:
        cx = item["cx"]
        cy = item["cy"] + roi_top
        cv2.circle(frame, (cx, cy), 4, (80, 80, 255), -1)

    cv2.putText(
        frame,
        f"{mode} lock={current_lane_locked} vis={visible_side} miss={current_missing_side}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"center={lane_center} width={last_lane_width} err={error} th={throttle:.2f} st={steering:.2f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    rec_stats = recorder.get_stats()

    cv2.putText(
        frame,
        f"REC={rec_stats['recording']} written={rec_stats['written_frames']} dropped={rec_stats['dropped_frames']}",
        (10, 80),
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
        state_left_x = int(left_x)
        state_right_x = int(right_x)
        state_lane_center = int(lane_center)
        state_lane_width = int(last_lane_width)
        state_lane_locked = current_lane_locked
        state_visible_side = visible_side
        state_missing_side = current_missing_side
        state_error = int(error)
        state_error_norm = float(error_norm)
        state_steering = float(steering)
        state_throttle = float(throttle)
        state_can_drive = bool(can_drive)

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
    print("Autopilot starts only after pressing START AUTOPILOT.")
    print("IMPORTANT: press START only when both lines are visible.")
    print("Recording saves clean camera frames to dataset/session_*/images/")
    print("CSV path: dataset/session_*/data.csv")
    print("ROI:", ROI_TOP_PERCENT, "-", ROI_BOTTOM_PERCENT)
    print("KP:", KP)
    print("Default lane width:", DEFAULT_LANE_WIDTH)
    print("Locked match max distance:", LOCKED_MATCH_MAX_DISTANCE)
    print("Straight throttle:", THROTTLE_STRAIGHT)
    print("Turn throttle:", THROTTLE_TURN)
    print("Sharp turn throttle:", THROTTLE_SHARP_TURN)
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        try:
            recorder.close()
        except Exception:
            pass

        if car is not None:
            car.set_throttle_percent(0.0)
            car.set_steering_percent(0.0)

        try:
            server.server_close()
        except Exception:
            pass

        print("Stopped safely.")


if __name__ == "__main__":
    main()
