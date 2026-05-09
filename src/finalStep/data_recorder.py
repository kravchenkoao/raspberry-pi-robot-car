import os
import csv
import json
import time
import queue
import threading
from datetime import datetime

import cv2


class DataRecorder:
    def __init__(
        self,
        base_dir="dataset",
        source="unknown",
        jpeg_quality=85,
        save_every_n_frames=2,
        max_queue_size=300,
        frame_color="RGB"
    ):
        self.base_dir = base_dir
        self.source = source
        self.jpeg_quality = jpeg_quality
        self.save_every_n_frames = save_every_n_frames
        self.frame_color = frame_color.upper()

        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None

        self.enabled = False
        self.stop_worker = False

        self.frame_counter = 0
        self.next_index = 1
        self.queued_counter = 0
        self.written_counter = 0
        self.dropped_counter = 0

        self.q = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.lock = threading.Lock()

        self.fieldnames = [
            "image",
            "steering",
            "throttle",
            "source",
            "timestamp",

            "can_drive",
            "auto_enabled",
            "mode",

            "error",
            "error_norm",
            "lane_center",
            "left_x",
            "right_x",
            "lane_width",
            "lane_locked",

            "visible_side",
            "missing_side",
            "contours"
        ]

    def start(self, meta=None):
        with self.lock:
            if self.enabled:
                return self.session_dir

            session_name = datetime.now().strftime(
                f"session_%Y-%m-%d_%H-%M-%S_{self.source}"
            )

            self.session_dir = os.path.join(self.base_dir, session_name)
            self.images_dir = os.path.join(self.session_dir, "images")
            os.makedirs(self.images_dir, exist_ok=True)

            self.csv_path = os.path.join(self.session_dir, "data.csv")
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=self.fieldnames,
                extrasaction="ignore"
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()

            meta_path = os.path.join(self.session_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta or {}, f, indent=4)

            self.frame_counter = 0
            self.next_index = 1
            self.queued_counter = 0
            self.written_counter = 0
            self.dropped_counter = 0

            self.stop_worker = False
            self.enabled = True

            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

            print("Recording started:", self.session_dir)

            return self.session_dir

    def stop(self):
        with self.lock:
            if not self.enabled and self.worker_thread is None:
                return

            self.enabled = False
            self.stop_worker = True
            worker = self.worker_thread

        if worker is not None:
            self.q.join()
            worker.join(timeout=5.0)

        with self.lock:
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()

            self.csv_file = None
            self.csv_writer = None
            self.worker_thread = None

            print("Recording stopped.")
            print("Written frames:", self.written_counter)
            print("Dropped frames:", self.dropped_counter)

    def close(self):
        self.stop()

    def is_recording(self):
        with self.lock:
            return self.enabled

    def get_stats(self):
        with self.lock:
            return {
                "recording": self.enabled,
                "session_dir": self.session_dir or "",
                "queued_frames": self.queued_counter,
                "written_frames": self.written_counter,
                "dropped_frames": self.dropped_counter,
                "queue_size": self.q.qsize()
            }

    def record(self, frame, steering, throttle, metadata=None):
        with self.lock:
            if not self.enabled:
                return False

            self.frame_counter += 1

            if self.frame_counter % self.save_every_n_frames != 0:
                return False

            idx = self.next_index
            self.next_index += 1

            filename = f"{idx:06d}.jpg"
            rel_path = os.path.join("images", filename)
            abs_path = os.path.join(self.images_dir, filename)

            row = {
                "image": rel_path.replace("\\", "/"),
                "steering": float(steering),
                "throttle": float(throttle),
                "source": self.source,
                "timestamp": time.time(),

                "can_drive": "",
                "auto_enabled": "",
                "mode": "",

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

            if metadata is not None:
                row.update(metadata)

        try:
            self.q.put_nowait((abs_path, frame.copy(), row))

            with self.lock:
                self.queued_counter += 1

            return True

        except queue.Full:
            with self.lock:
                self.dropped_counter += 1

            print("Recorder queue full. Frame skipped.")
            return False

    def _prepare_frame_for_save(self, frame):
        if self.frame_color == "RGB":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.frame_color == "BGR":
            return frame

        return frame

    def _worker(self):
        while not self.stop_worker or not self.q.empty():
            try:
                abs_path, frame, row = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                frame_to_save = self._prepare_frame_for_save(frame)

                ok = cv2.imwrite(
                    abs_path,
                    frame_to_save,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )

                if not ok:
                    raise RuntimeError("cv2.imwrite returned False")

                if self.csv_writer is not None:
                    self.csv_writer.writerow(row)

                if self.csv_file is not None:
                    self.csv_file.flush()

                with self.lock:
                    self.written_counter += 1

            except Exception as e:
                with self.lock:
                    self.dropped_counter += 1

                print("Recorder write error:", e)

            finally:
                self.q.task_done()
