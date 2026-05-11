"""Camera capture abstraction. Handles webcam index or IP-cam URL.

The capture runs in a background thread so the Flask MJPEG generator can read
the latest frame at its own pace without blocking on grab().
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("attendance.camera")


def _parse_source(source: str | int) -> int | str:
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if s.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        u = urlparse(s)
        if not u.path or u.path == "/":
            return s.rstrip("/") + "/video"
    return s


class Camera:
    ROTATIONS = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(self, source: str | int = 0, rotation: int = 0, front_camera: bool = False):
        self.source = _parse_source(source)
        self.rotation: int = rotation
        self.front_camera: bool = front_camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_idx = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

    def start(self) -> bool:
        self._stop.clear()
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            hint = ""
            if isinstance(self.source, str) and self.source.startswith(("http://", "https://", "rtsp://")):
                hint = ("  Phone tip: open the URL in a browser first to confirm it streams. "
                        "For the Android 'IP Webcam' app, the MJPEG path is /video — "
                        "we auto-append it if missing.")
            self.last_error = f"Could not open camera source: {self.source}.{hint}"
            log.error(self.last_error)
            return False
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        log.info("Camera started: %s", self.source)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _loop(self) -> None:
        backoff = 0.05
        while not self._stop.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.last_error = "read() returned no frame; will retry"
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 1.0)
                if backoff >= 1.0:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.source)
                continue
            backoff = 0.05
            frame = self._apply_rotation(frame)
            with self._lock:
                self._frame = frame
                self._frame_idx += 1

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        code = self.ROTATIONS.get(self.rotation)
        if code is not None:
            return cv2.rotate(frame, code)
        return frame

    def read(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_idx

    def latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame, _ = self.read()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None
