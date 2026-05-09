from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import os
import logging
from tracker import TrackerWithIdentity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attendance.app")
app = Flask(__name__)

# --- Dummy data (frontend bilan sinxronizatsiya uchun) ---
people_count = 24
sessions = [{'name': 'CS101 Monday', 'date': '2026-05-09'}, {'name': 'CS101 Wednesday', 'date': '2026-05-07'}]

class Pipeline:
    """Owns camera, YOLO, tracker. Thread-safe background processing."""
    def __init__(self):
        self.yolo = None
        self.tracker = TrackerWithIdentity(recognition_cooldown_frames=5, max_attempts=12)
        self.lock = threading.Lock()
        self.last_jpeg = None
        self._proc_thread = None
        self._proc_stop = threading.Event()
        self.last_status = {"fps": 0.0, "in_frame": 0, "identified": 0, "unknown": 0}

    def yolo_model(self):
        if self.yolo is None:
            from ultralytics import YOLO
            self.yolo = YOLO(os.environ.get("YOLO_MODEL", "yolov8n.pt"))
            log.info("✅ YOLOv8n loaded successfully.")
        return self.yolo

    def start_processing(self):
        if self._proc_thread and self._proc_thread.is_alive(): return
        self._proc_stop.clear()
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True, name="pipeline")
        self._proc_thread.start()
        log.info("🟢 Background processing started.")

    def stop_processing(self):
        self._proc_stop.set()
        if self._proc_thread:
            self._proc_thread.join(timeout=2)
            self._proc_thread = None

    def _process_loop(self):
        log.info("Processing loop running...")
        fps_t, fps_n = time.time(), 0
        # NOTE: Camera read() Commit 2 da qo'shiladi. Hozircha loop ishlayotganini tasdiqlaydi.
        while not self._proc_stop.is_set():
            time.sleep(0.05)  # Placeholder frame rate
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                self.last_status["fps"] = round(fps_n / (time.time() - fps_t), 1)
                fps_n, fps_t = 0, time.time()

    def annotate_frame(self, frame_bgr):
        """YOLO → SORT → bbox chizish (v1 skeleton)"""
        try:
            results = self.yolo_model()(frame_bgr, verbose=False, classes=[0], imgsz=480)
            boxes = results[0].boxes
            dets = np.concatenate([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy().reshape(-1,1)], axis=1) if len(boxes) > 0 else np.empty((0,5))
        except Exception as e:
            log.warning("YOLO inference skipped: %s", e)
            dets = np.empty((0,5))

        active = self.tracker.step(dets)
        self.last_status["in_frame"] = len(active)
        for info in active:
            x1,y1,x2,y2 = info.bbox
            color = (0,255,0) if info.name else (0,0,255)
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame_bgr, info.name or f"#{info.track_id}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame_bgr

PIPELINE = Pipeline()

# --- Sherigingiz route lari (o'zgarmasdan qoldi) ---
@app.route('/')
def index():
    return render_template('index.html', people_count=people_count, sessions=sessions)

@app.route('/enroll')
def enroll_page(): return render_template('base.html')

@app.route('/live')
def live_page(): return render_template('base.html')

@app.route('/dashboard')
def dashboard(): return render_template('base.html')

@app.route('/api/status')
def api_status():
    return jsonify(PIPELINE.last_status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)