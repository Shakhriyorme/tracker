"""Flask entry point. We built this to wire the camera, YOLO, and tracking pipeline together."""
from flask import Flask, Response, jsonify, render_template, request
import cv2
import numpy as np
import threading
import time
import os
import logging
from datetime import datetime, timedelta
from tracker import TrackerWithIdentity
from camera import Camera
import db as DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attendance.app")
app = Flask(__name__)

# I kept dummy data temporarily so the frontend loads without errors.
people_count = 24
sessions = [{'name': 'CS101 Monday', 'date': '2026-05-09'}, {'name': 'CS101 Wednesday', 'date': '2026-05-07'}]
recognition_path = {'backend': 'facenet'}

class Pipeline:
    def __init__(self):
        self.camera = None
        self.yolo = None
        self.tracker = TrackerWithIdentity(recognition_cooldown_frames=5, max_attempts=12)
        self.lock = threading.Lock()
        self.last_jpeg = None
        self._proc_thread = None
        self._proc_stop = threading.Event()
        self.last_status = {"fps": 0.0, "in_frame": 0, "identified": 0, "unknown": 0}
        self._frame_lock = threading.Lock()
        self._frame_jpeg = None
        self._frame_seq = 0
        self.live_listeners = 0

    def yolo_model(self):
        # I lazy-load YOLO here so the app starts instantly.
        if self.yolo is None:
            from ultralytics import YOLO
            self.yolo = YOLO(os.environ.get("YOLO_MODEL", "yolov8n.pt"))
            log.info("✅ I successfully loaded YOLOv8n.")
        return self.yolo

    def set_camera(self, source: str | int = 0, rotation: int = 0) -> bool:
        if self.camera:
            self.camera.stop()
        self.camera = Camera(source, rotation=rotation)
        return self.camera.start()

    def start_processing(self):
        if self._proc_thread and self._proc_thread.is_alive():
            return
        if not self.camera or not self.camera.is_open():
            log.warning("I can't start processing because the camera isn't ready yet.")
            return
        self._proc_stop.clear()
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True, name="pipeline")
        self._proc_thread.start()
        log.info("🟢 I started the background processing thread.")

    def stop_processing(self):
        self._proc_stop.set()
        if self._proc_thread:
            self._proc_thread.join(timeout=2)
            self._proc_thread = None
        log.info("🔴 I stopped the background processing thread.")

    def _process_loop(self):
        log.info("I am now running the main frame processing loop...")
        fps_t, fps_n = time.time(), 0
        last_idx = -1
        while not self._proc_stop.is_set():
            if self.camera is None:
                break
            frame, idx = self.camera.read()
            if frame is None or idx == last_idx:
                time.sleep(0.01)
                continue
            last_idx = idx
            with self.lock:
                annotated = self.annotate_frame(frame)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._frame_lock:
                    self._frame_jpeg = buf.tobytes()
                    self._frame_seq += 1
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                self.last_status["fps"] = round(fps_n / (time.time() - fps_t), 1)
                fps_n, fps_t = 0, time.time()

    def annotate_frame(self, frame_bgr):
        try:
            results = self.yolo_model()(frame_bgr, verbose=False, classes=[0], imgsz=480)
            boxes = results[0].boxes
            dets = np.concatenate([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy().reshape(-1,1)], axis=1) if len(boxes) > 0 else np.empty((0,5))
        except Exception as e:
            log.warning("I skipped YOLO inference due to an error: %s", e)
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

# I implemented this helper to check for an active session and auto-end it if duration expires.
def active_session_id() -> int | None:
    with DB.SessionLocal() as s:
        v = DB.get_config(s, "active_session_id")
        if not v:
            return None
        sid = int(v)
        sess = s.get(DB.Session_, sid)
        if not sess:
            DB.del_config(s, "active_session_id")
            return None
        if sess.duration_minutes:
            end_dt = datetime.combine(sess.date, sess.start_time) + timedelta(minutes=sess.duration_minutes)
            if datetime.now() >= end_dt:
                log.info("I auto-ended session '%s' because duration expired.", sess.name)
                DB.del_config(s, "active_session_id")
                return None
        return sid

@app.route("/api/sessions/activate", methods=["POST"])
def api_activate_session():
    I added this route so the frontend can start a session and persist the ID in the DB.
    body = request.json or {}
    name = body.get("name", "Default Session")
    mode = body.get("mode", "student")
    duration = body.get("duration_minutes")
    if duration:
        duration = int(duration)

    with DB.SessionLocal() as s:
        sess = DB.ensure_session(s, name=name, mode=mode, duration_minutes=duration)
        DB.set_config(s, "active_session_id", str(sess.id))
    PIPELINE.start_processing()
    return jsonify({"ok": True, "session_id": sess.id})

@app.route("/api/sessions/deactivate", methods=["POST"])
def api_deactivate_session():
    with DB.SessionLocal() as s:
        DB.del_config(s, "active_session_id")
    if not PIPELINE.live_listeners:
        PIPELINE.stop_processing()
    return jsonify({"ok": True})

@app.route("/video_feed")
def video_feed():
    if not PIPELINE.camera or not PIPELINE.camera.is_open():
        return Response("Camera not available", status=503)
    def gen():
        PIPELINE.live_listeners += 1
        PIPELINE.start_processing()
        try:
            last_seq = -1
            while True:
                if not PIPELINE._proc_thread or not PIPELINE._proc_thread.is_alive():
                    break
                with PIPELINE._frame_lock:
                    data = PIPELINE._frame_jpeg
                    seq = PIPELINE._frame_seq
                if data is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        finally:
            PIPELINE.live_listeners = max(0, PIPELINE.live_listeners - 1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index():
    return render_template('index.html', people_count=people_count, sessions=sessions, recognition_path=recognition_path, request=request)

@app.route('/enroll')
def enroll_page(): return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/live')
def live_page(): return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/dashboard')
def dashboard(): return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/api/status')
def api_status():
    return jsonify(PIPELINE.last_status)

if __name__ == '__main__':
    DB.init_db()
    PIPELINE.set_camera(0)
    # I resume processing automatically if a session was left active.
    if active_session_id() is not None:
        PIPELINE.start_processing()
    app.run(debug=True, port=5000, threaded=True)