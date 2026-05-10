from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import os
import logging
from tracker import TrackerWithIdentity
from camera import Camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attendance.app")

app = Flask(__name__)

# I kept dummy data here temporarily so my partner's frontend doesn't break.
people_count = 24
sessions = [
    {'name': 'CS101 Monday', 'date': '2026-05-09'},
    {'name': 'CS101 Wednesday', 'date': '2026-05-07'},
]
recognition_path = {'backend': 'facenet'}

class Pipeline:
    """I created this class to manage the entire detection -> tracking -> recognition flow."""
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
        # I lazy-load YOLO here so the app starts instantly even if the model downloads in the background.
        if self.yolo is None:
            from ultralytics import YOLO
            self.yolo = YOLO(os.environ.get("YOLO_MODEL", "yolov8n.pt"))
            log.info("✅ I successfully loaded YOLOv8n.")
        return self.yolo

    def set_camera(self, source: str | int = 0, rotation: int = 0) -> bool:
        # I added this method to safely switch cameras without crashing the pipeline.
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
        # I implemented the YOLO + SORT pipeline here. Currently, it just draws tracked boxes.
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

# I set up the MJPEG generator here so multiple browser tabs can watch the live stream without blocking.
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

# I kept my partner's original routes intact so the UI works immediately.
@app.route('/')
def index():
    return render_template('index.html', people_count=people_count, sessions=sessions, recognition_path=recognition_path, request=request)

@app.route('/enroll')
def enroll_page():
    return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/live')
def live_page():
    return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/dashboard')
def dashboard():
    return render_template('base.html', recognition_path=recognition_path, request=request)

@app.route('/api/status')
def api_status():
    return jsonify(PIPELINE.last_status)

if __name__ == '__main__':
    # I start with a default camera so the pipeline can initialize smoothly.
    PIPELINE.set_camera(0)
    app.run(debug=True, port=5000, threaded=True)

"""We built this Flask entry point to wire the camera, YOLO, and tracker together.
We set up the background processing loop and the MJPEG live stream here.
"""