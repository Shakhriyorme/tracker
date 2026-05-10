"""Flask app entry point.
We wire the camera, YOLO, SORT tracker, and face recognition together into a single Pipeline.
We expose REST endpoints for enrollment, session control, and live MJPEG streaming.
"""
import logging
import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from datetime import datetime, timedelta
from sqlalchemy import select

import db as DB
import recognition as REC
from camera import Camera
from tracker import TrackerWithIdentity, TrackInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attendance.app")
app = Flask(__name__)

# =================================================================== Pipeline ===
class Pipeline:
    """We manage the camera, detection, tracking, and recognition in one thread-safe class."""
    def __init__(self):
        self.camera = None
        self.yolo = None
        self.tracker = TrackerWithIdentity(recognition_cooldown_frames=5, max_attempts=12)
        self.lock = threading.Lock()
        self.match_threshold = 0.55  # We use a slightly relaxed threshold for stability
        self.recognition_path = None
        self.live_listeners = 0
        self.last_jpeg = None
        self.ffc_front = False
        self.last_status = {"fps": 0.0, "in_frame": 0, "identified": 0, "unknown": 0}
        self._fps_t = time.time()
        self._fps_n = 0
        self._yolo_lock = threading.Lock()
        self._proc_thread = None
        self._proc_stop = threading.Event()
        self._frame_jpeg = None
        self._frame_seq = 0
        self._frame_lock = threading.Lock()

    def yolo_model(self):
        """We lazy-load YOLO so the Flask server starts instantly even if the model downloads."""
        if self.yolo is None:
            with self._yolo_lock:
                if self.yolo is None:
                    from ultralytics import YOLO
                    self.yolo = YOLO(os.environ.get("YOLO_MODEL", "yolov8n.pt"))
                    log.info("We successfully loaded YOLOv8n for person detection.")
        return self.yolo

    def set_camera(self, source, rotation=0):
        """We safely switch the camera source and apply rotation if needed."""
        with self.lock:
            if self.camera:
                self.camera.stop()
            self.camera = Camera(source, rotation=rotation, front_camera=self.ffc_front)
            return self.camera.start()

    def ensure_camera(self):
        """We restore the last used camera from DB config if available."""
        if self.camera and self.camera.is_open():
            return True
        with DB.SessionLocal() as s:
            saved = DB.get_config(s, "camera_source", "0")
            rot = int(DB.get_config(s, "camera_rotation", "0"))
        return self.set_camera(saved or "0", rotation=rot)

    def start_processing(self):
        """We spawn a background thread to run the detection-tracking loop."""
        if self._proc_thread and self._proc_thread.is_alive():
            return
        if not self.ensure_camera():
            log.warning("We can't start processing because the camera isn't ready.")
            return
        self._proc_stop.clear()
        self._proc_thread = threading.Thread(target=self._process_loop, name="pipeline", daemon=True)
        self._proc_thread.start()
        log.info("We started the background processing thread.")

    def stop_processing(self):
        """We gracefully stop the processing thread and release resources."""
        self._proc_stop.set()
        if self._proc_thread:
            self._proc_thread.join(timeout=2)
            self._proc_thread = None
        log.info("We stopped the background processing thread.")

    @property
    def is_processing(self):
        return self._proc_thread is not None and self._proc_thread.is_alive()

    def _process_loop(self):
        """We run the main frame pipeline: read -> detect -> track -> annotate -> cache."""
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
                annotated = self.annotate_frame(frame, active_session_id())
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._frame_lock:
                    self._frame_jpeg = buf.tobytes()
                    self._frame_seq += 1
            self._fps_n += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self.last_status["fps"] = round(self._fps_n / (now - self._fps_t), 1)
                self._fps_n = 0
                self._fps_t = now

    def annotate_frame(self, frame_bgr, session_id):
        """We run YOLO -> SORT -> recognition per track and draw boxes on the frame."""
        try:
            results = self.yolo_model()(frame_bgr, verbose=False, classes=[0], imgsz=480)
            boxes = results[0].boxes
            dets = np.concatenate([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy().reshape(-1, 1)], axis=1) if len(boxes) > 0 else np.empty((0, 5))
        except Exception as e:
            log.warning("We skipped YOLO inference: %s", e)
            dets = np.empty((0, 5))

        active = self.tracker.step(dets)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        identified = unknown = 0

        for info in active:
            if info.name:
                identified += 1
            else:
                unknown += 1
            if self.tracker.needs_recognition(info):
                self.tracker.mark_attempt(info)
                self._try_recognize(info, rgb, session_id)

        self._draw(frame_bgr, active)
        self.last_status.update({"in_frame": len(active), "identified": identified, "unknown": unknown})
        return frame_bgr

    def _try_recognize(self, info, rgb, session_id):
        """We crop the face, run recognition, and attach identity if match passes threshold."""
        x1, y1, x2, y2 = info.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2 - x1 < 40 or y2 - y1 < 60:
            return
        crop = rgb[y1:y2, x1:x2]
        try:
            faces = REC.detect_and_embed(crop)
        except Exception as e:
            log.warning("We skipped recognition due to error: %s", e)
            return
        if not faces:
            return
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        with DB.SessionLocal() as s:
            person, dist = DB.find_nearest(s, face.embedding, threshold=self.match_threshold)
            if person:
                self.tracker.assign_identity(info, person.id, person.name)
                log.info("We identified %s on track %d (dist=%.3f)", person.name, info.track_id, dist)
                if session_id:
                    sess = s.get(DB.Session_, session_id)
                    if sess:
                        DB.record_arrival(s, person.id, sess)

    PERSON_PALETTE = [(212,182,6),(94,197,34),(246,92,139),(0,215,255),(255,191,0),
                      (180,105,255),(0,255,127),(255,144,30),(50,205,154),(147,20,255)]
    UNKNOWN_COLOR = (70,70,230)

    def _draw(self, frame, tracks):
        """We draw bounding boxes, names, and a HUD status bar on the frame."""
        pal = self.PERSON_PALETTE
        for info in tracks:
            x1,y1,x2,y2 = info.bbox
            color = pal[info.person_id % len(pal)] if info.person_id else self.UNKNOWN_COLOR
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{info.name}" if info.name else f"unknown #{info.track_id}"
            label_w = 8 + int(9.5 * len(label))
            cv2.rectangle(frame, (x1, y1-22), (x1+label_w, y1), color, -1)
            txt_color = (15,15,15) if info.name else (255,255,255)
            cv2.putText(frame, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)
        hud = f"{self.last_status['fps']:.1f} fps | in: {self.last_status['in_frame']} | id: {self.last_status['identified']} | unk: {self.last_status['unknown']}"
        cv2.rectangle(frame, (0,0), (frame.shape[1],26), (11,11,6), -1)
        cv2.putText(frame, hud, (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,240), 1, cv2.LINE_AA)

PIPELINE = Pipeline()

# =================================================================== Helpers ===
def active_session_id():
    """We check DB for an active session and auto-end it if duration expires."""
    with DB.SessionLocal() as s:
        v = DB.get_config(s, "active_session_id")
        if not v: return None
        sid = int(v)
        sess = s.get(DB.Session_, sid)
        if not sess:
            DB.del_config(s, "active_session_id")
            return None
        if sess.duration_minutes:
            end_dt = datetime.combine(sess.date, sess.start_time) + timedelta(minutes=sess.duration_minutes)
            if datetime.now() >= end_dt:
                log.info("We auto-ended session '%s' due to duration limit.", sess.name)
                DB.del_config(s, "active_session_id")
                return None
        return sid

# =================================================================== Routes ===
@app.route("/video_feed")
def video_feed():
    """We stream MJPEG frames to any number of concurrent browser tabs."""
    if not PIPELINE.ensure_camera():
        return Response("Camera not available", status=503)
    def gen():
        PIPELINE.live_listeners += 1
        PIPELINE.start_processing()
        try:
            last_seq = -1
            while True:
                if not PIPELINE.is_processing:
                    break
                with PIPELINE._frame_lock:
                    data, seq = PIPELINE._frame_jpeg, PIPELINE._frame_seq
                if data is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        finally:
            PIPELINE.live_listeners = max(0, PIPELINE.live_listeners - 1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/sessions/activate", methods=["POST"])
def api_activate_session():
    """We activate a session, store its ID in config, and start the pipeline."""
    body = request.json or {}
    name = body.get("name", "Default Session")
    mode = body.get("mode", "student")
    duration = int(body["duration_minutes"]) if body.get("duration_minutes") else None
    with DB.SessionLocal() as s:
        sess = DB.ensure_session(s, name=name, mode=mode, group=None, duration_minutes=duration)
        DB.set_config(s, "active_session_id", str(sess.id))
    PIPELINE.start_processing()
    return jsonify({"ok": True, "session_id": sess.id})

@app.route("/api/sessions/deactivate", methods=["POST"])
def api_deactivate_session():
    """We deactivate the current session and stop processing if no listeners remain."""
    with DB.SessionLocal() as s:
        DB.del_config(s, "active_session_id")
    if not PIPELINE.live_listeners:
        PIPELINE.stop_processing()
    return jsonify({"ok": True})

@app.route("/api/enroll/precheck", methods=["POST"])
def api_enroll_precheck():
    """We grab a live frame, detect a face, and check if the person is already enrolled."""
    if not PIPELINE.ensure_camera():
        return jsonify({"recognized": False, "error": "Camera unavailable"})
    frame, _ = PIPELINE.camera.read()
    if frame is None:
        return jsonify({"recognized": False, "error": "No frame"})
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = REC.detect_and_embed(rgb)
    if not faces:
        return jsonify({"recognized": False, "face_detected": False})
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    with DB.SessionLocal() as s:
        person, dist = DB.find_nearest(s, face.embedding, threshold=PIPELINE.match_threshold)
        if person:
            return jsonify({"recognized": True, "person": {"id": person.id, "name": person.name}, "distance": round(dist, 3)})
    return jsonify({"recognized": False, "face_detected": True})

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """We capture a frame, extract embedding, and store it as a face sample for the person."""
    body = request.json or {}
    name = body.get("name", "").strip()
    ext_id = body.get("external_id", "").strip()
    role = body.get("role", "student")
    pose = body.get("pose", "").lower() or None
    if not name or not ext_id:
        return jsonify({"error": "name and external_id required"}), 400
    if not PIPELINE.ensure_camera():
        return jsonify({"error": "Camera unavailable"}), 503

    for _ in range(20):
        frame, _ = PIPELINE.camera.read()
        if frame is not None: break
        time.sleep(0.05)
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = REC.detect_and_embed(rgb)
    if not faces:
        return jsonify({"error": "No face detected"}), 400
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

    with DB.SessionLocal() as s:
        person = s.execute(select(DB.Person).where(DB.Person.external_id == ext_id)).scalar_one_or_none()
        if not person:
            person = DB.Person(external_id=ext_id, name=name, role=role)
            s.add(person)
            s.flush()
        else:
            person.name = name
            person.role = role

        DB.add_face_sample(s, person, emb, pose_label=pose)
        s.commit()
        total = DB.count_samples(s, person.id)
        # We reset recognition cooldowns so newly enrolled faces are immediately trackable
        for info in PIPELINE.tracker._infos.values():
            if info.person_id is None:
                info.recognition_attempts = 0
                info.last_recognition_frame = -10**9
    return jsonify({"ok": True, "person": {"id": person.id, "name": person.name}, "total_samples": total})

@app.route("/api/persons")
def api_persons():
    """We return a list of all enrolled persons for the frontend UI."""
    with DB.SessionLocal() as s:
        people = s.execute(select(DB.Person).order_by(DB.Person.name)).scalars().all()
    return jsonify([{"id": p.id, "external_id": p.external_id, "name": p.name, "role": p.role} for p in people])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enroll")
def enroll_page(): return render_template("enroll.html")

@app.route("/live")
def live_page(): return render_template("live.html")

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    return jsonify(PIPELINE.last_status)

# =================================================================== Startup ===
if __name__ == "__main__":
    DB.init_db()
    PIPELINE.recognition_path = REC.init()
    if active_session_id() is not None:
        PIPELINE.start_processing()
    app.run(debug=True, port=5000, threaded=True)