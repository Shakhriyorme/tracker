"""Flask app entry point.

Wires together camera + YOLO + SORT + face recognition + DB. One Pipeline
instance is shared across the live-stream consumers.

Run:
    python app.py

Env:
    DATABASE_URL    sqlite (default) or postgresql://... for Neon
    HOST, PORT      bind options (default 127.0.0.1:5000)
    YOLO_MODEL      override (default yolov8n.pt)
    RECOGNITION     force backend: insightface | dlib | lbp
"""
from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (
    Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for,
)
from sqlalchemy import or_, select, func

import db as DB
import recognition as REC
from camera import Camera
from tracker import TrackerWithIdentity, TrackInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("attendance.app")

THUMBS_DIR = Path("static/thumbs")
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

STORE_THUMBNAILS = os.environ.get("STORE_THUMBNAILS", "false").lower() in ("1", "true", "yes", "on")

app = Flask(__name__)


# =============================================================== Pipeline ===

class Pipeline:
    """Owns camera, YOLO, tracker, recognition. Thread-safe over the live loop."""

    def __init__(self):
        self.camera: Optional[Camera] = None
        self.yolo = None
        self.tracker = TrackerWithIdentity(recognition_cooldown_frames=5, max_attempts=12)
        self.lock = threading.Lock()
        self.match_threshold = float(os.environ.get("MATCH_THRESHOLD", "0.5"))
        self.recognition_path: dict | None = None
        self.live_listeners = 0
        self.last_jpeg: bytes | None = None
        self.ffc_front: bool = False
        self.last_status = {"fps": 0.0, "in_frame": 0, "identified": 0, "unknown": 0}
        self._fps_t = time.time()
        self._fps_n = 0
        self._yolo_lock = threading.Lock()
        # Background processing thread
        self._proc_thread: Optional[threading.Thread] = None
        self._proc_stop = threading.Event()
        self._frame_jpeg: bytes | None = None
        self._frame_seq = 0
        self._frame_lock = threading.Lock()

    # ----- yolo lazy load --------------------------------------------------
    def yolo_model(self):
        if self.yolo is None:
            with self._yolo_lock:
                if self.yolo is None:
                    from ultralytics import YOLO
                    name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
                    log.info("Loading YOLO: %s", name)
                    self.yolo = YOLO(name)
        return self.yolo

    # ----- camera ----------------------------------------------------------
    def set_camera(self, source: str | int, rotation: int = 0) -> bool:
        with self.lock:
            if self.camera:
                self.camera.stop()
            self.camera = Camera(source, rotation=rotation, front_camera=self.ffc_front)
            return self.camera.start()

    @staticmethod
    def _safe_rotation(val: str) -> int:
        try:
            r = int(val)
            return r if r in (0, 90, 180, 270) else 0
        except (ValueError, TypeError):
            return 0

    def ensure_camera(self) -> bool:
        if self.camera and self.camera.is_open():
            return True
        with DB.SessionLocal() as s:
            saved = DB.get_config(s, "camera_source", "0")
            rotation = self._safe_rotation(DB.get_config(s, "camera_rotation", "0"))
        return self.set_camera(saved or "0", rotation=rotation)

    # ----- background processing thread ------------------------------------
    def start_processing(self) -> None:
        if self._proc_thread and self._proc_thread.is_alive():
            return
        if not self.ensure_camera():
            log.warning("Cannot start processing — camera unavailable")
            return
        self._proc_stop.clear()
        self._proc_thread = threading.Thread(target=self._process_loop, name="pipeline", daemon=True)
        self._proc_thread.start()
        log.info("Background processing started")

    def stop_processing(self) -> None:
        self._proc_stop.set()
        if self._proc_thread:
            self._proc_thread.join(timeout=2)
            self._proc_thread = None
        log.info("Background processing stopped")

    @property
    def is_processing(self) -> bool:
        return self._proc_thread is not None and self._proc_thread.is_alive()

    def _process_loop(self) -> None:
        last_idx = -1
        sid = active_session_id()
        sid_refresh = time.time()
        while not self._proc_stop.is_set():
            if self.camera is None:
                break
            now = time.time()
            if now - sid_refresh > 5.0:
                sid = active_session_id()
                sid_refresh = now
                if not self.live_listeners and sid is None:
                    break
            frame, idx = self.camera.read()
            if frame is None or idx == last_idx:
                time.sleep(0.01)
                continue
            last_idx = idx
            with self.lock:
                annotated = self.annotate_frame(frame, sid)
            data = self.cache_jpeg(annotated, quality=80)
            if data:
                with self._frame_lock:
                    self._frame_jpeg = data
                    self._frame_seq += 1
        log.info("Processing loop exited")

    # ----- main per-frame step --------------------------------------------
    def annotate_frame(self, frame_bgr: np.ndarray, session_id: int | None) -> np.ndarray:
        """Run detection -> tracking -> recognition -> annotate. Returns BGR frame."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # YOLO person detection
        try:
            results = self.yolo_model()(frame_bgr, verbose=False, classes=[0], imgsz=480)
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                dets = np.concatenate([xyxy, conf], axis=1)
            else:
                dets = np.empty((0, 5))
        except Exception as e:
            log.exception("YOLO failed: %s", e)
            dets = np.empty((0, 5))

        # SORT
        active = self.tracker.step(dets)

        identified = 0
        unknown = 0
        recognized_this_frame = False
        for info in active:
            if info.name:
                identified += 1
            else:
                unknown += 1

            if not recognized_this_frame and self.tracker.needs_recognition(info):
                self.tracker.mark_attempt(info)
                self._try_recognize(info, rgb, session_id)
                recognized_this_frame = True

        self._draw(frame_bgr, active)

        # FPS
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.last_status["fps"] = round(self._fps_n / (now - self._fps_t), 1)
            self._fps_n = 0
            self._fps_t = now
        self.last_status["in_frame"] = len(active)
        self.last_status["identified"] = identified
        self.last_status["unknown"] = unknown
        return frame_bgr

    def _try_recognize(self, info: TrackInfo, rgb: np.ndarray, session_id: int | None):
        x1, y1, x2, y2 = info.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2 - x1 < 40 or y2 - y1 < 60:
            return
        crop = rgb[y1:y2, x1:x2]
        try:
            faces = REC.detect_and_embed(crop)
        except Exception as e:
            log.exception("recognition error: %s", e)
            return
        if not faces:
            return
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        with DB.SessionLocal() as s:
            person, dist = DB.find_nearest(s, face.embedding, threshold=self.match_threshold)
            log.debug("track %d: nearest dist=%.3f (threshold=%.2f) person=%s",
                       info.track_id, dist, self.match_threshold,
                       person.name if person else "none")
            if person is None:
                if info.recognition_attempts >= self.tracker.max_attempts:
                    s.add(DB.Unknown(session_id=session_id))
                    s.commit()
                return

            # Duplicate-identity guard: if another active track already claims
            # this person, only the closer match keeps the identity. The other
            # track stays unknown until it actually gets a better match elsewhere.
            existing = next(
                (t for t in self.tracker._infos.values()
                 if t.person_id == person.id and t.track_id != info.track_id),
                None,
            )
            if existing is not None:
                # We don't store the other track's distance, so we use a simple
                # tiebreaker: the OLDER track (more frames) keeps the identity.
                # The newer/lower-confidence track stays unknown for now.
                log.info("Skipping duplicate identity assign for %s on track %d "
                         "(already on track %d)", person.name, info.track_id, existing.track_id)
                return

            self.tracker.assign_identity(info, person.id, person.name)
            if session_id is not None:
                sess = s.get(DB.Session_, session_id)
                if sess:
                    DB.record_arrival(s, person.id, sess)

    def cache_jpeg(self, frame_bgr: np.ndarray, quality: int = 80) -> bytes | None:
        """Encode + cache the latest annotated frame for /api/live/snapshot."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        data = buf.tobytes()
        self.last_jpeg = data
        return data

    PERSON_PALETTE = [
        (212, 182, 6),    # teal
        (94, 197, 34),    # green
        (246, 92, 139),   # purple
        (0, 215, 255),    # gold
        (255, 191, 0),    # sky blue
        (180, 105, 255),  # pink
        (0, 255, 127),    # spring green
        (255, 144, 30),   # dodger blue
        (50, 205, 154),   # aquamarine
        (147, 20, 255),   # deep pink
        (0, 165, 255),    # orange
        (203, 192, 100),  # light steel
    ]
    UNKNOWN_COLOR = (70, 70, 230)  # red BGR

    def _draw(self, frame: np.ndarray, tracks: list[TrackInfo]) -> None:
        pal = self.PERSON_PALETTE
        for info in tracks:
            x1, y1, x2, y2 = info.bbox
            if info.person_id:
                color = pal[info.person_id % len(pal)]
            else:
                color = self.UNKNOWN_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{info.name}" if info.name else f"unknown #{info.track_id}"
            label_w = 8 + int(9.5 * len(label))
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + label_w, y1), color, -1)
            text_color = (15, 15, 15) if info.name else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
        hud = f"{self.last_status['fps']:.1f} fps  |  in-frame {self.last_status['in_frame']}  |  id'd {self.last_status['identified']}  |  unk {self.last_status['unknown']}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (11, 11, 6), -1)
        cv2.putText(frame, hud, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 240), 1, cv2.LINE_AA)

    @staticmethod
    def _save_thumb(rgb: np.ndarray, prefix: str = "thumb") -> str | None:
        if not STORE_THUMBNAILS:
            return None  # matrix-only mode: never write pixels to disk
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{prefix}_{ts}.jpg"
        out = THUMBS_DIR / fname
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), bgr)
        return f"thumbs/{fname}"


PIPELINE = Pipeline()


# ============================================================ helpers / routes

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
                log.info("Auto-ending session '%s' (duration %dm expired)", sess.name, sess.duration_minutes)
                DB.del_config(s, "active_session_id")
                return None
        return sid


def active_session_dict() -> dict | None:
    sid = active_session_id()
    if not sid:
        return None
    with DB.SessionLocal() as s:
        sess = s.get(DB.Session_, sid)
        if not sess:
            return None
        return _session_to_dict(sess)


def _session_to_dict(sess: DB.Session_, member_ids: list[int] | None = None) -> dict:
    d = {
        "id": sess.id, "name": sess.name, "mode": sess.mode,
        "group_name": sess.group_name, "date": sess.date.isoformat(),
        "start_time": sess.start_time.strftime("%H:%M"),
        "late_threshold_minutes": sess.late_threshold_minutes,
        "duration_minutes": sess.duration_minutes,
    }
    if member_ids is not None:
        d["member_ids"] = member_ids
    return d


@app.context_processor
def inject_globals():
    with DB.SessionLocal() as s:
        cam = DB.get_config(s, "camera_source", "0")
    return {
        "camera_source": cam or "0",
        "active_session": active_session_dict(),
        "recognition_path": PIPELINE.recognition_path or {},
        "db_backend": "postgres" if DB.IS_POSTGRES else "sqlite",
        "pgvector": DB.PGVECTOR_AVAILABLE,
    }


# ----- Pages ----------------------------------------------------------------

@app.route("/")
def index():
    with DB.SessionLocal() as s:
        sess_rows = s.execute(
            select(DB.Session_).order_by(DB.Session_.date.desc(), DB.Session_.id.desc())
        ).scalars().all()
        sessions = []
        for x in sess_rows:
            mids = DB.get_session_member_ids(s, x.id)
            sessions.append(_session_to_dict(x, member_ids=mids))
        people = s.execute(select(DB.Person).order_by(DB.Person.name)).scalars().all()
        people_list = [_person_to_dict(p) for p in people]
        people_count = len(people_list)
    return render_template("index.html", sessions=sessions, people_count=people_count, people=people_list)


@app.route("/enroll")
def enroll_page():
    with DB.SessionLocal() as s:
        people = s.execute(select(DB.Person).order_by(DB.Person.created_at.desc())).scalars().all()
        people = [_person_to_dict(p) for p in people]
    return render_template("enroll.html", people=people)


@app.route("/live")
def live_page():
    return render_template("live.html")


@app.route("/dashboard")
def dashboard():
    sid = active_session_id()
    return render_template("dashboard.html", session_id=sid)


@app.route("/person/<int:pid>")
def person_page(pid: int):
    with DB.SessionLocal() as s:
        p = s.get(DB.Person, pid)
        if not p:
            abort(404)
    return render_template("person.html", person=_person_to_dict(p))


# ----- MJPEG stream ---------------------------------------------------------

@app.route("/video_feed")
def video_feed():
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


# ----- API: status / camera / sessions / persons / enrollment --------------

@app.route("/api/status")
def api_status():
    cam = PIPELINE.camera
    return jsonify({
        "camera_open": bool(cam and cam.is_open()),
        "camera_source": str(cam.source) if cam else None,
        "fps": PIPELINE.last_status["fps"],
        "in_frame": PIPELINE.last_status["in_frame"],
        "identified": PIPELINE.last_status["identified"],
        "unknown": PIPELINE.last_status["unknown"],
        "recognition": PIPELINE.recognition_path or {},
        "db": {"backend": "postgres" if DB.IS_POSTGRES else "sqlite", "pgvector": DB.PGVECTOR_AVAILABLE},
        "active_session": active_session_dict(),
        "ffc_front": PIPELINE.ffc_front,
        "processing": PIPELINE.is_processing,
    })


@app.route("/api/camera", methods=["POST"])
def api_camera():
    body = request.json or {}
    src = body.get("source")
    if src is None or src == "":
        return jsonify({"ok": False, "error": "missing source"}), 400
    is_ip = not str(src).strip().isdigit()
    if is_ip:
        rot = 270 if PIPELINE.ffc_front else 90
    else:
        rot = 0
    with DB.SessionLocal() as s:
        DB.set_config(s, "camera_rotation", str(rot))
    was_processing = PIPELINE.is_processing
    if was_processing:
        PIPELINE.stop_processing()
    ok = PIPELINE.set_camera(src, rotation=rot)
    if ok:
        with DB.SessionLocal() as s:
            DB.set_config(s, "camera_source", str(src))
        if was_processing or active_session_id() is not None:
            PIPELINE.start_processing()
    return jsonify({"ok": ok, "rotation": rot, "error": (None if ok else (PIPELINE.camera.last_error if PIPELINE.camera else "open failed"))})


@app.route("/api/camera/rotation", methods=["POST"])
def api_camera_rotation():
    body = request.json or {}
    rot = body.get("rotation", 0)
    try:
        rot = int(rot)
    except (ValueError, TypeError):
        return jsonify({"error": "rotation must be 0, 90, 180, or 270"}), 400
    if rot not in (0, 90, 180, 270):
        return jsonify({"error": "rotation must be 0, 90, 180, or 270"}), 400
    with DB.SessionLocal() as s:
        DB.set_config(s, "camera_rotation", str(rot))
    if PIPELINE.camera:
        PIPELINE.camera.rotation = rot
    return jsonify({"ok": True, "rotation": str(rot)})


@app.route("/api/sessions", methods=["GET", "POST"])
def api_sessions():
    if request.method == "POST":
        body = request.json or {}
        name = (body.get("name") or "").strip()
        mode = body.get("mode") or "student"
        group = body.get("group_name") or None
        start = body.get("start_time") or "09:00"
        late = int(body.get("late_threshold_minutes") or 15)
        dur_raw = body.get("duration_minutes")
        duration = int(dur_raw) if dur_raw not in (None, "", 0, "0") else None
        date_str = (body.get("session_date") or "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        from datetime import time as Time, date as Date_
        h, m = [int(x) for x in start.split(":")]
        for_date = Date_.fromisoformat(date_str) if date_str else None
        if for_date and for_date < Date_.today():
            return jsonify({"error": "Date can't be in the past."}), 400
        if (for_date is None or for_date == Date_.today()):
            now = datetime.now()
            requested = Time(h, m)
            if requested < now.time():
                h, m = now.hour, now.minute
        member_ids = body.get("member_ids") or []
        try:
            with DB.SessionLocal() as s:
                sess = DB.ensure_session(s, name=name, mode=mode, group=group,
                                         start_time=Time(h, m), late_minutes=late,
                                         for_date=for_date, duration_minutes=duration)
                if member_ids:
                    DB.set_session_members(s, sess.id, [int(x) for x in member_ids])
                DB.set_config(s, "active_session_id", str(sess.id))
                mids = DB.get_session_member_ids(s, sess.id)
                PIPELINE.start_processing()
                return jsonify(_session_to_dict(sess, member_ids=mids))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    with DB.SessionLocal() as s:
        rows = s.execute(select(DB.Session_).order_by(DB.Session_.date.desc())).scalars().all()
        return jsonify([_session_to_dict(x) for x in rows])


@app.route("/api/sessions/<int:sid>/activate", methods=["POST"])
def api_activate_session(sid: int):
    with DB.SessionLocal() as s:
        if not s.get(DB.Session_, sid):
            return jsonify({"error": "not found"}), 404
        DB.set_config(s, "active_session_id", str(sid))
    PIPELINE.start_processing()
    return jsonify({"ok": True})


@app.route("/api/sessions/deactivate", methods=["POST"])
def api_deactivate_session():
    with DB.SessionLocal() as s:
        DB.del_config(s, "active_session_id")
    if not PIPELINE.live_listeners:
        PIPELINE.stop_processing()
    return jsonify({"ok": True})


def _person_to_dict(p: DB.Person) -> dict:
    return {
        "id": p.id, "external_id": p.external_id, "name": p.name,
        "group_name": p.group_name, "role": p.role,
        "thumbnail": (url_for("static", filename=p.thumbnail_path) if p.thumbnail_path else None),
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


@app.route("/api/persons")
def api_persons():
    with DB.SessionLocal() as s:
        rows = s.execute(select(DB.Person).order_by(DB.Person.name)).scalars().all()
        return jsonify([_person_to_dict(p) for p in rows])


@app.route("/api/persons/<int:pid>", methods=["DELETE"])
def api_delete_person(pid: int):
    with DB.SessionLocal() as s:
        p = s.get(DB.Person, pid)
        if not p:
            return jsonify({"error": "not found"}), 404
        s.delete(p)
        s.commit()
    for info in list(PIPELINE.tracker._infos.values()):
        if info.person_id == pid:
            info.person_id = None
            info.name = None
            info.recognition_attempts = 0
            info.last_recognition_frame = -10**9
    return jsonify({"ok": True})


@app.route("/api/enroll/precheck", methods=["POST"])
def api_enroll_precheck():
    """Grab a frame, run recognition. If someone is already enrolled, return their info."""
    if not PIPELINE.ensure_camera():
        return jsonify({"recognized": False})
    frame, _ = PIPELINE.camera.read()
    if frame is None:
        return jsonify({"recognized": False})
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = REC.detect_and_embed(rgb)
    if not faces:
        return jsonify({"recognized": False, "face_detected": False})
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    detected_pose = REC.estimate_pose(rgb, mirror=PIPELINE.ffc_front)
    with DB.SessionLocal() as s:
        person, dist = DB.find_nearest(s, face.embedding, threshold=PIPELINE.match_threshold)
        if person:
            return jsonify({
                "recognized": True,
                "person": _person_to_dict(person),
                "distance": round(dist, 3),
                "pose": detected_pose,
                "samples": DB.count_samples(s, person.id),
            })
    return jsonify({"recognized": False, "face_detected": True, "pose": detected_pose})


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    body = request.json or {}
    name = (body.get("name") or "").strip()
    ext_id = (body.get("external_id") or "").strip()
    group = body.get("group_name") or None
    role = body.get("role") or "student"
    pose = (body.get("pose") or "").strip().lower() or None
    if not name or not ext_id:
        return jsonify({"error": "name and external_id required"}), 400
    if not PIPELINE.ensure_camera():
        return jsonify({"error": "camera not available"}), 503

    for _retry in range(20):
        frame, _idx = PIPELINE.camera.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        return jsonify({"error": "no frame from camera"}), 400
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = REC.detect_and_embed(rgb)
    if not faces:
        return jsonify({"error": "no face detected"}), 400
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    detected_pose = REC.estimate_pose(rgb, mirror=PIPELINE.ffc_front)

    emb = face.embedding
    n = emb / (np.linalg.norm(emb) + 1e-9)

    with DB.SessionLocal() as s:
        existing = s.execute(select(DB.Person).where(DB.Person.external_id == ext_id)).scalar_one_or_none()
        if existing:
            person = existing
            person.name = name
            person.group_name = group
            person.role = role
        else:
            person = DB.Person(name=name, external_id=ext_id, group_name=group, role=role)
            s.add(person)
            s.flush()

        # Replace old sample for this pose instead of stacking indefinitely
        if pose:
            old = s.execute(
                select(DB.FaceSample).where(
                    DB.FaceSample.person_id == person.id,
                    DB.FaceSample.pose_label == pose,
                )
            ).scalars().all()
            for o in old:
                s.delete(o)

        DB.add_face_sample(s, person, n, pose_label=pose)
        DB.store_embedding(person, n)
        s.commit()
        total_samples = DB.count_samples(s, person.id)
        for info in PIPELINE.tracker._infos.values():
            if info.person_id is None:
                info.recognition_attempts = 0
                info.last_recognition_frame = -10**9
        return jsonify({
            "ok": True, "person": _person_to_dict(person),
            "total_samples": total_samples,
            "pose": pose,
            "detected_pose": detected_pose,
            "is_update": existing is not None,
        })


# ----- API: dashboard data --------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    sid = request.args.get("session_id", type=int) or active_session_id()
    if not sid:
        return jsonify({"error": "no active session"}), 404
    with DB.SessionLocal() as s:
        sess = s.get(DB.Session_, sid)
        if not sess:
            return jsonify({"error": "not found"}), 404
        attendances = s.execute(
            select(DB.Attendance).where(DB.Attendance.session_id == sid)
        ).scalars().all()
        att_by_pid = {a.person_id: a for a in attendances}

        member_ids = DB.get_session_member_ids(s, sid)
        if member_ids:
            att_pids = list(att_by_pid.keys())
            all_ids = list(set(member_ids) | set(att_pids))
            people_q = select(DB.Person).where(DB.Person.id.in_(all_ids))
        elif sess.group_name:
            att_pids = list(att_by_pid.keys())
            if att_pids:
                people_q = select(DB.Person).where(
                    or_(DB.Person.group_name == sess.group_name,
                        DB.Person.id.in_(att_pids))
                )
            else:
                people_q = select(DB.Person).where(DB.Person.group_name == sess.group_name)
        else:
            people_q = select(DB.Person)
        people = s.execute(people_q).scalars().all()

        present, late, absent = 0, 0, 0
        rows = []
        for p in people:
            a = att_by_pid.get(p.id)
            if a is None:
                status = "absent"
                absent += 1
            elif a.status == "late":
                status = "late"; late += 1
            else:
                status = "present"; present += 1
            rows.append({
                "person_id": p.id, "name": p.name, "external_id": p.external_id,
                "status": status,
                "arrival_ts": a.arrival_ts.isoformat() if a else None,
                "departure_ts": a.departure_ts.isoformat() if a and a.departure_ts else None,
                "hours": (
                    round((a.departure_ts - a.arrival_ts).total_seconds() / 3600, 2)
                    if a and a.departure_ts else None
                ),
                "thumbnail": (url_for("static", filename=p.thumbnail_path) if p.thumbnail_path else None),
            })

        # Hourly arrivals histogram — scoped to session window
        start_h = sess.start_time.hour
        if sess.duration_minutes:
            end_dt = datetime.combine(sess.date, sess.start_time) + timedelta(minutes=sess.duration_minutes)
            end_h = min(end_dt.hour + 1, 24)
        else:
            end_h = max((a.arrival_ts.hour + 1 for a in attendances), default=start_h + 1)
            end_h = max(end_h, datetime.now().hour + 1) if sess.date == Date.today() else end_h
        end_h = min(max(end_h, start_h + 1), 24)
        hourly_range = list(range(start_h, end_h))
        hourly = {h: 0 for h in hourly_range}
        for a in attendances:
            h = a.arrival_ts.hour
            if h in hourly:
                hourly[h] += 1

        # Unknowns count for this session
        unk_count = s.execute(
            select(func.count(DB.Unknown.id)).where(DB.Unknown.session_id == sid)
        ).scalar() or 0

        return jsonify({
            "session": _session_to_dict(sess),
            "counts": {"present": present, "late": late, "absent": absent, "total": len(people)},
            "rows": rows,
            "hourly_labels": [f"{h}:00" for h in hourly_range],
            "hourly": list(hourly.values()),
            "unknown_count": int(unk_count),
        })


@app.route("/api/dashboard/export")
def api_dashboard_export():
    sid = request.args.get("session_id", type=int) or active_session_id()
    if not sid:
        return Response("no active session", 404)
    with DB.SessionLocal() as s:
        sess = s.get(DB.Session_, sid)
        if not sess:
            return Response("not found", 404)
        att_rows = s.execute(
            select(DB.Attendance).where(DB.Attendance.session_id == sid)).scalars().all()
        att_by_pid = {a.person_id: a for a in att_rows}
        member_ids = DB.get_session_member_ids(s, sid)
        if member_ids:
            all_ids = list(set(member_ids) | set(att_by_pid.keys()))
            people_q = select(DB.Person).where(DB.Person.id.in_(all_ids))
        elif sess.group_name:
            att_pids = list(att_by_pid.keys())
            if att_pids:
                people_q = select(DB.Person).where(
                    or_(DB.Person.group_name == sess.group_name,
                        DB.Person.id.in_(att_pids))
                )
            else:
                people_q = select(DB.Person).where(DB.Person.group_name == sess.group_name)
        else:
            people_q = select(DB.Person)
        people = s.execute(people_q).scalars().all()

        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["external_id", "name", "group", "role", "status", "arrival_ts", "departure_ts", "hours"])
        for p in people:
            a = att_by_pid.get(p.id)
            status = "absent" if a is None else a.status
            arr = a.arrival_ts.isoformat() if a else ""
            dep = a.departure_ts.isoformat() if a and a.departure_ts else ""
            hrs = (
                round((a.departure_ts - a.arrival_ts).total_seconds() / 3600, 2)
                if a and a.departure_ts else ""
            )
            w.writerow([p.external_id, p.name, p.group_name or "", p.role, status, arr, dep, hrs])

        data = out.getvalue().encode("utf-8")
        fname = f"attendance_{sess.name}_{sess.date.isoformat()}.csv"
        return Response(data, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})


# ----- API: person history --------------------------------------------------

@app.route("/api/person/<int:pid>")
def api_person(pid: int):
    with DB.SessionLocal() as s:
        p = s.get(DB.Person, pid)
        if not p:
            return jsonify({"error": "not found"}), 404
        # last 30 days, joined with session for the date
        cutoff = Date.today() - timedelta(days=30)
        rows = s.execute(
            select(DB.Attendance, DB.Session_).join(DB.Session_)
            .where(DB.Attendance.person_id == pid, DB.Session_.date >= cutoff)
            .order_by(DB.Session_.date.desc())
        ).all()
        history = []
        by_date: dict[str, str] = {}
        for a, sess in rows:
            d = sess.date.isoformat()
            history.append({
                "date": d, "session": sess.name, "status": a.status,
                "arrival_ts": a.arrival_ts.isoformat(),
                "departure_ts": a.departure_ts.isoformat() if a.departure_ts else None,
            })
            # priority for the calendar tile color: late beats present
            if d not in by_date or by_date[d] == "present":
                by_date[d] = a.status
        # build last 30 days grid
        grid = []
        for i in range(29, -1, -1):
            day = Date.today() - timedelta(days=i)
            grid.append({"date": day.isoformat(), "status": by_date.get(day.isoformat(), "none")})
        # attendance %
        sessions_total = s.execute(
            select(func.count(DB.Session_.id)).where(DB.Session_.date >= cutoff)
        ).scalar() or 0
        present_count = sum(1 for h in history if h["status"] in ("present", "late"))
        pct = round(100 * present_count / sessions_total, 1) if sessions_total else 0.0
        return jsonify({
            "person": _person_to_dict(p),
            "history": history,
            "grid": grid,
            "attendance_pct": pct,
            "sessions_total": int(sessions_total),
        })


# ----- API: unknowns --------------------------------------------------------

@app.route("/api/unknowns")
def api_unknowns():
    with DB.SessionLocal() as s:
        rows = s.execute(
            select(DB.Unknown).order_by(DB.Unknown.seen_ts.desc()).limit(60)
        ).scalars().all()
        return jsonify([
            {"id": u.id, "seen_ts": u.seen_ts.isoformat(),
             "thumbnail": (url_for("static", filename=u.thumbnail_path) if u.thumbnail_path else None),
             "session_id": u.session_id}
            for u in rows
        ])


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "ts": datetime.now().isoformat()})


@app.route("/api/live/snapshot")
def api_live_snapshot():
    """Return the current annotated frame as a downloadable JPEG.
    Uses the cached frame from the MJPEG generator if /live is open;
    otherwise grabs + annotates one frame on demand."""
    data = PIPELINE.last_jpeg
    if data is None:
        if not PIPELINE.ensure_camera():
            return Response("camera not available", 503)
        frame, _ = PIPELINE.camera.read()
        if frame is None:
            return Response("no frame from camera", 503)
        with PIPELINE.lock:
            annotated = PIPELINE.annotate_frame(frame, active_session_id())
        data = PIPELINE.cache_jpeg(annotated, quality=90)
        if data is None:
            return Response("encode failed", 500)
    fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    return Response(data, mimetype="image/jpeg",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/api/thumbs/wipe", methods=["POST"])
def api_thumbs_wipe():
    """Delete every JPEG in static/thumbs/ and null every thumbnail_path column.
    Embeddings (matrices) are untouched — recognition keeps working."""
    deleted_files = 0
    for f in THUMBS_DIR.glob("*"):
        try:
            f.unlink(); deleted_files += 1
        except OSError:
            pass
    with DB.SessionLocal() as s:
        cleared = 0
        for p in s.execute(select(DB.Person)).scalars():
            if p.thumbnail_path:
                p.thumbnail_path = None; cleared += 1
        for u in s.execute(select(DB.Unknown)).scalars():
            if u.thumbnail_path:
                u.thumbnail_path = None; cleared += 1
        s.commit()
    return jsonify({"deleted_files": deleted_files, "cleared_rows": cleared,
                    "store_thumbnails": STORE_THUMBNAILS})


# ----- Session edit / delete ----------------------------------------------

@app.route("/api/sessions/<int:sid>", methods=["DELETE", "PATCH"])
def api_session_modify(sid: int):
    if request.method == "DELETE":
        with DB.SessionLocal() as s:
            sess = s.get(DB.Session_, sid)
            if not sess:
                return jsonify({"error": "not found"}), 404
            # If this is the active session, clear it so /live and /dashboard
            # don't keep referencing a deleted row.
            active = DB.get_config(s, "active_session_id")
            if active and active.isdigit() and int(active) == sid:
                DB.del_config(s, "active_session_id")
            s.delete(sess)
            s.commit()
        return jsonify({"ok": True})

    body = request.json or {}
    with DB.SessionLocal() as s:
        sess = s.get(DB.Session_, sid)
        if not sess:
            return jsonify({"error": "not found"}), 404
        if "name" in body and body["name"]:
            sess.name = body["name"].strip()
        if "mode" in body and body["mode"] in ("student", "worker"):
            sess.mode = body["mode"]
        if "group_name" in body:
            sess.group_name = body["group_name"] or None
        if "start_time" in body and body["start_time"]:
            from datetime import time as Time
            h, m = [int(x) for x in body["start_time"].split(":")]
            sess.start_time = Time(h, m)
        if "late_threshold_minutes" in body and body["late_threshold_minutes"] not in (None, ""):
            sess.late_threshold_minutes = int(body["late_threshold_minutes"])
        if "duration_minutes" in body:
            dur = body["duration_minutes"]
            sess.duration_minutes = int(dur) if dur not in (None, "", 0, "0") else None
        if "session_date" in body and body["session_date"]:
            from datetime import date as Date_
            sess.date = Date_.fromisoformat(body["session_date"])
        if "member_ids" in body:
            DB.set_session_members(s, sid, [int(x) for x in (body["member_ids"] or [])])
        s.commit()
        s.refresh(sess)
        mids = DB.get_session_member_ids(s, sid)
        return jsonify(_session_to_dict(sess, member_ids=mids))


# ----- Phone IP cam: front/back toggle -------------------------------------

@app.route("/api/camera/toggle_front", methods=["POST"])
def api_camera_toggle_front():
    """Toggle the phone's front/back camera. Server tracks the state so the
    frontend never gets out of sync."""
    cam = PIPELINE.camera
    if not cam or not isinstance(cam.source, str) or not cam.source.startswith(("http://", "https://")):
        return jsonify({"error": "Only works when the camera is a phone IP-cam URL."}), 400
    use_front = not PIPELINE.ffc_front
    from urllib.parse import urlparse
    import urllib.request
    u = urlparse(cam.source)
    base = f"{u.scheme}://{u.netloc}"
    target = f"{base}/settings/ffc?set={'on' if use_front else 'off'}"
    try:
        with urllib.request.urlopen(target, timeout=3) as r:
            r.read()
    except Exception as e:
        return jsonify({"error": f"Phone didn't respond at {target}: {e}"}), 502
    PIPELINE.ffc_front = use_front
    rot = 270 if use_front else 90
    was_processing = PIPELINE.is_processing
    if was_processing:
        PIPELINE.stop_processing()
    with DB.SessionLocal() as s:
        DB.set_config(s, "camera_rotation", str(rot))
    PIPELINE.set_camera(cam.source, rotation=rot)
    if was_processing or active_session_id() is not None:
        PIPELINE.start_processing()
    return jsonify({"ok": True, "front": use_front, "rotation": rot})


# ============================================================ entry point ===

def _startup():
    info = DB.init_db()
    log.info("DB ready: %s", info)
    rec_info = REC.init(prefer=os.environ.get("RECOGNITION"))
    PIPELINE.recognition_path = rec_info
    if "MATCH_THRESHOLD" not in os.environ:
        thresholds = {"insightface": 0.45, "facenet": 0.38, "dlib": 0.40, "lbp": 0.40}
        PIPELINE.match_threshold = thresholds.get(rec_info["backend"], 0.50)
        log.info("Auto-set match_threshold=%.2f for %s backend", PIPELINE.match_threshold, rec_info["backend"])


if __name__ == "__main__":
    _startup()
    if active_session_id() is not None:
        log.info("Resuming active session — starting background processing")
        PIPELINE.start_processing()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # threaded=True so the MJPEG generator and other requests can run concurrently
    app.run(host=host, port=port, threaded=True, debug=False)
