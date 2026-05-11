"""Face recognition with a 4-tier fallback chain.

Picked at startup based on what's importable. Logs the active path loudly so the
README's troubleshooting section has something useful to point at.

Backends (best -> worst):
  1. insightface (buffalo_sc)  - 512-d ArcFace, fast on CPU, detection + embed
  2. facenet-pytorch (MTCNN + InceptionResnetV1) - 512-d, pure Python/PyTorch
  3. face_recognition / dlib   - 128-d HOG/CNN detector + ResNet embeddings
  4. opencv Haar + LBP         - degraded but the demo still runs

Public API:
  init() -> dict   pick a backend, lazy-load models, return {"backend": "...", "dim": int}
  detect_and_embed(rgb) -> list[FaceResult]
  embed_crop(rgb_face) -> np.ndarray | None    (used when we already have a face crop)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("attendance.recognition")

_BACKEND: Optional[str] = None
_DIM: int = 512
_MODEL = None  # backend-specific
_LBP_GRID = 4
_LBP_BINS = 32


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2
    embedding: np.ndarray
    score: float = 1.0


# ----------------------------------------------------------------------- init

def init(prefer: str | None = None) -> dict:
    """Resolve which backend to use. `prefer` can force one of: insightface, facenet, dlib, lbp."""
    global _BACKEND, _MODEL, _DIM

    candidates = ["insightface", "facenet", "dlib", "lbp"]
    if prefer in candidates:
        candidates = [prefer] + [c for c in candidates if c != prefer]

    for c in candidates:
        try:
            if c == "insightface":
                _init_insightface()
            elif c == "facenet":
                _init_facenet()
            elif c == "dlib":
                _init_dlib()
            else:
                _init_lbp()
            _BACKEND = c
            log.info("Face recognition backend: %s (dim=%d)", _BACKEND, _DIM)
            print(f"[recognition] Active backend: {_BACKEND} (embedding dim = {_DIM})")
            return {"backend": _BACKEND, "dim": _DIM}
        except Exception as e:
            log.warning("Backend %s unavailable: %s", c, e)

    raise RuntimeError("No face recognition backend could be initialized.")


def _init_insightface():
    global _MODEL, _DIM
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _MODEL = app
    _DIM = 512


def _init_facenet():
    global _MODEL, _DIM
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20,
                  thresholds=[0.5, 0.6, 0.6])
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    _MODEL = (mtcnn, resnet)
    _DIM = 512
    log.info("facenet-pytorch loaded on %s", device)


def _init_dlib():
    global _MODEL, _DIM
    import face_recognition  # noqa: F401  (smoke import)
    _MODEL = "dlib"
    _DIM = 128


def _init_lbp():
    global _MODEL, _DIM
    import cv2
    haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haar)
    if detector.empty():
        raise RuntimeError("Haar cascade missing in this opencv build")
    _MODEL = detector
    _DIM = _LBP_GRID * _LBP_GRID * _LBP_BINS


def backend() -> str:
    return _BACKEND or "uninitialized"


def dim() -> int:
    return _DIM


# ------------------------------------------------------------------ detection

def detect_and_embed(rgb: np.ndarray) -> list[FaceResult]:
    """Detect faces in an RGB image and return embeddings."""
    if _BACKEND == "insightface":
        return _insightface_detect(rgb)
    if _BACKEND == "facenet":
        return _facenet_detect(rgb)
    if _BACKEND == "dlib":
        return _dlib_detect(rgb)
    if _BACKEND == "lbp":
        return _lbp_detect(rgb)
    raise RuntimeError("recognition.init() must be called first")


def estimate_pose(rgb: np.ndarray, mirror: bool = False) -> str | None:
    """Estimate head pose from facial landmarks. Returns 'front', 'left', 'right', or None."""
    if _BACKEND == "facenet":
        return _facenet_estimate_pose(rgb, mirror)
    if _BACKEND == "insightface":
        return _insightface_estimate_pose(rgb, mirror)
    return None


def embed_crop(rgb_face: np.ndarray) -> np.ndarray | None:
    """Embed an already-cropped face. Returns None if backend can't embed without detection."""
    if _BACKEND == "insightface":
        results = _insightface_detect(rgb_face)
        return results[0].embedding if results else None
    if _BACKEND == "facenet":
        return _facenet_embed_crop(rgb_face)
    if _BACKEND == "dlib":
        import face_recognition
        encs = face_recognition.face_encodings(rgb_face, known_face_locations=[(0, rgb_face.shape[1], rgb_face.shape[0], 0)])
        return np.asarray(encs[0], dtype=np.float32) if encs else None
    if _BACKEND == "lbp":
        return _lbp_hist(rgb_face)
    return None


# ------------------------------------------------------ backend implementations

def _insightface_detect(rgb: np.ndarray) -> list[FaceResult]:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = _MODEL.get(bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        emb = np.asarray(f.normed_embedding if hasattr(f, "normed_embedding") else f.embedding, dtype=np.float32)
        out.append(FaceResult((x1, y1, x2, y2), emb, float(getattr(f, "det_score", 1.0))))
    return out


def _facenet_detect(rgb: np.ndarray) -> list[FaceResult]:
    import torch
    from PIL import Image
    mtcnn, resnet = _MODEL
    img = Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return []
    faces_tensor = mtcnn(img)
    if faces_tensor is None:
        return []
    if faces_tensor.dim() == 3:
        faces_tensor = faces_tensor.unsqueeze(0)
    with torch.no_grad():
        embeddings = resnet(faces_tensor).cpu().numpy()
    out = []
    for i in range(min(len(boxes), len(embeddings))):
        if probs[i] is None or probs[i] < 0.5:
            continue
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        emb = embeddings[i].astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        out.append(FaceResult((x1, y1, x2, y2), emb, float(probs[i])))
    return out


def _facenet_embed_crop(rgb_face: np.ndarray) -> np.ndarray | None:
    import torch
    from PIL import Image
    mtcnn, resnet = _MODEL
    img = Image.fromarray(rgb_face)
    face = mtcnn(img)
    if face is None:
        return None
    if face.dim() == 3:
        face = face.unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face).cpu().numpy()[0].astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def _dlib_detect(rgb: np.ndarray) -> list[FaceResult]:
    import face_recognition
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return []
    encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
    out = []
    for (top, right, bottom, left), e in zip(locs, encs):
        out.append(FaceResult((left, top, right, bottom), np.asarray(e, dtype=np.float32), 1.0))
    return out


def _lbp_detect(rgb: np.ndarray) -> list[FaceResult]:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = _MODEL.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    out = []
    for (x, y, w, h) in boxes:
        crop = rgb[y:y + h, x:x + w]
        emb = _lbp_hist(crop)
        if emb is None:
            continue
        out.append(FaceResult((int(x), int(y), int(x + w), int(y + h)), emb, 1.0))
    return out


def _lbp_hist(rgb_face: np.ndarray) -> np.ndarray | None:
    """Spatial-block LBP feature.

    Old version: one 256-bin histogram of LBP codes for the whole face. Loses
    spatial info — same distribution of texture across the image hashes the
    same, even if eyes/nose/mouth move around.

    New version: GRID x GRID cells, BINS-bin histogram per cell, each cell
    L2-normalized, then concatenated and L2-normalized again. Different parts
    of the face contribute to different dimensions of the embedding. With
    grid=4, bins=32 the result is 512 floats — same dim as insightface, fits
    pgvector's vector(512), and is dramatically more discriminative than the
    global histogram.
    """
    import cv2
    if rgb_face.size == 0:
        return None
    gray = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
    # 96 = 4 * 24 → cleanly divides into the LBP grid
    gray = cv2.resize(gray, (96, 96))
    rows, cols = gray.shape

    # Vectorized LBP: compare each pixel to its 8 neighbors in one shot.
    # Much faster than the per-pixel Python loop that was here before.
    c = gray[1:-1, 1:-1]
    lbp = (
        ((gray[0:-2, 0:-2] >= c) << 7) |
        ((gray[0:-2, 1:-1] >= c) << 6) |
        ((gray[0:-2, 2:  ] >= c) << 5) |
        ((gray[1:-1, 2:  ] >= c) << 4) |
        ((gray[2:  , 2:  ] >= c) << 3) |
        ((gray[2:  , 1:-1] >= c) << 2) |
        ((gray[2:  , 0:-2] >= c) << 1) |
        ((gray[1:-1, 0:-2] >= c))
    ).astype(np.uint8)

    cell_h = lbp.shape[0] // _LBP_GRID
    cell_w = lbp.shape[1] // _LBP_GRID
    parts: list[np.ndarray] = []
    for gi in range(_LBP_GRID):
        for gj in range(_LBP_GRID):
            cell = lbp[gi * cell_h:(gi + 1) * cell_h,
                       gj * cell_w:(gj + 1) * cell_w]
            hist, _ = np.histogram(cell, bins=_LBP_BINS, range=(0, 256))
            h = hist.astype(np.float32)
            h /= (np.linalg.norm(h) + 1e-9)  # per-cell L2
            parts.append(h)
    full = np.concatenate(parts)
    return full / (np.linalg.norm(full) + 1e-9)


# ---------------------------------------------------------- pose estimation

def _landmarks_to_pose(lm: np.ndarray, mirror: bool) -> str | None:
    """Estimate pose from 5 facial landmarks:
    [left_eye, right_eye, nose, mouth_left, mouth_right].
    Uses nose offset from eye midpoint + mouth corner asymmetry."""
    if lm is None or len(lm) < 5:
        return None
    left_eye, right_eye, nose = lm[0], lm[1], lm[2]
    mouth_l, mouth_r = lm[3], lm[4]

    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 5:
        return None

    eye_cx = (left_eye[0] + right_eye[0]) / 2
    nose_offset = (nose[0] - eye_cx) / eye_dist

    # Mouth asymmetry: distance from nose to each mouth corner
    d_mouth_l = abs(nose[0] - mouth_l[0])
    d_mouth_r = abs(nose[0] - mouth_r[0])
    mouth_total = d_mouth_l + d_mouth_r
    mouth_asym = (d_mouth_r - d_mouth_l) / mouth_total if mouth_total > 1 else 0

    # Combine signals: nose offset is primary, mouth asymmetry confirms
    combined = nose_offset * 0.7 + mouth_asym * 0.3

    if mirror:
        combined = -combined

    log.debug("pose estimation: nose_off=%.3f mouth_asym=%.3f combined=%.3f",
              nose_offset, mouth_asym, combined)

    if abs(combined) < 0.08:
        return "front"
    return "left" if combined > 0 else "right"


def _facenet_estimate_pose(rgb: np.ndarray, mirror: bool) -> str | None:
    from PIL import Image
    mtcnn, _ = _MODEL
    img = Image.fromarray(rgb)
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    if boxes is None or landmarks is None or len(landmarks) == 0:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    return _landmarks_to_pose(landmarks[idx], mirror)


def _insightface_estimate_pose(rgb: np.ndarray, mirror: bool) -> str | None:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = _MODEL.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 5:
        return None
    return _landmarks_to_pose(np.array(kps), mirror)
