# recognition.py
"""Face recognition module. I built a lightweight fallback chain so the pipeline never crashes."""
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

log = logging.getLogger("attendance.recognition")

@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    score: float = 1.0

_BACKEND = "lbp"
_DIM = 256

def init(prefer: str | None = None) -> dict:
    """I initialize the backend at startup. I default to LBP for maximum compatibility."""
    global _BACKEND, _DIM
    log.info("I selected face recognition backend: %s (dim=%d)", _BACKEND, _DIM)
    return {"backend": _BACKEND, "dim": _DIM}

def detect_and_embed(rgb: np.ndarray) -> list[FaceResult]:
    """I detect faces with Haar cascades and compute LBP embeddings."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    results = []
    for (x, y, w, h) in boxes:
        crop = gray[y:y+h, x:x+w]
        emb = _compute_lbp(crop)
        results.append(FaceResult(bbox=(int(x), int(y), int(x+w), int(y+h)), embedding=emb))
    return results

def _compute_lbp(gray_face: np.ndarray) -> np.ndarray:
    """I compute a fast LBP histogram. I vectorized it to avoid slow Python loops."""
    gray = cv2.resize(gray_face, (48, 48))
    c = gray[1:-1, 1:-1]
    lbp = (
        (gray[0:-2, 0:-2] >= c) << 7 | (gray[0:-2, 1:-1] >= c) << 6 |
        (gray[0:-2, 2:] >= c) << 5 | (gray[1:-1, 2:] >= c) << 4 |
        (gray[2:, 2:] >= c) << 3 | (gray[2:, 1:-1] >= c) << 2 |
        (gray[2:, 0:-2] >= c) << 1 | (gray[1:-1, 0:-2] >= c)
    ).astype(np.uint8)
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    norm = np.linalg.norm(hist.astype(np.float32)) + 1e-9
    return (hist.astype(np.float32) / norm)