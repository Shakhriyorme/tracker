"""Bu — SORT tracking’ni upgrade qilgan system. 
Ya’ni objectni shunchaki kuzatmaydi, balki kimligini ham eslab qoladi. 
Endi system ham ‘bu kim edi?’ deb o'ylanib qolmaydi."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sort_tracker import Sort

@dataclass
class TrackInfo:
    track_id: int
    bbox: tuple[int, int, int, int]
    person_id: Optional[int] = None
    name: Optional[str] = None
    last_recognition_frame: int = -10**9
    recognition_attempts: int = 0

class TrackerWithIdentity:
    def __init__(self, recognition_cooldown_frames: int = 5, max_attempts: int = 12):
        self.sort = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        self._infos: dict[int, TrackInfo] = {}
        self.cooldown = recognition_cooldown_frames
        self.max_attempts = max_attempts
        self.frame_idx = 0

    def step(self, person_dets_xyxy_conf: np.ndarray) -> list[TrackInfo]:
        self.frame_idx += 1
        tracks = self.sort.update(person_dets_xyxy_conf)
        active_ids = set()
        out = []
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            active_ids.add(tid)
            box = (int(x1), int(y1), int(x2), int(y2))
            info = self._infos.get(tid)
            if info is None:
                info = TrackInfo(track_id=tid, bbox=box)
                self._infos[tid] = info
            else:
                info.bbox = box
            out.append(info)
        # Garbage collection for dropped tracks
        for dead in [t for t in self._infos if t not in active_ids]:
            self._infos.pop(dead, None)
        return out

    def needs_recognition(self, info: TrackInfo) -> bool:
        if info.name is not None or info.recognition_attempts >= self.max_attempts:
            return False
        return (self.frame_idx - info.last_recognition_frame) >= self.cooldown

    def mark_attempt(self, info: TrackInfo) -> None:
        info.last_recognition_frame = self.frame_idx
        info.recognition_attempts += 1

    def assign_identity(self, info: TrackInfo, person_id: int, name: str) -> None:
        info.person_id = person_id
        info.name = name