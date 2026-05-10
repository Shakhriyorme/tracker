"""Tracker with identity management wrapper around SORT.
We manage per-track cooldowns, attempt limits, and identity assignment here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sort_tracker import Sort

@dataclass
class TrackInfo:
    """We store track state, identity, and recognition metrics."""
    track_id: int
    bbox: tuple[int, int, int, int]
    person_id: Optional[int] = None
    name: Optional[str] = None
    last_recognition_frame: int = -10**9
    recognition_attempts: int = 0

class TrackerWithIdentity:
    """We wrap SORT to add identity caching and recognition throttling."""
    def __init__(self, recognition_cooldown_frames: int = 5, max_attempts: int = 12):
        # We initialize SORT with stable tracking parameters.
        self.sort = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        self._infos: dict[int, TrackInfo] = {}
        self.cooldown = recognition_cooldown_frames
        self.max_attempts = max_attempts
        self.frame_idx = 0

    def step(self, person_dets_xyxy_conf: np.ndarray) -> list[TrackInfo]:
        """We advance the tracker by one frame and return active tracks."""
        self.frame_idx += 1
        tracks = self.sort.update(person_dets_xyxy_conf)

        active_ids: set[int] = set()
        out: list[TrackInfo] = []
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

        # We clean up tracks that SORT has dropped to prevent memory leaks.
        for dead in [t for t in self._infos if t not in active_ids]:
            self._infos.pop(dead, None)
        return out

    def needs_recognition(self, info: TrackInfo) -> bool:
        """We check if a track is eligible for face recognition this frame."""
        # We skip if already identified or if max attempts reached.
        if info.name is not None:
            return False
        if info.recognition_attempts >= self.max_attempts:
            return False
        # We enforce cooldown to avoid CPU overload.
        return (self.frame_idx - info.last_recognition_frame) >= self.cooldown

    def mark_attempt(self, info: TrackInfo) -> None:
        """We log a recognition attempt for this track."""
        info.last_recognition_frame = self.frame_idx
        info.recognition_attempts += 1

    def assign_identity(self, info: TrackInfo, person_id: int, name: str) -> None:
        """We attach a person's identity to a track permanently."""
        info.person_id = person_id
        info.name = name