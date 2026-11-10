"""Object tracking: gives detections stable IDs across frames and reports what happens to them."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .detector import Detection
from .events import EVENT_APPEARED, EVENT_DISAPPEARED, EVENT_MOVING, EVENT_STOPPED, Event
from .states import format_duration

CONFIRM_HITS = 2  # one-frame flickers never become objects
MAX_OBJECTS_PER_RULE = 200  # a badly tuned rule must not stall the program; largest blobs win
TRAIL_POINTS = 64
SPEED_WINDOW = 0.5  # seconds; measuring over a window ignores centroid jitter between frames
SPEED_MIN_SPAN = 0.2
MOVING_ABOVE = 15.0  # pixels per second; the gap between the limits prevents flapping
STOPPED_BELOW = 5.0


@dataclass(frozen=True)
class Track:
    id: int
    rule: str
    x: int
    y: int
    width: int
    height: int
    center: tuple[int, int]
    score: float | None
    age: float  # seconds since first seen
    speed: float  # pixels per second
    moving: bool
    visible: bool  # False while briefly unseen, before it counts as gone
    trail: tuple[tuple[int, int], ...]


class _Live:
    def __init__(self, track_id: int, detection: Detection, now: float):
        self.id = track_id
        self.detection = detection
        self.first_seen = self.last_seen = now
        self.hits = 1
        self.confirmed = False
        self.moving = False
        self.speed = 0.0
        self.points: deque[tuple[float, int, int]] = deque([(now, *detection.center)], maxlen=TRAIL_POINTS)

    def see(self, detection: Detection, now: float) -> None:
        self.detection = detection
        self.last_seen = now
        self.hits += 1
        self.points.append((now, *detection.center))
        self.speed = self._measure_speed(now)

    def _measure_speed(self, now: float) -> float:
        start = next((p for p in self.points if now - p[0] <= SPEED_WINDOW), None)
        if start is None or now - start[0] < SPEED_MIN_SPAN:
            return self.speed
        _, x, y = self.points[-1]
        return math.hypot(x - start[1], y - start[2]) / (now - start[0])


class ObjectTracker:
    """Each rule's detections are matched to its existing objects, nearest pairs first."""

    def __init__(self):
        self._live: list[_Live] = []
        self._next_id = 1
        self._now = 0.0

    def update(
        self, detections: Mapping[str, Sequence[Detection]], now: float, max_distance: float, lost_seconds: float
    ) -> tuple[list[Track], list[Event]]:
        self._now = now
        events: list[Event] = []
        wall = time.time()
        # A rule that reports nothing at all was disabled or removed: nothing "disappeared" in the game.
        self._live = [live for live in self._live if live.detection.rule in detections]

        for rule, found in detections.items():
            self._match_rule(rule, list(found)[:MAX_OBJECTS_PER_RULE], now, max_distance, wall, events)

        kept: list[_Live] = []
        for live in self._live:
            if now - live.last_seen > lost_seconds:
                if live.confirmed:
                    name = f"{live.detection.rule} #{live.id}"
                    message = f"{name} zniknął po {format_duration(live.last_seen - live.first_seen)}"
                    events.append(Event(EVENT_DISAPPEARED, live.detection.rule, message, wall, live.id))
            else:
                kept.append(live)
        self._live = kept
        return self.snapshot(), events

    def _match_rule(self, rule: str, found: list[Detection], now: float, max_distance: float, wall: float, events: list[Event]) -> None:
        candidates = [live for live in self._live if live.detection.rule == rule]
        used_found: set[int] = set()
        used_live: set[int] = set()
        if found and candidates:
            found_xy = np.array([d.center for d in found], float)
            live_xy = np.array([live.detection.center for live in candidates], float)
            distances = np.linalg.norm(found_xy[:, None, :] - live_xy[None, :, :], axis=2)
            for index in np.argsort(distances, axis=None):
                f, l = np.unravel_index(index, distances.shape)
                if distances[f, l] > max_distance:
                    break
                if f in used_found or l in used_live:
                    continue
                used_found.add(int(f))
                used_live.add(int(l))
                self._see(candidates[l], found[f], now, wall, events)
        for index, detection in enumerate(found):
            if index not in used_found:
                self._live.append(_Live(self._next_id, detection, now))
                self._next_id += 1

    @staticmethod
    def _see(live: _Live, detection: Detection, now: float, wall: float, events: list[Event]) -> None:
        live.see(detection, now)
        name = f"{detection.rule} #{live.id}"
        if not live.confirmed and live.hits >= CONFIRM_HITS:
            live.confirmed = True
            events.append(Event(EVENT_APPEARED, detection.rule, f"{name} pojawił się", wall, live.id))
        if not live.confirmed:
            return
        if not live.moving and live.speed >= MOVING_ABOVE:
            live.moving = True
            events.append(Event(EVENT_MOVING, detection.rule, f"{name} zaczął się ruszać", wall, live.id))
        elif live.moving and live.speed <= STOPPED_BELOW:
            live.moving = False
            events.append(Event(EVENT_STOPPED, detection.rule, f"{name} zatrzymał się", wall, live.id))

    def snapshot(self) -> list[Track]:
        """Confirmed objects, oldest first."""
        return [
            Track(
                id=live.id,
                rule=live.detection.rule,
                x=live.detection.x,
                y=live.detection.y,
                width=live.detection.width,
                height=live.detection.height,
                center=live.detection.center,
                score=live.detection.score,
                age=self._now - live.first_seen,
                speed=live.speed,
                moving=live.moving,
                visible=live.last_seen == self._now,
                trail=tuple((x, y) for _, x, y in live.points),
            )
            for live in self._live
            if live.confirmed
        ]
