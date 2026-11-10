"""Detection history: recent counts per rule, statistics, the event log and CSV export."""
from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .events import EVENT_APPEARED, Event

MAX_SAMPLES = 3000  # about 10 minutes per rule at 5 samples per second
MAX_EVENTS = 2000


@dataclass(frozen=True)
class RuleStats:
    current: int
    average: float
    maximum: int
    mean_score: float | None
    total_objects: int  # distinct tracked objects since the history was last cleared


def _format_time(wall: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall))


class DetectionHistory:
    def __init__(self):
        self._samples: dict[str, deque[tuple[float, int, float | None]]] = {}
        self._totals: dict[str, int] = {}
        self.events: deque[Event] = deque(maxlen=MAX_EVENTS)

    def add_sample(self, wall: float, counts: dict[str, tuple[int, float | None]]) -> None:
        for rule, (count, score) in counts.items():
            self._samples.setdefault(rule, deque(maxlen=MAX_SAMPLES)).append((wall, count, score))

    def add_event(self, event: Event) -> None:
        self.events.append(event)
        if event.kind == EVENT_APPEARED:
            self._totals[event.source] = self._totals.get(event.source, 0) + 1

    def rules(self) -> list[str]:
        return list(self._samples)

    def series(self, rule: str, window: float, now: float) -> list[tuple[float, int]]:
        """(seconds ago, count) pairs from the last `window` seconds, oldest first."""
        return [(now - wall, count) for wall, count, _ in self._samples.get(rule, ()) if now - wall <= window]

    def stats(self, rule: str, window: float, now: float) -> RuleStats:
        recent = [s for s in self._samples.get(rule, ()) if now - s[0] <= window]
        counts = [count for _, count, _ in recent]
        scores = [score for _, _, score in recent if score is not None]
        return RuleStats(
            current=counts[-1] if counts else 0,
            average=sum(counts) / len(counts) if counts else 0.0,
            maximum=max(counts, default=0),
            mean_score=sum(scores) / len(scores) if scores else None,
            total_objects=self._totals.get(rule, 0),
        )

    def clear(self) -> None:
        self._samples.clear()
        self._totals.clear()
        self.events.clear()

    # Semicolons and whole percents open correctly in a Polish Excel; the BOM keeps diacritics.
    def export_samples_csv(self, path: str | Path) -> int:
        rows = sorted(
            (wall, rule, count, score)
            for rule, samples in self._samples.items()
            for wall, count, score in samples
        )
        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["czas", "regula", "liczba", "pewnosc_proc"])
            for wall, rule, count, score in rows:
                writer.writerow([_format_time(wall), rule, count, "" if score is None else round(score * 100)])
        return len(rows)

    def export_events_csv(self, path: str | Path) -> int:
        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["czas", "typ", "zrodlo", "id_obiektu", "opis"])
            for event in self.events:
                track = "" if event.track_id is None else event.track_id
                writer.writerow([_format_time(event.timestamp), event.kind, event.source, track, event.message])
        return len(self.events)
