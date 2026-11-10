"""Debug overlay drawn on the preview: object IDs, confidence, trails, regions and timings."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from .plugin import FrameResult
from .profile import Profile

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.36
LINE_HEIGHT = 13
PANEL_RULE_LINES = 6
_POLISH = str.maketrans({"ł": "l", "Ł": "L"})


@dataclass(frozen=True)
class FrameTimings:
    capture_ms: float = 0.0
    detect_ms: float = 0.0
    plugins_ms: float = 0.0


def ascii_text(text: str) -> str:
    """OpenCV draws ASCII only; other letters would come out as question marks."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_POLISH))
    return decomposed.encode("ascii", "ignore").decode("ascii")


def _label(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    cv2.putText(canvas, text, (x, y), FONT, TEXT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), FONT, TEXT_SCALE, color, 1, cv2.LINE_AA)


def _panel(canvas: np.ndarray, lines: list[str]) -> None:
    """Text block in the bottom left corner on a darkened background."""
    height, width = canvas.shape[:2]
    panel_w = min(width, max(cv2.getTextSize(line, FONT, TEXT_SCALE, 1)[0][0] for line in lines) + 12)
    panel_h = min(height, LINE_HEIGHT * len(lines) + 8)
    top = height - panel_h
    canvas[top:height, 0:panel_w] = (canvas[top:height, 0:panel_w] * 0.25).astype(np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (6, top + 14 + index * LINE_HEIGHT), FONT, TEXT_SCALE, (235, 235, 235), 1, cv2.LINE_AA)


def draw_debug(canvas: np.ndarray, profile: Profile, result: FrameResult, fps: float, timings: FrameTimings) -> None:
    height, width = canvas.shape[:2]
    colors = {item.name: item.draw_bgr for item in profile.items}

    for name in {item.region for item in profile.items if item.enabled and item.region is not None}:
        x, y, w, h = profile.region_rect(name, width, height)
        cv2.rectangle(canvas, (x, y), (x + max(w, 1) - 1, y + max(h, 1) - 1), (150, 150, 150), 1)
        _label(canvas, ascii_text(name), x + 3, y + 11, (190, 190, 190))

    for track in result.tracks:
        color = colors.get(track.rule, (255, 255, 255))
        if len(track.trail) >= 2:
            cv2.polylines(canvas, [np.array(track.trail, np.int32)], False, color, 1, cv2.LINE_AA)
        if not track.visible:
            color = (150, 150, 150)  # briefly unseen: still tracked at its last position
        corner = (track.x + track.width - 1, track.y + track.height - 1)
        cv2.rectangle(canvas, (track.x, track.y), corner, color, 1)
        confidence = "" if track.score is None else f" {track.score:.0%}"
        motion = f" {track.speed:.0f}px/s" if track.moving else ""
        text = f"#{track.id} {ascii_text(track.rule)}{confidence}{motion}"
        _label(canvas, text, track.x, min(height - 3, track.y + track.height + 11), color)

    lines = [
        f"{fps:.0f} FPS  {width}x{height}  obiekty: {len(result.tracks)}",
        f"przechwyt {timings.capture_ms:.1f} ms  wykrywanie {timings.detect_ms:.1f} ms  pluginy {timings.plugins_ms:.1f} ms",
    ]
    active = [ascii_text(name) for name, status in result.states.items() if status.active]
    if active:
        lines.append("stany: " + ", ".join(active))
    rules = list(result.detections.items())
    for rule, found in rules[:PANEL_RULE_LINES]:
        scores = [d.score for d in found if d.score is not None]
        confidence = f"  srednio {sum(scores) / len(scores):.0%}" if scores else ""
        lines.append(f"{ascii_text(rule)}: {len(found)}{confidence}")
    if len(rules) > PANEL_RULE_LINES:
        lines.append(f"... i jeszcze {len(rules) - PANEL_RULE_LINES}")
    _panel(canvas, lines)
