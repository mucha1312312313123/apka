"""Background thread: capture, detection, plugins, tracking and rendering of the preview."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ..automation import MODE_OFF, Automation
from ..debug_overlay import FrameTimings, draw_debug
from ..detector import ColorRule, Detection, annotate, hsv_mask
from ..input_win import WindowsInput, emergency_key_pressed
from ..plugin import FrameResult, Pipeline, Plugin, PluginHost, process_frame
from ..profile import Profile
from ..states import StateStatus
from ..tracking import Track

MODE_DETECTIONS = "detections"
MODE_MASK = "mask"
SAMPLE_PERIOD = 0.2  # seconds between history samples


@dataclass
class FramePacket:
    image: QImage
    frame: np.ndarray  # raw BGR frame, without any drawing
    detections: dict[str, list[Detection]]
    fps: float
    covered: bool
    states: dict[str, StateStatus]
    tracks: list[Track]
    task: str | None = None


def render_view(
    frame: np.ndarray, profile: Profile, result: FrameResult, mode: str, selected: str | None, labels: bool = True
) -> np.ndarray:
    """Build the preview image. The mask view shows what the selected rule matches."""
    rule = next((r for r in profile.items if r.name == selected), None)
    height, width = frame.shape[:2]
    rect = profile.region_rect(rule.region, width, height) if rule is not None else None

    if mode == MODE_MASK and isinstance(rule, ColorRule):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = hsv_mask(hsv, rule.lower, rule.upper)
        x, y, w, h = rect
        keep = np.zeros(mask.shape, np.uint8)
        keep[y:y + h, x:x + w] = 255
        mask = cv2.bitwise_and(mask, keep)
        canvas = (frame * 0.2).astype(np.uint8)
        canvas[mask > 0] = frame[mask > 0]
    else:
        canvas = annotate(frame, profile.items, result.detections, labels)

    if rule is not None and rule.region is not None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + max(w, 1) - 1, y + max(h, 1) - 1), (200, 200, 200), 1, cv2.LINE_4)
    return canvas


def to_qimage(canvas: np.ndarray) -> QImage:
    height, width = canvas.shape[:2]
    return QImage(canvas.data, width, height, 3 * width, QImage.Format_BGR888).copy()


def summarize(detections: dict[str, list[Detection]]) -> dict[str, tuple[int, float | None]]:
    """Count and mean confidence per rule, for the history."""
    summary = {}
    for rule, found in detections.items():
        scores = [d.score for d in found if d.score is not None]
        summary[rule] = (len(found), sum(scores) / len(scores) if scores else None)
    return summary


class VisionWorker(QThread):
    packetReady = Signal(object)
    stateChanged = Signal(str, str)  # state of the program, message
    gameStateChanged = Signal(object)  # StateChange; never skipped, unlike frames
    eventDetected = Signal(object)  # Event; never skipped
    sampleReady = Signal(float, object)  # wall clock, summarize() result
    automationStopped = Signal(str)  # automation switched itself off; the reason
    sourceInfoChanged = Signal(object)  # WindowSource.describe() result, when it changes

    def __init__(self, source_factory: Callable, profile: Profile, plugins: list[Plugin], fps: float, parent=None):
        super().__init__(parent)
        self._source_factory = source_factory
        self._plugins = plugins
        self._lock = threading.Lock()
        self._profile = profile.copy()
        self._mode, self._selected = MODE_DETECTIONS, None
        self._debug = False
        self._automation_mode = MODE_OFF
        self._task_requests: list[str] = []
        self._crop: tuple[int, int, int, int] | None = None
        self._paused = False
        self._period = 1.0 / fps
        self._ready_for_packet = threading.Event()
        self._ready_for_packet.set()

    def set_profile(self, profile: Profile) -> None:
        with self._lock:
            self._profile = profile.copy()

    def set_view(self, mode: str, selected: str | None, debug: bool = False) -> None:
        with self._lock:
            self._mode, self._selected, self._debug = mode, selected, debug

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def set_crop(self, crop: tuple[int, int, int, int] | None) -> None:
        with self._lock:
            self._crop = crop

    def set_automation_mode(self, mode: str) -> None:
        with self._lock:
            self._automation_mode = mode

    def request_task(self, name: str) -> None:
        with self._lock:
            self._task_requests.append(name)

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._period = 1.0 / fps

    def acknowledge_packet(self) -> None:
        """The interface drew the last frame; a slow UI simply skips frames."""
        self._ready_for_packet.set()

    def _announce(self, state: str, message: str = "") -> None:
        if (state, message) != getattr(self, "_last_state", None):
            self._last_state = (state, message)
            self.stateChanged.emit(state, message)

    def run(self) -> None:
        try:
            source = self._source_factory()
        except Exception as exc:
            self._announce("error", f"Nie można uruchomić źródła obrazu: {exc}")
            return

        with self._lock:
            start_profile = self._profile
        host = PluginHost(self._plugins)
        host.start(start_profile.copy())
        pipeline = Pipeline(start_profile, host)
        automation = Automation(start_profile, WindowsInput)
        target_provider = getattr(source, "input_target", None)

        def emit_events(events) -> None:
            for event in events:
                self.eventDetected.emit(event)

        emergency_key_pressed()  # discard a press from before this run
        applied_crop = None
        source_info = None
        held = None
        fps = 0.0
        timings = FrameTimings()
        last_frame_time = time.perf_counter()
        last_sample = 0.0
        covered, last_cover_check = False, 0.0
        try:
            while not self.isInterruptionRequested():
                started = time.perf_counter()
                with self._lock:
                    profile, mode, selected, debug = self._profile, self._mode, self._selected, self._debug
                    paused, period = self._paused, self._period
                    automation_mode = self._automation_mode
                    requests, self._task_requests = self._task_requests, []
                    crop = self._crop
                if crop != applied_crop and hasattr(source, "set_crop"):
                    source.set_crop(crop)
                    applied_crop = crop
                    held = None  # a held frame has the old size
                if profile is not pipeline.profile:
                    pipeline.set_profile(profile)
                    automation.set_profile(profile)
                emit_events(automation.set_mode(automation_mode))
                for name in requests:
                    automation.request(name)
                # Checked even with no picture: the stop key must always work.
                if automation_mode != MODE_OFF and emergency_key_pressed():
                    emit_events(automation.emergency_stop("naciśnięto F12"))
                    with self._lock:
                        self._automation_mode = MODE_OFF
                    self.automationStopped.emit(automation.stop_reason)
                    continue

                # While paused the held frame is re-processed, so edits show at once.
                if paused and held is not None:
                    frame = held
                else:
                    try:
                        frame = source.read()
                    except Exception as exc:
                        pipeline.idle()
                        emit_events(automation.interrupt("błąd przechwytywania"))
                        self._announce("error", f"Błąd przechwytywania: {exc}")
                        self.msleep(500)
                        continue
                    if frame is None:
                        pipeline.idle()
                        emit_events(automation.interrupt("brak okna klienta"))
                        self._announce("waiting", "Czekam na okno klienta. Nie może być zamknięte ani zminimalizowane.")
                        self.msleep(300)
                        continue
                    held = frame
                self._announce("paused" if paused else "running")
                captured = time.perf_counter()

                try:
                    result = process_frame(frame, profile)
                    detected = time.perf_counter()
                    if paused:
                        # A held frame says nothing new about the game: nothing advances.
                        pipeline.idle()
                        pipeline.describe(result, time.monotonic())
                        emit_events(automation.interrupt("pauza"))
                    else:
                        changes, events = pipeline.step(result, time.monotonic())
                        for change in changes:
                            self.gameStateChanged.emit(change)
                        emit_events(events)
                        emit_events(automation.step(result, time.monotonic(), target_provider))
                        if automation.stop_reason is not None:
                            reason, automation.stop_reason = automation.stop_reason, None
                            with self._lock:
                                self._automation_mode = MODE_OFF
                            self.automationStopped.emit(reason)
                        if detected - last_sample >= SAMPLE_PERIOD:
                            last_sample = detected
                            self.sampleReady.emit(time.time(), summarize(result.detections))
                    finished = time.perf_counter()
                    timings = FrameTimings(
                        (captured - started) * 1000, (detected - captured) * 1000, (finished - detected) * 1000
                    )
                    if finished - last_cover_check >= 1.0 and hasattr(source, "is_covered"):
                        covered, last_cover_check = source.is_covered(), finished
                        info = source.describe()
                        if info != source_info:
                            source_info = info
                            self.sourceInfoChanged.emit(info)
                    if self._ready_for_packet.is_set():
                        self._ready_for_packet.clear()
                        # The debug overlay has its own, fuller labels.
                        canvas = render_view(frame, profile, result, mode, selected, labels=not debug)
                        if debug:
                            draw_debug(canvas, profile, result, fps, timings)
                        host.draw(canvas, result)
                        self.packetReady.emit(
                            FramePacket(
                                to_qimage(canvas), frame, result.detections, fps, covered, result.states, result.tracks,
                                automation.runner.task.name if automation.runner.task else None,
                            )
                        )
                except Exception:
                    print(traceback.format_exc())
                    self._announce("error", "Błąd przetwarzania klatki. Szczegóły są w dzienniku.")
                    self.msleep(500)
                    continue

                now = time.perf_counter()
                instant = 1.0 / max(now - last_frame_time, 1e-6)
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
                last_frame_time = now
                remaining = period - (now - started)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
        finally:
            host.stop()
            source.close()
