"""Plugin API, plugin loader and the per-frame vision pipeline."""
from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection, detect_rule, match_template
from .events import Event, event_from_state_change
from .profile import Profile
from .states import StateChange, StateStatus, StateTracker
from .tracking import ObjectTracker, Track


@dataclass
class FrameResult:
    frame: np.ndarray  # BGR image of the client area
    detections: dict[str, list[Detection]]  # rule name -> blobs, largest first
    timestamp: float
    states: dict[str, StateStatus] = field(default_factory=dict)
    tracks: list[Track] = field(default_factory=list)

    def best(self, rule: str) -> Detection | None:
        """The largest blob for a rule, or None."""
        found = self.detections.get(rule)
        return found[0] if found else None

    def is_active(self, state: str) -> bool:
        """Whether a named state is currently active; False if there is no such state."""
        status = self.states.get(state)
        return status is not None and status.active


class Plugin:
    """Subclass this in a file inside the plugins/ folder and override what you need."""

    name = "plugin"

    def on_start(self, profile: Profile) -> None:
        pass

    def on_frame(self, result: FrameResult) -> None:
        pass

    def on_state_change(self, change: StateChange) -> None:
        """Called when a state turns on or off."""

    def on_event(self, event: Event) -> None:
        """Called for every event: objects appearing, disappearing, moving, and state changes."""

    def draw(self, canvas: np.ndarray, result: FrameResult) -> None:
        """Draw on the preview image; called only when the preview is open."""

    def on_stop(self) -> None:
        pass


def process_frame(frame: np.ndarray, profile: Profile) -> FrameResult:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    detections = {
        rule.name: detect_rule(hsv, rule, profile.region_rect(rule.region, width, height))
        for rule in profile.rules
        if rule.enabled
    }
    for template in profile.templates:
        if template.enabled:
            detections[template.name] = match_template(frame, template, profile.region_rect(template.region, width, height))
    return FrameResult(frame=frame, detections=detections, timestamp=time.time())


def advance_states(result: FrameResult, tracker: StateTracker, host: "PluginHost", now: float) -> list[StateChange]:
    """Run plugins and update states for a live frame.

    Plugins run first, so conditions can use detections they add. Their on_frame therefore
    sees the states as of the previous frame; on_state_change and draw see the new ones.
    """
    result.states = tracker.snapshot(now)
    host.frame(result)
    changes = tracker.update(result.detections, now)
    result.states = tracker.snapshot(now)
    for change in changes:
        host.state_change(change)
    return changes


class Pipeline:
    """What happens to each live frame after detection: plugins, states, tracking, events."""

    def __init__(self, profile: Profile, host: "PluginHost"):
        self.profile = profile
        self.host = host
        self.states = StateTracker(profile.states)
        self.objects = ObjectTracker()

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile
        self.states.set_rules(profile.states)

    def idle(self) -> None:
        """No new picture is arriving: time must not count towards state delays."""
        self.states.clear_pending()

    def describe(self, result: FrameResult, now: float) -> None:
        """Fill in current states and objects without advancing them, e.g. for a held frame."""
        result.states = self.states.snapshot(now)
        result.tracks = self.objects.snapshot()

    def step(self, result: FrameResult, now: float) -> tuple[list[StateChange], list[Event]]:
        result.tracks = self.objects.snapshot()
        changes = advance_states(result, self.states, self.host, now)
        scale = result.frame.shape[1] / self.profile.reference_size[0]
        result.tracks, track_events = self.objects.update(
            result.detections, now, self.profile.track_max_distance * scale, self.profile.track_lost_ms / 1000
        )
        events = [event_from_state_change(change) for change in changes] + track_events
        for event in events:
            self.host.event(event)
        return changes, events


def load_plugins(folder: str | Path) -> list[Plugin]:
    """Instantiate each Plugin subclass found in folder/*.py; files starting with _ are skipped."""
    plugins: list[Plugin] = []
    directory = Path(folder)
    if not directory.is_dir():
        return plugins
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"colorvision_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, Plugin) and cls is not Plugin and cls.__module__ == module_name:
                    plugins.append(cls())
        except Exception:
            print(f"[plugin] Nie wczytano {path.name}:\n{traceback.format_exc()}", file=sys.stderr)
    return plugins


class PluginHost:
    """Calls plugins and disables any that raise, so one bug cannot stop the loop."""

    def __init__(self, plugins: list[Plugin]):
        self.plugins = list(plugins)

    def _call(self, method: str, *args) -> None:
        for plugin in list(self.plugins):
            try:
                getattr(plugin, method)(*args)
            except Exception:
                print(
                    f"[plugin] {plugin.name} wyłączony po błędzie w {method}:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                self.plugins.remove(plugin)

    def start(self, profile: Profile) -> None:
        self._call("on_start", profile)

    def frame(self, result: FrameResult) -> None:
        self._call("on_frame", result)

    def state_change(self, change: StateChange) -> None:
        self._call("on_state_change", change)

    def event(self, event: Event) -> None:
        self._call("on_event", event)

    def draw(self, canvas: np.ndarray, result: FrameResult) -> None:
        self._call("draw", canvas, result)

    def stop(self) -> None:
        self._call("on_stop")
