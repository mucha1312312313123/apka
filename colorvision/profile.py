"""Loading and saving detection profiles (JSON). Template pictures are embedded as
base64 PNG, so a single profile file carries everything to another computer."""
from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .automation import Decision, Step, Task
from .detector import SHAPE_ANY, ColorRule, Rect, TemplateRule
from .states import Condition, StateRule


class ProfileError(Exception):
    """The profile file is missing or invalid."""


@dataclass
class Profile:
    reference_size: tuple[int, int] = (765, 503)
    regions: dict[str, Rect] = field(default_factory=dict)
    rules: list[ColorRule] = field(default_factory=list)
    templates: list[TemplateRule] = field(default_factory=list)
    states: list[StateRule] = field(default_factory=list)
    track_max_distance: int = 60  # pixels between frames, at the reference size
    track_lost_ms: int = 1000  # how long an unseen object is kept before it counts as gone
    tasks: list[Task] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def items(self) -> list:
        """Colour rules and templates together; their names are unique across both."""
        return [*self.rules, *self.templates]

    def copy(self) -> "Profile":
        """Independent copy that is safe to hand to another thread (rules are immutable)."""
        return Profile(
            self.reference_size, dict(self.regions), list(self.rules), list(self.templates), list(self.states),
            self.track_max_distance, self.track_lost_ms, list(self.tasks), list(self.decisions),
        )

    def region_rect(self, name: str | None, frame_w: int, frame_h: int) -> Rect:
        """Scale a named region from the reference size to the frame and clip it."""
        if name is None:
            return (0, 0, frame_w, frame_h)
        if name not in self.regions:
            raise ProfileError(f"Nieznany region: {name!r}")
        x, y, w, h = self.regions[name]
        sx, sy = frame_w / self.reference_size[0], frame_h / self.reference_size[1]
        left, top = max(0, round(x * sx)), max(0, round(y * sy))
        right, bottom = min(frame_w, round((x + w) * sx)), min(frame_h, round((y + h) * sy))
        return (left, top, max(0, right - left), max(0, bottom - top))


def _decode_image(text: str) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(text, validate=True), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError("uszkodzony obraz wzorca")
    return image


def _encode_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("nie można zakodować obrazu wzorca")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def load_profile(path: str | Path) -> Profile:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = Profile(
            reference_size=tuple(raw.get("reference_size", (765, 503))),
            regions={k: tuple(v) for k, v in raw.get("regions", {}).items()},
            rules=[
                ColorRule(
                    name=r["name"],
                    lower=tuple(r["lower"]),
                    upper=tuple(r["upper"]),
                    region=r.get("region"),
                    min_area=r.get("min_area", 10),
                    max_area=r.get("max_area"),
                    merge_px=r.get("merge_px", 3),
                    draw_bgr=tuple(r.get("draw_bgr", (0, 255, 255))),
                    enabled=bool(r.get("enabled", True)),
                    shape=r.get("shape", SHAPE_ANY),
                    min_confidence=r.get("min_confidence", 0.0),
                )
                for r in raw.get("rules", [])
            ],
            templates=[
                TemplateRule(
                    name=t["name"],
                    image=_decode_image(t["png_base64"]),
                    threshold=t.get("threshold", 0.9),
                    region=t.get("region"),
                    max_results=t.get("max_results", 10),
                    draw_bgr=tuple(t.get("draw_bgr", (255, 160, 60))),
                    enabled=bool(t.get("enabled", True)),
                    captured_size=tuple(t.get("captured_size", (765, 503))),
                )
                for t in raw.get("templates", [])
            ],
            states=[
                StateRule(
                    name=s["name"],
                    conditions=tuple(
                        Condition(c["item"], c.get("op", ">="), c.get("value", 1)) for c in s.get("conditions", [])
                    ),
                    mode=s.get("mode", "all"),
                    enter_ms=s.get("enter_ms", 300),
                    exit_ms=s.get("exit_ms", 300),
                    enabled=bool(s.get("enabled", True)),
                )
                for s in raw.get("states", [])
            ],
            track_max_distance=raw.get("track_max_distance", 60),
            track_lost_ms=raw.get("track_lost_ms", 1000),
            tasks=[
                Task(t["name"], tuple(Step(**s) for s in t.get("steps", [])))
                for t in raw.get("tasks", [])
            ],
            decisions=[
                Decision(
                    name=d["name"],
                    task=d["task"],
                    priority=d.get("priority", 0),
                    when_state=d.get("when_state"),
                    unless_state=d.get("unless_state"),
                    conditions=tuple(
                        Condition(c["item"], c.get("op", ">="), c.get("value", 1)) for c in d.get("conditions", [])
                    ),
                    cooldown_ms=d.get("cooldown_ms", 5000),
                    interrupt=bool(d.get("interrupt", False)),
                    enabled=bool(d.get("enabled", True)),
                )
                for d in raw.get("decisions", [])
            ],
        )
        if profile.track_max_distance <= 0 or profile.track_lost_ms < 0:
            raise ValueError("track_max_distance musi być dodatnie, a track_lost_ms nie może być ujemne")
    except OSError as exc:
        raise ProfileError(f"Nie można odczytać profilu {path}: {exc}") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError, binascii.Error) as exc:
        raise ProfileError(f"Błędny profil {path}: {exc}") from exc

    names = [item.name for item in profile.items]
    if len(names) != len(set(names)):
        raise ProfileError(f"Błędny profil {path}: nazwy reguł się powtarzają")
    state_names = [state.name for state in profile.states]
    if len(state_names) != len(set(state_names)):
        raise ProfileError(f"Błędny profil {path}: nazwy stanów się powtarzają")
    task_names = [task.name for task in profile.tasks]
    decision_names = [decision.name for decision in profile.decisions]
    if len(task_names) != len(set(task_names)) or len(decision_names) != len(set(decision_names)):
        raise ProfileError(f"Błędny profil {path}: nazwy zadań lub decyzji się powtarzają")
    for decision in profile.decisions:
        if decision.task not in task_names:
            raise ProfileError(f"Decyzja {decision.name!r} używa nieznanego zadania {decision.task!r}")
    for item in profile.items:
        if item.region is not None and item.region not in profile.regions:
            raise ProfileError(f"Reguła {item.name!r} używa nieznanego regionu {item.region!r}")
    return profile


def save_profile(profile: Profile, path: str | Path) -> None:
    """Write through a temporary file so a crash cannot corrupt the profile."""
    data = {
        "track_max_distance": profile.track_max_distance,
        "track_lost_ms": profile.track_lost_ms,
        "reference_size": list(profile.reference_size),
        "regions": {k: list(v) for k, v in profile.regions.items()},
        "rules": [
            {
                "name": r.name,
                "lower": list(r.lower),
                "upper": list(r.upper),
                "region": r.region,
                "min_area": r.min_area,
                "max_area": r.max_area,
                "merge_px": r.merge_px,
                "draw_bgr": list(r.draw_bgr),
                "enabled": r.enabled,
                "shape": r.shape,
                "min_confidence": r.min_confidence,
            }
            for r in profile.rules
        ],
        "templates": [
            {
                "name": t.name,
                "threshold": t.threshold,
                "region": t.region,
                "max_results": t.max_results,
                "draw_bgr": list(t.draw_bgr),
                "enabled": t.enabled,
                "captured_size": list(t.captured_size),
                "png_base64": _encode_image(t.image),
            }
            for t in profile.templates
        ],
        "states": [
            {
                "name": s.name,
                "mode": s.mode,
                "enter_ms": s.enter_ms,
                "exit_ms": s.exit_ms,
                "enabled": s.enabled,
                "conditions": [{"item": c.item, "op": c.op, "value": c.value} for c in s.conditions],
            }
            for s in profile.states
        ],
        "tasks": [{"name": t.name, "steps": [asdict(step) for step in t.steps]} for t in profile.tasks],
        "decisions": [
            {
                "name": d.name,
                "task": d.task,
                "priority": d.priority,
                "when_state": d.when_state,
                "unless_state": d.unless_state,
                "conditions": [{"item": c.item, "op": c.op, "value": c.value} for c in d.conditions],
                "cooldown_ms": d.cooldown_ms,
                "interrupt": d.interrupt,
                "enabled": d.enabled,
            }
            for d in profile.decisions
        ],
    }
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
