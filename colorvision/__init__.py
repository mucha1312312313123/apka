"""Colour-based vision for a game client window: no injection, screen pixels only."""
from .detector import ColorRule, Detection
from .events import Event
from .plugin import FrameResult, Plugin
from .profile import Profile
from .states import Condition, StateChange, StateRule, StateStatus
from .tracking import Track

__all__ = [
    "ColorRule", "Condition", "Detection", "Event", "FrameResult", "Plugin", "Profile",
    "StateChange", "StateRule", "StateStatus", "Track",
]
