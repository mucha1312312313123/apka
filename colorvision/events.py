"""Events: moments worth noticing, produced by the object tracker and the state manager."""
from __future__ import annotations

from dataclasses import dataclass

from .states import StateChange, format_duration

EVENT_APPEARED = "appeared"
EVENT_DISAPPEARED = "disappeared"
EVENT_MOVING = "moving"
EVENT_STOPPED = "stopped"
EVENT_STATE_ON = "state_on"
EVENT_STATE_OFF = "state_off"


@dataclass(frozen=True)
class Event:
    kind: str
    source: str  # rule or state name
    message: str
    timestamp: float  # wall clock
    track_id: int | None = None


def event_from_state_change(change: StateChange) -> Event:
    if change.active:
        return Event(EVENT_STATE_ON, change.name, f"Stan {change.name}: start", change.timestamp)
    message = f"Stan {change.name}: koniec po {format_duration(change.previous_duration)}"
    return Event(EVENT_STATE_OFF, change.name, message, change.timestamp)
