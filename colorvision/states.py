"""State manager: named states derived from detections over time.

States are independent of each other, so several can be active at once. A state flips
only after its condition has held for the configured delay, so one wrong frame cannot
switch it. It only observes: nothing here controls the game.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

OP_AT_LEAST = ">="
OP_AT_MOST = "<="
OP_EXACTLY = "=="
OPERATORS = (OP_AT_LEAST, OP_AT_MOST, OP_EXACTLY)
MODE_ALL = "all"
MODE_ANY = "any"
MODES = (MODE_ALL, MODE_ANY)


@dataclass(frozen=True)
class Condition:
    """How many objects a colour rule, template or plugin must report."""

    item: str
    op: str = OP_AT_LEAST
    value: int = 1

    def __post_init__(self) -> None:
        if not self.item:
            raise ValueError("condition needs an item name")
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}")
        if self.value < 0:
            raise ValueError("condition value must be >= 0")

    def holds(self, detections: Mapping[str, Sequence]) -> bool:
        # An item that reported nothing at all (deleted, disabled) is unknown, not zero:
        # otherwise "exactly 0" would turn true merely because a rule was switched off.
        found = detections.get(self.item)
        if found is None:
            return False
        count = len(found)
        if self.op == OP_AT_LEAST:
            return count >= self.value
        if self.op == OP_AT_MOST:
            return count <= self.value
        return count == self.value


@dataclass(frozen=True)
class StateRule:
    name: str
    conditions: tuple[Condition, ...] = ()
    mode: str = MODE_ALL
    enter_ms: int = 300
    exit_ms: int = 300
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("state needs a name")
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: unknown mode {self.mode!r}")
        if self.enter_ms < 0 or self.exit_ms < 0:
            raise ValueError(f"{self.name}: delays must be >= 0")

    def holds(self, detections: Mapping[str, Sequence]) -> bool:
        if not self.conditions:
            return False
        results = (condition.holds(detections) for condition in self.conditions)
        return all(results) if self.mode == MODE_ALL else any(results)


@dataclass(frozen=True)
class StateStatus:
    name: str
    active: bool
    duration: float  # seconds in the current value


@dataclass(frozen=True)
class StateChange:
    name: str
    active: bool
    previous_duration: float  # seconds spent in the previous value
    timestamp: float  # wall clock, for display


@dataclass
class _Entry:
    active: bool
    since: float
    pending_since: float | None = None


class StateTracker:
    """Tracks states over time. `now` is a monotonic clock in seconds."""

    def __init__(self, rules: Iterable[StateRule] = ()):
        self._rules: list[StateRule] = []
        self._entries: dict[str, _Entry] = {}
        self.set_rules(rules)

    def set_rules(self, rules: Iterable[StateRule]) -> None:
        """Change definitions while running; states that remain keep their value and time."""
        self._rules = [rule for rule in rules if rule.enabled]
        names = {rule.name for rule in self._rules}
        self._entries = {name: entry for name, entry in self._entries.items() if name in names}

    def clear_pending(self) -> None:
        """Forget unfinished delays, e.g. when frames stopped arriving during a pause."""
        for entry in self._entries.values():
            entry.pending_since = None

    def update(self, detections: Mapping[str, Sequence], now: float) -> list[StateChange]:
        changes: list[StateChange] = []
        for rule in self._rules:
            entry = self._entries.setdefault(rule.name, _Entry(False, now))
            holds = rule.holds(detections)
            if holds == entry.active:
                entry.pending_since = None
                continue
            if entry.pending_since is None:
                entry.pending_since = now
            delay_ms = rule.enter_ms if holds else rule.exit_ms
            if (now - entry.pending_since) * 1000 >= delay_ms:
                changes.append(StateChange(rule.name, holds, now - entry.since, time.time()))
                entry.active, entry.since, entry.pending_since = holds, now, None
        return changes

    def snapshot(self, now: float) -> dict[str, StateStatus]:
        return {
            name: StateStatus(name, entry.active, now - entry.since)
            for name, entry in self._entries.items()
        }


def missing_items(rule: StateRule, known: Iterable[str]) -> list[str]:
    """Condition items that nothing in the profile provides (a plugin still might)."""
    available = set(known)
    return sorted({c.item for c in rule.conditions if c.item not in available})


def rename_item(states: Iterable[StateRule], old: str, new: str) -> list[StateRule]:
    """Keep conditions pointing at a rule or template after it is renamed."""
    return [
        replace(state, conditions=tuple(
            replace(c, item=new) if c.item == old else c for c in state.conditions
        ))
        for state in states
    ]


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes}:{secs:02}"
