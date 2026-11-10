"""Decision engine, task manager and action manager.

Decisions pick which task to start, a task runs its steps one by one, and each step goes
through the action manager, the only place allowed to touch the mouse or keyboard.
Off by default. Input is deliberately plain: fixed positions and no randomised timing or
movement, and nothing here tries to disguise automation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .events import Event
from .states import Condition

MODE_OFF = "off"
MODE_SIMULATE = "simulate"
MODE_ARMED = "armed"

STEP_LOG = "log"
STEP_WAIT = "wait"
STEP_WAIT_FOR = "wait_for"
STEP_CLICK = "click"
STEP_KEY = "key"
STEP_KINDS = (STEP_LOG, STEP_WAIT, STEP_WAIT_FOR, STEP_CLICK, STEP_KEY)
BUTTONS = ("left", "right")
KEYS = (
    tuple("abcdefghijklmnopqrstuvwxyz0123456789")
    + tuple(f"f{n}" for n in range(1, 12))  # F12 is reserved for the emergency stop
    + ("space", "enter", "esc", "tab", "shift", "ctrl", "up", "down", "left", "right")
)

EVENT_TASK_START = "task_start"
EVENT_TASK_DONE = "task_done"
EVENT_TASK_FAILED = "task_failed"
EVENT_ACTION = "action"

MIN_INPUT_GAP = 0.1  # seconds between real or simulated inputs
MAX_INPUTS_PER_ARMING = 500  # runaway protection: real control switches itself off after this


@dataclass(frozen=True)
class Step:
    kind: str
    target: str = ""  # rule or template, for click and wait_for
    button: str = "left"
    key: str = ""
    ms: int = 0  # duration of wait, time limit of wait_for
    op: str = ">="
    value: int = 1
    message: str = ""

    def __post_init__(self) -> None:
        if self.kind not in STEP_KINDS:
            raise ValueError(f"unknown step {self.kind!r}")
        if self.kind in (STEP_CLICK, STEP_WAIT_FOR) and not self.target:
            raise ValueError(f"step {self.kind} needs a target")
        if self.button not in BUTTONS:
            raise ValueError(f"unknown button {self.button!r}")
        if self.kind == STEP_KEY and self.key not in KEYS:
            raise ValueError(f"unknown key {self.key!r}")
        if self.ms < 0:
            raise ValueError("step time must be >= 0")
        if self.kind == STEP_WAIT_FOR:
            Condition(self.target, self.op, self.value)


@dataclass(frozen=True)
class Task:
    name: str
    steps: tuple[Step, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task needs a name")


@dataclass(frozen=True)
class Decision:
    """Start `task` when everything listed holds. A higher priority is checked first."""

    name: str
    task: str
    priority: int = 0
    when_state: str | None = None
    unless_state: str | None = None
    conditions: tuple[Condition, ...] = ()
    cooldown_ms: int = 5000
    interrupt: bool = False  # may replace a running task of lower priority
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.task:
            raise ValueError("decision needs a name and a task")
        if self.cooldown_ms < 0:
            raise ValueError(f"{self.name}: cooldown must be >= 0")
        if not (self.when_state or self.unless_state or self.conditions):
            raise ValueError(f"{self.name}: decision needs at least one condition")

    def holds(self, result) -> bool:
        if self.when_state and not result.is_active(self.when_state):
            return False
        if self.unless_state and (self.unless_state not in result.states or result.is_active(self.unless_state)):
            return False
        return all(condition.holds(result.detections) for condition in self.conditions)


class InputBackend(Protocol):
    def click(self, x: int, y: int, button: str) -> None: ...
    def press(self, key: str) -> None: ...


class ActionManager:
    """Performs actions and enforces the safety rules. Returns (ok, message)."""

    def __init__(self, real_backend_factory: Callable[[], InputBackend] | None = None):
        self._real_factory = real_backend_factory
        self._real: InputBackend | None = None
        self.mode = MODE_OFF
        self.inputs = 0
        self._last_input = float("-inf")

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.inputs = 0

    def ready(self, now: float) -> bool:
        return now - self._last_input >= MIN_INPUT_GAP

    def _real_backend(self) -> InputBackend:
        if self._real is None:
            self._real = self._real_factory()
        return self._real

    def click(self, step: Step, result, target_rect, now: float) -> tuple[bool, str]:
        found = result.best(step.target)
        if found is None:
            return False, f"Nie znaleziono celu: {step.target}"
        x, y = found.center
        label = "lewy" if step.button == "left" else "prawy"
        if self.mode == MODE_SIMULATE:
            self._last_input = now
            return True, f"Symulacja: klik {label} w {step.target} ({x}, {y})"
        if target_rect is None:
            return False, "Okno klienta nie jest aktywne albo jest przykryte"
        left, top, width, height = target_rect
        if not (0 <= x < width and 0 <= y < height):
            return False, f"Cel {step.target} jest poza oknem klienta"
        self._real_backend().click(left + x, top + y, step.button)
        self._count(now)
        return True, f"Klik {label} w {step.target} ({x}, {y})"

    def press(self, step: Step, target_rect, now: float) -> tuple[bool, str]:
        if self.mode == MODE_SIMULATE:
            self._last_input = now
            return True, f"Symulacja: klawisz {step.key}"
        if target_rect is None:
            return False, "Okno klienta nie jest aktywne albo jest przykryte"
        self._real_backend().press(step.key)
        self._count(now)
        return True, f"Klawisz {step.key}"

    def _count(self, now: float) -> None:
        self._last_input = now
        self.inputs += 1


class TaskRunner:
    """Task manager: one task at a time, advancing without ever blocking the frame loop."""

    def __init__(self, actions: ActionManager):
        self.actions = actions
        self.task: Task | None = None
        self.priority = 0
        self._index = 0
        self._wait_until = 0.0
        self._deadline: float | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None

    def start(self, task: Task, priority: int) -> Event:
        self.task, self.priority, self._index, self._wait_until, self._deadline = task, priority, 0, 0.0, None
        return Event(EVENT_TASK_START, task.name, f"Zadanie {task.name}: start", time.time())

    def cancel(self, reason: str) -> Event | None:
        if self.task is None:
            return None
        name, self.task = self.task.name, None
        return Event(EVENT_TASK_FAILED, name, f"Zadanie {name}: przerwane ({reason})", time.time())

    def _finish(self, kind: str, message: str) -> Event:
        name, self.task = self.task.name, None
        return Event(kind, name, f"Zadanie {name}: {message}", time.time())

    def tick(self, result, now: float, target_rect) -> list[Event]:
        events: list[Event] = []
        while self.task is not None:
            if now < self._wait_until:
                break
            if self._index >= len(self.task.steps):
                events.append(self._finish(EVENT_TASK_DONE, "ukończone"))
                break
            step = self.task.steps[self._index]
            name = self.task.name
            if step.kind == STEP_LOG:
                events.append(Event(EVENT_ACTION, name, step.message or "(pusty komunikat)", time.time()))
            elif step.kind == STEP_WAIT:
                self._wait_until = now + step.ms / 1000
            elif step.kind == STEP_WAIT_FOR:
                if self._deadline is None:
                    self._deadline = now + step.ms / 1000
                if not Condition(step.target, step.op, step.value).holds(result.detections):
                    if now >= self._deadline:
                        events.append(self._finish(EVENT_TASK_FAILED, f"nie doczekano się {step.target}"))
                    break
                self._deadline = None
            else:
                if not self.actions.ready(now):
                    break
                if step.kind == STEP_CLICK:
                    ok, message = self.actions.click(step, result, target_rect, now)
                else:
                    ok, message = self.actions.press(step, target_rect, now)
                if not ok:
                    events.append(self._finish(EVENT_TASK_FAILED, message))
                    break
                events.append(Event(EVENT_ACTION, name, message, time.time()))
                self._index += 1
                break  # at most one input per frame, so the next step sees a fresh picture
            self._index += 1
        return events


class Automation:
    """Decision engine plus the parts it drives. Call step() once for each live frame."""

    def __init__(self, profile, real_backend_factory: Callable[[], InputBackend] | None = None):
        self.actions = ActionManager(real_backend_factory)
        self.runner = TaskRunner(self.actions)
        self.mode = MODE_OFF
        self.stop_reason: str | None = None
        self._last_started: dict[str, float] = {}
        self._requests: list[str] = []
        self.set_profile(profile)

    def set_profile(self, profile) -> None:
        self._tasks = {task.name: task for task in profile.tasks}
        self._decisions = sorted(
            (d for d in profile.decisions if d.enabled), key=lambda d: d.priority, reverse=True
        )

    def set_mode(self, mode: str) -> list[Event]:
        events = []
        if mode != self.mode:
            cancelled = self.runner.cancel("zmiana trybu")
            if cancelled:
                events.append(cancelled)
            self.mode = mode
            self.actions.set_mode(mode)
            self.stop_reason = None
        return events

    def request(self, task_name: str) -> None:
        """Run a task by hand, as soon as nothing else is running."""
        self._requests.append(task_name)

    def interrupt(self, reason: str) -> list[Event]:
        """The picture stopped being live: a running task must not continue blindly."""
        cancelled = self.runner.cancel(reason)
        return [cancelled] if cancelled else []

    def emergency_stop(self, reason: str) -> list[Event]:
        events = self.set_mode(MODE_OFF)
        self.stop_reason = reason
        self._requests.clear()
        return events

    def _choose(self, result, now: float):
        for decision in self._decisions:
            if decision.task not in self._tasks:
                continue
            last = self._last_started.get(decision.name, float("-inf"))
            if (now - last) * 1000 >= decision.cooldown_ms and decision.holds(result):
                return decision
        return None

    def step(self, result, now: float, target_provider: Callable[[], tuple | None] | None = None) -> list[Event]:
        if self.mode == MODE_OFF:
            return []
        events: list[Event] = []
        if self.actions.inputs >= MAX_INPUTS_PER_ARMING:
            return self.emergency_stop(f"osiągnięto limit {MAX_INPUTS_PER_ARMING} akcji")

        decision = self._choose(result, now)
        if decision is not None and self.runner.busy:
            replaces = decision.interrupt and decision.priority > self.runner.priority
            if replaces:
                events.append(self.runner.cancel(f"wyższy priorytet: {decision.name}"))
            else:
                decision = None
        if not self.runner.busy:
            if decision is not None:
                self._last_started[decision.name] = now
                events.append(self.runner.start(self._tasks[decision.task], decision.priority))
            elif self._requests:
                task = self._tasks.get(self._requests.pop(0))
                if task is not None:
                    events.append(self.runner.start(task, 0))
        if self.runner.busy:
            rect = target_provider() if (target_provider and self.mode == MODE_ARMED) else None
            events.extend(self.runner.tick(result, now, rect))
        return events
