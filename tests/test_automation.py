import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from colorvision.automation import (
    EVENT_ACTION, EVENT_TASK_DONE, EVENT_TASK_FAILED, EVENT_TASK_START, MAX_INPUTS_PER_ARMING, MIN_INPUT_GAP, MODE_ARMED,
    MODE_OFF, MODE_SIMULATE, STEP_CLICK, STEP_KEY, STEP_LOG, STEP_WAIT, STEP_WAIT_FOR, Automation, Decision, Step, Task,
)
from colorvision.detector import Detection
from colorvision.input_win import virtual_key
from colorvision.plugin import FrameResult
from colorvision.profile import Profile, ProfileError, load_profile, save_profile
from colorvision.states import Condition, StateStatus

DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "profiles" / "default.json"
RECT = (1000, 2000, 765, 503)


class Recorder:
    def __init__(self):
        self.calls = []

    def click(self, x, y, button):
        self.calls.append(("click", x, y, button))

    def press(self, key):
        self.calls.append(("press", key))


def seen(rule="npc", x=100, y=50, count=1):
    return {rule: [Detection(rule, x, y, 10, 10, 100, (x + 5, y + 5)) for _ in range(count)]}


def result(detections=None, active=(), inactive=()):
    frame = np.zeros((503, 765, 3), np.uint8)
    states = {n: StateStatus(n, True, 1.0) for n in active} | {n: StateStatus(n, False, 1.0) for n in inactive}
    return FrameResult(frame, detections if detections is not None else {}, 0.0, states=states)


def automation(tasks, decisions=(), mode=MODE_SIMULATE):
    recorder = Recorder()
    auto = Automation(Profile(tasks=list(tasks), decisions=list(decisions)), lambda: recorder)
    auto.set_mode(mode)
    return auto, recorder


CLICK_TASK = Task("klik", (Step(STEP_CLICK, target="npc"),))
ON_NPC = Decision("gdy npc", "klik", conditions=(Condition("npc"),), cooldown_ms=0)
target = lambda: RECT


def kinds(events):
    return [e.kind for e in events]


# ------------------------------------------------------------------- safety

def test_automation_is_off_by_default_and_does_nothing():
    recorder = Recorder()
    auto = Automation(Profile(tasks=[CLICK_TASK], decisions=[ON_NPC]), lambda: recorder)
    assert auto.mode == MODE_OFF
    assert auto.step(result(seen()), 0.0, target) == []
    assert recorder.calls == []


def test_simulation_never_touches_real_input():
    auto, recorder = automation([CLICK_TASK], [ON_NPC])
    events = auto.step(result(seen()), 0.0, target)
    assert recorder.calls == []
    assert any("Symulacja" in e.message for e in events)


def test_armed_click_lands_at_object_centre_in_screen_coordinates():
    auto, recorder = automation([CLICK_TASK], [ON_NPC], MODE_ARMED)
    auto.step(result(seen(x=100, y=50)), 0.0, target)
    assert recorder.calls == [("click", 1105, 2055, "left")]


def test_armed_input_is_refused_when_client_is_not_active():
    auto, recorder = automation([CLICK_TASK], [ON_NPC], MODE_ARMED)
    events = auto.step(result(seen()), 0.0, lambda: None)
    assert recorder.calls == []
    assert EVENT_TASK_FAILED in kinds(events)


def test_armed_without_a_window_source_is_refused():
    auto, recorder = automation([CLICK_TASK], [ON_NPC], MODE_ARMED)
    auto.step(result(seen()), 0.0, None)
    assert recorder.calls == []


def test_click_outside_the_client_is_refused():
    auto, recorder = automation([CLICK_TASK], [ON_NPC], MODE_ARMED)
    events = auto.step(result(seen(x=5000, y=50)), 0.0, target)
    assert recorder.calls == []
    assert EVENT_TASK_FAILED in kinds(events)


def test_inputs_are_rate_limited():
    task = Task("dwa", (Step(STEP_KEY, key="a"), Step(STEP_KEY, key="b")))
    auto, recorder = automation([task], [Decision("d", "dwa", conditions=(Condition("npc"),), cooldown_ms=9999)], MODE_ARMED)
    auto.step(result(seen()), 0.0, target)
    assert recorder.calls == [("press", "a")], "one input per frame"
    auto.step(result(seen()), MIN_INPUT_GAP / 2, target)
    assert recorder.calls == [("press", "a")], "too soon"
    auto.step(result(seen()), MIN_INPUT_GAP, target)
    assert recorder.calls == [("press", "a"), ("press", "b")]


def test_limit_of_inputs_switches_control_off():
    auto, recorder = automation([CLICK_TASK], [ON_NPC], MODE_ARMED)
    auto.actions.inputs = MAX_INPUTS_PER_ARMING
    auto.step(result(seen()), 0.0, target)
    assert auto.mode == MODE_OFF
    assert "limit" in auto.stop_reason
    assert recorder.calls == []


def test_emergency_stop_cancels_the_task_and_turns_everything_off():
    task = Task("dlugie", (Step(STEP_WAIT, ms=60000), Step(STEP_KEY, key="a")))
    auto, recorder = automation([task], [Decision("d", "dlugie", conditions=(Condition("npc"),))], MODE_ARMED)
    auto.step(result(seen()), 0.0, target)
    events = auto.emergency_stop("naciśnięto F12")
    assert EVENT_TASK_FAILED in kinds(events)
    assert auto.mode == MODE_OFF and not auto.runner.busy
    assert auto.step(result(seen()), 100.0, target) == []
    assert recorder.calls == []


def test_interrupt_cancels_a_running_task():
    task = Task("dlugie", (Step(STEP_WAIT, ms=60000), Step(STEP_KEY, key="a")))
    auto, recorder = automation([task], [Decision("d", "dlugie", conditions=(Condition("npc"),))], MODE_ARMED)
    auto.step(result(seen()), 0.0, target)
    assert kinds(auto.interrupt("pauza")) == [EVENT_TASK_FAILED]
    assert auto.interrupt("pauza") == []


def test_f12_cannot_be_used_as_a_task_key():
    with pytest.raises(ValueError):
        Step(STEP_KEY, key="f12")


@pytest.mark.parametrize("key, code", [("a", 0x41), ("7", 0x37), ("f1", 0x70), ("f11", 0x7A), ("space", 0x20), ("left", 0x25)])
def test_virtual_key_codes(key, code):
    assert virtual_key(key) == code


# ------------------------------------------------------------------- task manager

def test_steps_run_in_order_with_waits():
    task = Task("t", (Step(STEP_LOG, message="a"), Step(STEP_WAIT, ms=500), Step(STEP_LOG, message="b")))
    auto, _ = automation([task], [Decision("d", "t", conditions=(Condition("npc"),), cooldown_ms=99999)])
    first = auto.step(result(seen()), 0.0, target)
    assert [e.message for e in first if e.kind == EVENT_ACTION] == ["a"]
    assert auto.step(result(seen()), 0.4, target) == []
    later = auto.step(result(seen()), 0.5, target)
    assert [e.message for e in later if e.kind == EVENT_ACTION] == ["b"]
    assert EVENT_TASK_DONE in kinds(later)
    assert not auto.runner.busy


def test_wait_for_continues_when_the_object_appears():
    task = Task("t", (Step(STEP_WAIT_FOR, target="drzwi", ms=1000), Step(STEP_LOG, message="jest")))
    auto, _ = automation([task])
    auto.request("t")
    assert EVENT_TASK_DONE not in kinds(auto.step(result({"drzwi": []}), 0.0))
    assert EVENT_TASK_DONE in kinds(auto.step(result(seen("drzwi")), 0.5))


def test_wait_for_fails_after_its_time_limit():
    task = Task("t", (Step(STEP_WAIT_FOR, target="drzwi", ms=1000),))
    auto, _ = automation([task])
    auto.request("t")
    auto.step(result({"drzwi": []}), 0.0)
    assert auto.runner.busy
    events = auto.step(result({"drzwi": []}), 1.0)
    assert EVENT_TASK_FAILED in kinds(events)


def test_missing_click_target_fails_the_task():
    auto, _ = automation([CLICK_TASK])
    auto.request("klik")
    events = auto.step(result({}), 0.0, target)
    assert kinds(events) == [EVENT_TASK_START, EVENT_TASK_FAILED]


def test_manual_request_runs_only_a_known_task():
    auto, _ = automation([Task("t", (Step(STEP_LOG, message="x"),))])
    auto.request("nie ma")
    assert auto.step(result(), 0.0) == []
    auto.request("t")
    assert EVENT_TASK_DONE in kinds(auto.step(result(), 1.0))


def test_changing_mode_cancels_a_running_task():
    task = Task("t", (Step(STEP_WAIT, ms=60000),))
    auto, _ = automation([task])
    auto.request("t")
    auto.step(result(), 0.0)
    assert kinds(auto.set_mode(MODE_ARMED)) == [EVENT_TASK_FAILED]


# ------------------------------------------------------------------- decision engine

LOG = lambda name: Task(name, (Step(STEP_LOG, message=name),))


def started(events):
    return [e.source for e in events if e.kind == EVENT_TASK_START]


def test_highest_priority_decision_wins():
    decisions = [
        Decision("niska", "a", priority=1, conditions=(Condition("npc"),)),
        Decision("wysoka", "b", priority=5, conditions=(Condition("npc"),)),
    ]
    auto, _ = automation([LOG("a"), LOG("b")], decisions)
    assert started(auto.step(result(seen()), 0.0)) == ["b"]


def test_state_conditions():
    decision = Decision("d", "a", when_state="Walka", unless_state="Pauza", cooldown_ms=0)
    assert decision.holds(result(active=["Walka"], inactive=["Pauza"]))
    assert not decision.holds(result(inactive=["Walka", "Pauza"]))
    assert not decision.holds(result(active=["Walka", "Pauza"]))
    assert not decision.holds(result(active=["Walka"])), "an unknown state is not treated as off"


def test_cooldown_limits_restarts():
    auto, _ = automation([LOG("a")], [Decision("d", "a", conditions=(Condition("npc"),), cooldown_ms=1000)])
    assert started(auto.step(result(seen()), 0.0)) == ["a"]
    assert started(auto.step(result(seen()), 0.5)) == []
    assert started(auto.step(result(seen()), 1.0)) == ["a"]


def test_running_task_is_replaced_only_by_higher_priority_with_interrupt():
    long_task = Task("dlugie", (Step(STEP_WAIT, ms=60000),))
    low = Decision("niska", "dlugie", priority=1, conditions=(Condition("a"),), cooldown_ms=99999)
    polite = Decision("grzeczna", "x", priority=5, conditions=(Condition("b"),), cooldown_ms=0)
    urgent = Decision("pilna", "y", priority=9, interrupt=True, conditions=(Condition("c"),), cooldown_ms=0)
    auto, _ = automation([long_task, LOG("x"), LOG("y")], [low, polite, urgent])
    auto.step(result(seen("a")), 0.0)
    assert auto.runner.task.name == "dlugie"
    assert started(auto.step(result(seen("b")), 1.0)) == []
    events = auto.step(result(seen("c")), 2.0)
    assert EVENT_TASK_FAILED in kinds(events)
    assert started(events) == ["y"]


def test_disabled_decision_is_ignored_and_edits_apply():
    decision = Decision("d", "a", conditions=(Condition("npc"),), enabled=False)
    auto, _ = automation([LOG("a")], [decision])
    assert auto.step(result(seen()), 0.0) == []
    auto.set_profile(Profile(tasks=[LOG("a")], decisions=[Decision("d", "a", conditions=(Condition("npc"),))]))
    assert started(auto.step(result(seen()), 1.0)) == ["a"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Step("skok"),
        lambda: Step(STEP_CLICK),
        lambda: Step(STEP_CLICK, target="a", button="srodkowy"),
        lambda: Step(STEP_KEY, key="żaden"),
        lambda: Step(STEP_WAIT, ms=-1),
        lambda: Step(STEP_WAIT_FOR, target="a", op="!="),
        lambda: Task(" "),
        lambda: Decision("d", "t"),
        lambda: Decision("", "t", when_state="s"),
        lambda: Decision("d", "t", when_state="s", cooldown_ms=-1),
    ],
)
def test_invalid_definitions_are_rejected(build):
    with pytest.raises(ValueError):
        build()


# ------------------------------------------------------------------- profiles

def full_profile():
    task = Task("t", (
        Step(STEP_CLICK, target="npc", button="right"), Step(STEP_KEY, key="f1"), Step(STEP_WAIT, ms=250),
        Step(STEP_WAIT_FOR, target="npc", ms=900, op="==", value=0), Step(STEP_LOG, message="Żółć"),
    ))
    decision = Decision("d", "t", 3, "A", "B", (Condition("npc", "<=", 2),), 750, True, False)
    return Profile(tasks=[task], decisions=[decision])


def test_tasks_and_decisions_survive_save_and_load(tmp_path):
    path = tmp_path / "p.json"
    profile = full_profile()
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.tasks == profile.tasks
    assert loaded.decisions == profile.decisions
    assert loaded.copy().tasks == profile.tasks


def test_default_profile_only_logs():
    profile = load_profile(DEFAULT_PROFILE)
    assert all(step.kind == STEP_LOG for task in profile.tasks for step in task.steps)
    assert [d.when_state for d in profile.decisions] == ["Ekwipunek pełny"]


@pytest.mark.parametrize(
    "content",
    [
        '{"tasks": [{"steps": []}]}',
        '{"tasks": [{"name": "t", "steps": [{"kind": "skok"}]}]}',
        '{"tasks": [{"name": "t", "steps": [{"kind": "log", "nieznane": 1}]}]}',
        '{"tasks": [{"name": "t"}, {"name": "t"}]}',
        '{"decisions": [{"name": "d", "task": "brak", "when_state": "s"}]}',
        '{"tasks": [{"name": "t"}], "decisions": [{"name": "d", "task": "t"}]}',
    ],
)
def test_invalid_automation_in_profile_is_reported(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)


# ------------------------------------------------------------------- interface

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from colorvision.capture import write_png  # noqa: E402
from colorvision.gui import theme  # noqa: E402
from colorvision.gui.automation_dialogs import DecisionDialog, TaskDialog  # noqa: E402
from colorvision.gui.main_window import MainWindow  # noqa: E402
from colorvision.detector import ColorRule  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.STYLESHEET)
    return application


def spin(condition, timeout_ms=5000):
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return condition()


@pytest.fixture
def window(app, tmp_path):
    path = tmp_path / "profile.json"
    save_profile(
        Profile(
            rules=[ColorRule("green", (50, 200, 150), (70, 255, 255))],
            tasks=[Task("klik", (Step(STEP_CLICK, target="green"),))],
            decisions=[Decision("gdy widac", "klik", conditions=(Condition("green"),), cooldown_ms=200)],
        ),
        path,
    )
    win = MainWindow(settings_path=tmp_path / "s.ini", plugins_dir=tmp_path / "none", profile_path=path)
    win.warnings = []
    win._warn = lambda title, text: win.warnings.append(title)
    win._confirm = lambda title, text: True
    win.show()
    app.processEvents()
    yield win
    if win.worker is not None:
        win.worker.requestInterruption()
        win.worker.wait(3000)
    win.dirty = False
    win.close()


def choose_mode(window, mode):
    window.automation_combo.setCurrentIndex(window.automation_combo.findData(mode))
    window._on_automation_mode_chosen()


def start_image(window, tmp_path):
    frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
    frame[100:105, 200:225] = (0, 255, 0)
    image = tmp_path / "k.png"
    write_png(image, frame)
    window.use_image(str(image))
    assert spin(lambda: window.last_packet is not None)


def event_messages(window):
    table = window.events_table
    return [table.item(r, 3).text() for r in range(table.rowCount())]


def test_window_starts_with_automation_off(window):
    assert window._automation_mode() == MODE_OFF


def test_lists_show_tasks_and_decisions(window):
    assert window.tasks_table.item(0, 0).text() == "klik"
    assert window.tasks_table.item(0, 1).text() == "Kliknij obiekt"
    assert window.decisions_table.item(0, 2).text() == "klik"


def test_simulation_runs_decisions_and_shows_events(window, tmp_path):
    start_image(window, tmp_path)
    assert not any("Symulacja" in m for m in event_messages(window)), "nothing happens until a mode is chosen"
    choose_mode(window, MODE_SIMULATE)
    assert spin(lambda: any("Symulacja: klik lewy w green" in m for m in event_messages(window)))
    assert spin(lambda: any("ukończone" in m for m in event_messages(window)))


def test_wait_as_the_last_step_is_really_waited_for():
    auto, _ = automation([Task("t", (Step(STEP_WAIT, ms=1000),))])
    auto.request("t")
    auto.step(result(), 0.0)
    assert auto.runner.busy
    assert auto.step(result(), 0.9) == []
    assert kinds(auto.step(result(), 1.0)) == [EVENT_TASK_DONE]


def test_armed_mode_needs_confirmation(window):
    window._confirm = lambda title, text: False
    choose_mode(window, MODE_ARMED)
    assert window._automation_mode() == MODE_OFF


def test_armed_is_refused_for_a_saved_image(window, tmp_path):
    start_image(window, tmp_path)
    choose_mode(window, MODE_ARMED)
    assert window._automation_mode() == MODE_OFF
    assert window.warnings == ["Prawdziwe sterowanie"]


def test_armed_shows_a_warning_banner_and_stop_turns_it_off(window, app):
    window.window_combo.setEditText("OKNO_KTOREGO_NIE_MA_98765")
    window._start_worker()
    choose_mode(window, MODE_ARMED)
    assert "PRAWDZIWE STEROWANIE" in window.banner.text()
    window.start_action.setChecked(False)
    assert spin(lambda: window.worker is None)
    assert window._automation_mode() == MODE_OFF
    assert "PRAWDZIWE" not in window.banner.text()


def test_worker_reports_a_self_stop(window):
    window.window_combo.setEditText("OKNO_KTOREGO_NIE_MA_98765")
    window._start_worker()
    choose_mode(window, MODE_SIMULATE)
    window._on_automation_stopped("naciśnięto F12")
    assert window._automation_mode() == MODE_OFF
    assert "F12" in window.log_view.toPlainText()


def test_manual_run_needs_a_running_mode(window, tmp_path):
    window.tasks_table.setCurrentCell(0, 0)
    window._run_selected_task()
    assert window.warnings == ["Uruchom zadanie"]
    start_image(window, tmp_path)
    window.profile.decisions.clear()
    window.worker.set_profile(window.profile)
    choose_mode(window, MODE_SIMULATE)
    window.tasks_table.setCurrentCell(0, 0)
    window.tasks_table.selectRow(0)
    window._run_selected_task()
    assert spin(lambda: any("Zadanie klik: start" in m for m in event_messages(window)))


def test_task_in_use_cannot_be_deleted_and_rename_follows(window):
    window.tasks_table.setCurrentCell(0, 0)
    window.tasks_table.selectRow(0)
    window._delete_task()
    assert window.warnings == ["Zadanie jest używane"]
    window._ask_task = lambda task: Task("nowa", task.steps)
    window._edit_task()
    assert window.profile.decisions[0].task == "nowa"


def test_renaming_a_rule_updates_tasks_and_decisions(window):
    window.rules_tree.clearSelection()
    window.rules_tree.topLevelItem(0).setSelected(True)
    window.name_edit.setText("zielony")
    window._on_name_edited()
    assert window.profile.tasks[0].steps[0].target == "zielony"
    assert window.profile.decisions[0].conditions[0].item == "zielony"


def test_add_edit_delete_and_save(window):
    window._ask_task = lambda task: Task("drugie", (Step(STEP_LOG, message="x"),))
    window._add_task()
    window._ask_decision = lambda d: Decision("nowa", "drugie", when_state="S")
    window._add_decision()
    assert window._save()
    loaded = load_profile(window.profile_path)
    assert [t.name for t in loaded.tasks] == ["klik", "drugie"]
    assert [d.name for d in loaded.decisions] == ["gdy widac", "nowa"]

    window.decisions_table.setCurrentCell(1, 0)
    window.decisions_table.selectRow(1)
    window._delete_decision()
    assert [d.name for d in window.profile.decisions] == ["gdy widac"]
    window.decisions_table.item(0, 0).setCheckState(Qt.Unchecked)
    assert window.profile.decisions[0].enabled is False


def test_decision_needs_a_task_first(window):
    window.profile.tasks.clear()
    window._add_decision()
    assert window.warnings == ["Brak zadań"]


def test_task_dialog_builds_every_step_kind(app):
    dialog = TaskDialog(None, ["green"], {"zajete"})
    dialog.name_edit.setText(" Zadanie ")
    expected = [
        Step(STEP_LOG, message="hej"), Step(STEP_WAIT, ms=500), Step(STEP_WAIT_FOR, target="green", ms=900, op="==", value=0),
        Step(STEP_CLICK, target="green", button="right"), Step(STEP_KEY, key="f1"),
    ]
    dialog.remove_row(dialog.rows[0])
    for step in expected:
        dialog.add_row(step)
    dialog.accept()
    assert dialog.result_task == Task("Zadanie", tuple(expected))


@pytest.mark.parametrize(
    "name, prepare, fragment",
    [
        ("", lambda d: None, "nazwę"),
        ("zajete", lambda d: None, "istnieje"),
        ("ok", lambda d: d.remove_row(d.rows[0]), "krok"),
    ],
)
def test_task_dialog_refuses_invalid_input(app, name, prepare, fragment):
    dialog = TaskDialog(None, ["green"], {"zajete"})
    dialog.name_edit.setText(name)
    prepare(dialog)
    dialog.accept()
    assert dialog.result_task is None
    assert fragment in dialog.error_label.text()


def test_click_step_without_target_is_refused(app):
    dialog = TaskDialog(None, [], set())
    dialog.name_edit.setText("ok")
    row = dialog.rows[0]
    row.kind_combo.setCurrentIndex(row.kind_combo.findData(STEP_CLICK))
    dialog.accept()
    assert dialog.result_task is None


def test_decision_dialog_builds_and_round_trips(app):
    decision = Decision("d", "b", 3, "S", None, (Condition("green", "==", 0),), 700, True, False)
    dialog = DecisionDialog(decision, ["a", "b"], ["S", "T"], ["green"], set())
    dialog.accept()
    assert dialog.result_decision == decision


def test_decision_dialog_requires_a_condition(app):
    dialog = DecisionDialog(None, ["a"], ["S"], ["green"], set())
    dialog.name_edit.setText("d")
    dialog.accept()
    assert dialog.result_decision is None
    assert "stan" in dialog.error_label.text()
