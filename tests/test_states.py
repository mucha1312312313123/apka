import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from colorvision.__main__ import main
from colorvision.capture import write_png
from colorvision.detector import ColorRule
from colorvision.plugin import FrameResult, Plugin, PluginHost, advance_states
from colorvision.profile import Profile, ProfileError, load_profile, save_profile
from colorvision.states import (
    MODE_ALL, MODE_ANY, OP_AT_MOST, OP_EXACTLY, Condition, StateRule, StateTracker, format_duration, missing_items,
    rename_item,
)

DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "profiles" / "default.json"
GREEN = (0, 255, 0)
SEEN = {"a": ["x"]}
NOT_SEEN = {"a": []}


def tracker_for(enter_ms=0, exit_ms=0, **kwargs):
    return StateTracker([StateRule("s", (Condition("a"),), enter_ms=enter_ms, exit_ms=exit_ms, **kwargs)])


def result_with(detections):
    return FrameResult(np.zeros((1, 1, 3), np.uint8), detections, 0.0)


@pytest.mark.parametrize(
    "condition, count, expected",
    [
        (Condition("a"), 1, True),
        (Condition("a"), 0, False),
        (Condition("a", value=3), 2, False),
        (Condition("a", OP_AT_MOST, 2), 2, True),
        (Condition("a", OP_AT_MOST, 2), 3, False),
        (Condition("a", OP_EXACTLY, 0), 0, True),
        (Condition("a", OP_EXACTLY, 0), 1, False),
    ],
)
def test_condition_counts(condition, count, expected):
    assert condition.holds({"a": ["x"] * count}) is expected


def test_unknown_item_never_holds_even_for_exactly_zero():
    assert Condition("a", OP_EXACTLY, 0).holds({}) is False


def test_all_and_any_modes():
    both = (Condition("a"), Condition("b"))
    detections = {"a": ["x"], "b": []}
    assert StateRule("s", both, MODE_ALL).holds(detections) is False
    assert StateRule("s", both, MODE_ANY).holds(detections) is True
    assert StateRule("s", ()).holds(detections) is False


def test_state_turns_on_only_after_the_delay():
    tracker = tracker_for(enter_ms=500)
    assert tracker.update(SEEN, 0.0) == []
    assert tracker.update(SEEN, 0.4) == []
    changes = tracker.update(SEEN, 0.5)
    assert [(c.name, c.active) for c in changes] == [("s", True)]
    assert tracker.snapshot(0.5)["s"].active is True


def test_one_wrong_frame_does_not_flip_the_state():
    tracker = tracker_for(enter_ms=500)
    tracker.update(SEEN, 0.0)
    tracker.update(NOT_SEEN, 0.3)  # interruption restarts the delay
    assert tracker.update(SEEN, 0.6) == []
    assert tracker.update(SEEN, 1.0) == []
    assert len(tracker.update(SEEN, 1.1)) == 1


def test_exit_delay_and_reported_duration():
    tracker = tracker_for(exit_ms=1000)
    tracker.update(SEEN, 10.0)
    assert tracker.update(NOT_SEEN, 15.0) == []
    assert tracker.update(SEEN, 15.5) == [], "brief gap keeps the state on"
    tracker.update(NOT_SEEN, 20.0)
    changes = tracker.update(NOT_SEEN, 21.0)
    assert [(c.active, c.previous_duration) for c in changes] == [(False, 11.0)]
    assert tracker.snapshot(23.0)["s"].duration == 2.0


def test_time_in_a_state_is_tracked():
    tracker = tracker_for()
    tracker.update(SEEN, 5.0)
    status = tracker.snapshot(8.5)["s"]
    assert status.active and status.duration == 3.5


def test_clear_pending_discards_unfinished_delay():
    tracker = tracker_for(enter_ms=500)
    tracker.update(SEEN, 0.0)
    tracker.clear_pending()
    assert tracker.update(SEEN, 100.0) == [], "time without frames does not count"
    assert len(tracker.update(SEEN, 100.5)) == 1


def test_set_rules_keeps_surviving_states_and_drops_others():
    tracker = StateTracker([StateRule("s", (Condition("a"),), enter_ms=0), StateRule("t", (Condition("a"),), enter_ms=0)])
    tracker.update(SEEN, 0.0)
    tracker.set_rules([StateRule("s", (Condition("a", value=1),), enter_ms=0, exit_ms=5000)])
    snapshot = tracker.snapshot(1.0)
    assert list(snapshot) == ["s"]
    assert snapshot["s"].active and snapshot["s"].duration == 1.0


def test_disabled_state_is_not_tracked():
    tracker = tracker_for(enabled=False)
    assert tracker.update(SEEN, 0.0) == []
    assert tracker.snapshot(0.0) == {}


def test_rename_and_missing_helpers():
    state = StateRule("s", (Condition("old"), Condition("other")))
    renamed = rename_item([state], "old", "new")[0]
    assert [c.item for c in renamed.conditions] == ["new", "other"]
    assert missing_items(renamed, ["new"]) == ["other"]


@pytest.mark.parametrize("seconds, text", [(0, "0:00"), (65.9, "1:05"), (3725, "1:02:05")])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


@pytest.mark.parametrize(
    "build",
    [
        lambda: Condition(""),
        lambda: Condition("a", "!="),
        lambda: Condition("a", value=-1),
        lambda: StateRule(" "),
        lambda: StateRule("s", mode="xor"),
        lambda: StateRule("s", enter_ms=-1),
    ],
)
def test_invalid_definitions_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_states_survive_save_and_load(tmp_path):
    path = tmp_path / "p.json"
    state = StateRule("Pełny ą", (Condition("a", OP_EXACTLY, 0),), MODE_ANY, 100, 200, False)
    save_profile(Profile(states=[state]), path)
    assert load_profile(path).states == [state]


def test_old_profile_without_states_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"rules": []}', encoding="utf-8")
    assert load_profile(path).states == []


@pytest.mark.parametrize(
    "content",
    [
        '{"states": [{"name": "s", "mode": "xor"}]}',
        '{"states": [{"conditions": []}]}',
        '{"states": [{"name": "s", "conditions": [{"op": ">="}]}]}',
        '{"states": [{"name": "s"}, {"name": "s"}]}',
    ],
)
def test_invalid_states_in_profile_are_reported(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)


def test_default_profile_has_example_states():
    states = {s.name: s for s in load_profile(DEFAULT_PROFILE).states}
    assert states.keys() == {"W walce", "Ekwipunek pełny"}
    assert states["W walce"].mode == MODE_ANY


def test_plugins_can_provide_conditions_and_are_told_about_changes():
    seen, states_in_frame = [], []

    class Provider(Plugin):
        name = "provider"

        def on_frame(self, result):
            states_in_frame.append(result.is_active("s"))
            result.detections["npc"] = ["x"]

        def on_state_change(self, change):
            seen.append((change.name, change.active))

    tracker = StateTracker([StateRule("s", (Condition("npc"),), enter_ms=0)])
    host = PluginHost([Provider()])
    first, second = result_with({}), result_with({})
    advance_states(first, tracker, host, 0.0)
    advance_states(second, tracker, host, 1.0)
    assert seen == [("s", True)]
    assert first.is_active("s") and second.is_active("s")
    assert states_in_frame == [False, True], "on_frame sees the previous frame's states"
    assert first.is_active("no such state") is False


def test_crashing_state_plugin_is_disabled_without_stopping_others(capsys):
    seen = []

    class Crash(Plugin):
        name = "crash"

        def on_state_change(self, change):
            raise RuntimeError("boom")

    class Fine(Plugin):
        name = "fine"

        def on_state_change(self, change):
            seen.append(change.name)

    host = PluginHost([Crash(), Fine()])
    advance_states(result_with({"a": ["x"]}), tracker_for(), host, 0.0)
    assert seen == ["s"]
    assert [p.name for p in host.plugins] == ["fine"]


def test_command_line_reports_state_changes(tmp_path, capsys):
    frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
    frame[100:105, 200:225] = GREEN
    image = tmp_path / "k.png"
    write_png(image, frame)
    profile = tmp_path / "p.json"
    save_profile(
        Profile(
            rules=[ColorRule("g", (55, 200, 150), (65, 255, 255))],
            states=[StateRule("Widac", (Condition("g"),), enter_ms=0)],
        ),
        profile,
    )
    code = main(["run", "--image", str(image), "--profile", str(profile), "--plugins", str(tmp_path / "none"),
                 "--no-preview", "--max-frames", "2", "--fps", "200"])
    assert code == 0
    assert capsys.readouterr().out.count("Stan Widac: start") == 1


# ------------------------------------------------------------------- interface

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from colorvision.gui import theme  # noqa: E402
from colorvision.gui.main_window import MainWindow  # noqa: E402
from colorvision.gui.state_dialog import StateDialog  # noqa: E402


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


def green_frame():
    frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
    frame[100:105, 200:225] = GREEN
    return frame


@pytest.fixture
def window(app, tmp_path):
    path = tmp_path / "profile.json"
    save_profile(
        Profile(
            rules=[ColorRule("green", (55, 200, 150), (65, 255, 255)), ColorRule("blue", (110, 200, 150), (130, 255, 255))],
            states=[
                StateRule("Widac", (Condition("green"),), enter_ms=0, exit_ms=0),
                StateRule("Nie ma", (Condition("green", OP_EXACTLY, 0),), enter_ms=0, exit_ms=0),
            ],
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


def state_row(window, name):
    table = window.states_table
    return next(r for r in range(table.rowCount()) if table.item(r, 0).data(Qt.UserRole) == name)


def start(window, tmp_path, frame):
    image = tmp_path / "klatka.png"
    write_png(image, frame)
    window.use_image(str(image))
    assert spin(lambda: window.last_packet is not None)


def test_states_are_listed_with_their_conditions(window):
    table = window.states_table
    assert table.rowCount() == 2
    assert table.item(state_row(window, "Widac"), 3).text() == "green: co najmniej 1"
    assert table.item(state_row(window, "Widac"), 1).text() == "–"


def test_live_state_shows_in_table_history_status_and_log(window, app, tmp_path):
    start(window, tmp_path, green_frame())
    table = window.states_table
    row = state_row(window, "Widac")
    assert spin(lambda: table.item(row, 1).text() == "Aktywny")
    assert table.item(state_row(window, "Nie ma"), 1).text() == "Nieaktywny"
    assert window.active_states_label.text() == "Stany: Widac"
    assert spin(lambda: window.states_history.count() >= 1)
    assert "Widac: start" in window.states_history.item(0).text()
    assert "Stan Widac: start" in window.log_view.toPlainText()

    window.start_action.setChecked(False)
    assert spin(lambda: window.worker is None)
    assert table.item(row, 1).text() == "–"
    assert window.active_states_label.text() == ""


def test_editing_a_state_applies_while_running(window, app, tmp_path):
    start(window, tmp_path, green_frame())
    table = window.states_table
    assert spin(lambda: table.item(state_row(window, "Widac"), 1).text() == "Aktywny")
    table.setCurrentCell(state_row(window, "Widac"), 0)
    window._ask_state = lambda state: StateRule(state.name, (Condition("blue"),), enter_ms=0, exit_ms=0)
    window._edit_state()
    assert window.dirty
    assert "blue" in table.item(state_row(window, "Widac"), 3).text()
    assert spin(lambda: table.item(state_row(window, "Widac"), 1).text() == "Nieaktywny")


def test_add_duplicate_delete_and_disable(window):
    window._ask_state = lambda state: StateRule("Nowy", (Condition("green"),))
    window._add_state()
    assert [s.name for s in window.profile.states][-1] == "Nowy"

    window.states_table.setCurrentCell(state_row(window, "Nowy"), 0)
    window._duplicate_state()
    assert "Nowy kopia" in [s.name for s in window.profile.states]

    window.states_table.setCurrentCell(state_row(window, "Nowy"), 0)
    window._delete_state()
    assert "Nowy" not in [s.name for s in window.profile.states]

    item = window.states_table.item(state_row(window, "Widac"), 0)
    item.setCheckState(Qt.Unchecked)
    assert next(s for s in window.profile.states if s.name == "Widac").enabled is False


def test_cancelled_dialog_changes_nothing(window):
    window._ask_state = lambda state: None
    window._add_state()
    assert len(window.profile.states) == 2
    assert not window.dirty


def test_renaming_a_rule_updates_conditions(window):
    window.rules_tree.clearSelection()
    window.rules_tree.topLevelItem(0).setSelected(True)
    window.name_edit.setText("zielony")
    window._on_name_edited()
    assert window.profile.states[0].conditions[0].item == "zielony"
    assert "zielony" in window.states_table.item(0, 3).text()


def test_deleting_a_rule_warns_and_marks_the_state(window):
    questions = []
    window._confirm = lambda title, text: questions.append(text) or True
    window.rules_tree.clearSelection()
    window.rules_tree.topLevelItem(0).setSelected(True)
    window._delete_rule()
    assert "Widac" in questions[0]
    assert "⚠ brak: green" in window.states_table.item(state_row(window, "Widac"), 3).text()


def test_states_saved_with_the_profile(window):
    window._ask_state = lambda state: StateRule("Nowy", (Condition("blue"),))
    window._add_state()
    assert window._save()
    assert "Nowy" in [s.name for s in load_profile(window.profile_path).states]


def test_pause_freezes_state_timing(window, app, tmp_path):
    start(window, tmp_path, green_frame())
    assert spin(lambda: window.states_history.count() >= 1)
    window.pause_action.setChecked(True)
    assert spin(lambda: window._state == "paused")
    count = window.states_history.count()
    window._ask_state = lambda state: StateRule("Nowy", (Condition("green"),), enter_ms=0)
    window._add_state()
    assert not spin(lambda: window.states_history.count() > count, timeout_ms=300), "held frame must not change states"
    window.pause_action.setChecked(False)
    assert spin(lambda: window.states_history.count() > count)


def test_dialog_builds_a_state(app):
    dialog = StateDialog(None, ["green", "blue"], {"Zajete"})
    dialog.name_edit.setText("  Walka ")
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(MODE_ANY))
    dialog.enter_spin.setValue(100)
    dialog.exit_spin.setValue(2000)
    dialog.rows[0].item_combo.setEditText("blue")
    dialog.rows[0].op_combo.setCurrentIndex(dialog.rows[0].op_combo.findData(OP_EXACTLY))
    dialog.rows[0].value_spin.setValue(0)
    dialog.add_row(Condition("wynik_pluginu", value=2))
    dialog.accept()
    assert dialog.result_rule == StateRule(
        "Walka", (Condition("blue", OP_EXACTLY, 0), Condition("wynik_pluginu", value=2)), MODE_ANY, 100, 2000
    )


def test_dialog_keeps_values_of_an_existing_state(app):
    state = StateRule("S", (Condition("usunieta", OP_AT_MOST, 4),), MODE_ANY, 111, 222, enabled=False)
    dialog = StateDialog(state, ["green"], set())
    assert dialog.rows[0].item_combo.currentText() == "usunieta"
    dialog.accept()
    assert dialog.result_rule == state


@pytest.mark.parametrize(
    "name, prepare, message",
    [
        ("", lambda d: None, "nazwę"),
        ("Zajete", lambda d: None, "istnieje"),
        ("Ok", lambda d: d.rows[0].item_combo.setEditText(" "), "warunek"),
        ("Ok", lambda d: d.remove_row(d.rows[0]), "warunek"),
    ],
)
def test_dialog_refuses_invalid_input(app, name, prepare, message):
    dialog = StateDialog(None, ["green"], {"Zajete"})
    dialog.name_edit.setText(name)
    prepare(dialog)
    dialog.accept()
    assert dialog.result_rule is None
    assert message in dialog.error_label.text()
